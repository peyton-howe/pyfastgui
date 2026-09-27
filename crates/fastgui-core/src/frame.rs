use std::sync::{Arc, Mutex};

/// Pixel layout of a `CpuFrame`. RGBA8 only for now — enough to cover numpy/buffer-protocol
/// sources; more formats (e.g. planar YUV from real camera capture) can be added later
/// without changing how frames are submitted or consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8,
}

/// A single CPU-side image, ready to be uploaded to a GPU texture.
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

/// An integer pixel rect (physical pixels, top-left origin) — a damaged region of a chrome frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelRect {
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Smallest rect containing both (an empty rect contributes nothing).
    pub fn union(&self, other: &PixelRect) -> PixelRect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let (x, y) = (self.x.min(other.x), self.y.min(other.y));
        PixelRect {
            x,
            y,
            width: self.right().max(other.right()) - x,
            height: self.bottom().max(other.bottom()) - y,
        }
    }

    pub fn intersects(&self, other: &PixelRect) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && self.x < other.right()
            && other.x < self.right()
            && self.y < other.bottom()
            && other.y < self.bottom()
    }
}

/// The chrome (widget UI) image for one window, borrowed from `fastgui-chrome`'s retained
/// buffer. `data` is always the *whole* tightly packed RGBA8 frame; `damage` says which parts
/// changed since the previous `ChromeFrame` so a backend whose texture already holds that
/// previous frame can upload just those rects. `None` means everything changed — and a backend
/// that has no texture of this size yet must upload all of `data` regardless.
pub struct ChromeFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
    pub damage: Option<&'a [PixelRect]>,
}

/// `ChromeQuad::kind`: a solid rect. `rect` is its exact (fractional) edges; the shader gives
/// edge pixels their covered fraction, the same box-filter coverage tiny-skia's CPU fill uses.
pub const QUAD_SOLID: u32 = 0;
/// `ChromeQuad::kind`: a filled circle, `params = [cx, cy, radius, _]`; `rect` is its bounds.
pub const QUAD_CIRCLE: u32 = 1;
/// `ChromeQuad::kind`: premultiplied RGBA pixels from the chrome atlas, copied 1:1. `rect` is
/// the whole-pixel destination and `params = [atlas_x, atlas_y, _, _]` its source origin.
pub const QUAD_SPRITE: u32 = 2;

/// One instance of the chrome's instanced quad draw, in physical pixels, top-left origin.
/// `#[repr(C)]` and 64 bytes: backends upload the slice as-is as instance data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChromeQuad {
    /// `[left, top, right, bottom]`.
    pub rect: [f32; 4],
    /// Straight (not premultiplied) RGBA; unused for sprites.
    pub color: [f32; 4],
    pub params: [f32; 4],
    pub kind: u32,
    pub _pad: [u32; 3],
}

/// Pixels to write into the chrome atlas texture before drawing: `pixels` is premultiplied
/// RGBA8, tightly packed (`rect.width * 4` bytes per row).
pub struct AtlasUpload<'a> {
    pub rect: PixelRect,
    pub pixels: &'a [u8],
}

/// The chrome as GPU draw data: quads in paint order over one `atlas_size`² RGBA8 atlas.
/// When `atlas_size` differs from the backend's current atlas (first frame, growth), the backend
/// makes a fresh atlas; `atlas_uploads` then covers every sprite `quads` uses.
pub struct ChromeQuads<'a> {
    pub width: u32,
    pub height: u32,
    pub quads: &'a [ChromeQuad],
    pub atlas_size: u32,
    /// The atlas was repacked from scratch, so uploads may overwrite slots that quads already
    /// in flight on the GPU still sample: wait for the GPU before writing them.
    pub atlas_repacked: bool,
    pub atlas_uploads: Vec<AtlasUpload<'a>>,
}

/// A single "latest frame wins" mailbox: a fast producer thread submits frames without ever
/// blocking on (or queueing behind) the render thread's cadence, and the render thread picks
/// up whatever is newest — dropping any frame that arrived and was overwritten before it got
/// consumed. This is deliberately separate from the `command_channel` used for widget
/// mutations: those are applied in order and must not be dropped, frames are the opposite.
pub struct FrameSlot<T>(Arc<Mutex<Option<T>>>);

// Manual impl: `#[derive(Clone)]` would add a spurious `T: Clone` bound, even though cloning
// a `FrameSlot` only ever clones the `Arc` handle, never a `T`.
impl<T> Clone for FrameSlot<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> FrameSlot<T> {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    pub fn submit(&self, frame: T) {
        *self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(frame);
    }

    /// Take the newest submitted frame, if any arrived since the last call.
    pub fn take_latest(&self) -> Option<T> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
    }
}

impl<T> Default for FrameSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_wins_and_take_clears() {
        let slot = FrameSlot::new();
        assert!(slot.take_latest().is_none());
        slot.submit(1);
        slot.submit(2);
        slot.submit(3);
        assert_eq!(slot.take_latest(), Some(3));
        assert!(slot.take_latest().is_none());
    }

    #[test]
    fn cloned_handles_share_mailbox() {
        let slot = FrameSlot::new();
        let other = slot.clone();
        slot.submit("a");
        assert_eq!(other.take_latest(), Some("a"));
        other.submit("b");
        assert_eq!(slot.take_latest(), Some("b"));
    }
}
