use std::any::Any;
use std::sync::{Arc, Mutex};

/// Pixel layout of a `CpuFrame`. RGBA8 only for now — enough to cover numpy/buffer-protocol
/// sources; more formats (e.g. planar YUV from real camera capture) can be added later
/// without changing how frames are submitted or consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8,
}

/// Soft upper bound for a CPU `Image` / `Viewport` frame edge. Larger uploads exceed common
/// GPU `maxImageDimension2D` limits (often 16384) and produce validation errors while still
/// leaving the app running.
pub const MAX_CPU_FRAME_EXTENT: u32 = 16384;

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

    /// The overlap of two rects (empty when they don't overlap).
    pub fn intersect(&self, other: &PixelRect) -> PixelRect {
        let (x, y) = (self.x.max(other.x), self.y.max(other.y));
        let (right, bottom) = (self.right().min(other.right()), self.bottom().min(other.bottom()));
        PixelRect { x, y, width: right.saturating_sub(x), height: bottom.saturating_sub(y) }
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
/// A circle cut by a clip rect (inside a `ScrollArea`): `rect` is the clipped extent and the
/// shader multiplies circle coverage by `rect`'s box coverage.
pub const QUAD_CIRCLE_CLIPPED: u32 = 3;

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
    /// Quads from here on are popups (overlays): backends draw them after `Viewport` layers,
    /// the rest before. Equal to `quads.len()` when no popup is open.
    pub overlay_start: usize,
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

    /// Drop a pending frame. A CPU submit and an imported submit share a layer, and each one
    /// clears the other so the render thread does not upload bytes it is about to replace.
    pub fn clear(&self) {
        *self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }
}

/// A GPU resource the viewport samples directly, instead of copying a `CpuFrame`.
///
/// `Metal` carries an already-retained backend object in `retained` (`fastgui-render-mtl`
/// downcasts it). `DmaBuf` owns a duplicated file descriptor; dropping it closes that fd unless
/// [`DmaBufImport::take_fd`] has handed the fd to the driver.
pub enum ImportedLayer {
    Metal {
        /// `id<MTLTexture>`, retained inside `retained`.
        texture: usize,
        width: u32,
        height: u32,
        /// `id<MTLEvent>`, or 0 when this submission does not wait.
        wait_event: usize,
        wait_value: u64,
        /// `id<MTLEvent>`, or 0 when this submission does not signal.
        signal_event: usize,
        signal_value: u64,
        retained: Box<dyn Any + Send>,
    },
    DmaBuf(DmaBufImport),
}

/// Linear RGBA8 dma-buf (or opaque fd) to import as a sampled image. `modifier == 0` means
/// linear, with no DRM format modifier. The first cut does not accept NV12.
#[derive(Debug)]
pub struct DmaBufImport {
    fd: i32,
    pub width: u32,
    pub height: u32,
    /// Bytes from the start of one row to the next. `0` means tightly packed (`width * 4`).
    pub stride: u32,
    pub modifier: u64,
}

impl DmaBufImport {
    pub fn new(fd: i32, width: u32, height: u32, stride: u32, modifier: u64) -> Self {
        Self { fd, width, height, stride, modifier }
    }

    pub fn fd(&self) -> i32 {
        self.fd
    }

    /// The fd the driver is about to consume. Later drops do not close it.
    pub fn take_fd(&mut self) -> i32 {
        let fd = self.fd;
        self.fd = -1;
        fd
    }
}

impl Drop for DmaBufImport {
    fn drop(&mut self) {
        if self.fd >= 0 {
            close_fd(self.fd);
            self.fd = -1;
        }
    }
}

impl std::fmt::Debug for ImportedLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportedLayer::Metal {
                texture,
                width,
                height,
                wait_event,
                wait_value,
                signal_event,
                signal_value,
                ..
            } => f
                .debug_struct("Metal")
                .field("texture", texture)
                .field("width", width)
                .field("height", height)
                .field("wait_event", wait_event)
                .field("wait_value", wait_value)
                .field("signal_event", signal_event)
                .field("signal_value", signal_value)
                .finish(),
            ImportedLayer::DmaBuf(buf) => f.debug_tuple("DmaBuf").field(buf).finish(),
        }
    }
}

fn close_fd(fd: i32) {
    #[cfg(unix)]
    {
        extern "C" {
            fn close(fd: i32) -> i32;
        }
        unsafe { close(fd) };
    }
    #[cfg(not(unix))]
    {
        let _ = fd;
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
        slot.submit(4);
        slot.clear();
        assert!(slot.take_latest().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn dmabuf_drop_closes_fd_and_take_fd_disarms_it() {
        use std::io::{ErrorKind, Read};
        use std::os::fd::{FromRawFd, OwnedFd};

        let (a, mut b) = std::os::unix::net::UnixStream::pair().unwrap();
        b.set_nonblocking(true).unwrap();
        let mut byte = [0];
        let raw = std::os::fd::IntoRawFd::into_raw_fd(a);
        let mut import = DmaBufImport::new(raw, 2, 3, 8, 0);
        assert_eq!(import.fd(), raw);
        let taken = import.take_fd();
        assert_eq!(taken, raw);
        assert_eq!(import.fd(), -1);
        drop(import);
        assert_eq!(b.read(&mut byte).unwrap_err().kind(), ErrorKind::WouldBlock);
        drop(unsafe { OwnedFd::from_raw_fd(taken) });
        assert_eq!(b.read(&mut byte).unwrap(), 0);

        let (a, mut b) = std::os::unix::net::UnixStream::pair().unwrap();
        b.set_nonblocking(true).unwrap();
        let raw = std::os::fd::IntoRawFd::into_raw_fd(a);
        drop(DmaBufImport::new(raw, 1, 1, 4, 0));
        assert_eq!(b.read(&mut byte).unwrap(), 0, "drop closes the fd");
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
