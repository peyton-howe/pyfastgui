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
