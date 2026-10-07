//! Backend-neutral half of the CUDA → `Viewport` path: the exported handles a CUDA importer
//! needs, and the slot table the producer (a Python thread) and the render thread share.
//!
//! Kept in `fastgui-app` so [`crate::Command::CreateCudaSurface`] does not depend on the Vulkan
//! crate. Metal never sends that command.
//!
//! Each CUDA layer owns [`CUDA_SLOTS`] slots in one exported linear buffer. A producer writes a
//! whole frame into a free slot on its CUDA stream, signals the layer's `ready` timeline
//! semaphore, and publishes the slot once that GPU work has finished. The render thread takes
//! the newest published slot, waits on `ready`, copies the slot into its sampled image, and
//! signals the `release` timeline semaphore when the copy is done. The slot is free again from
//! that point on; the next producer of that slot waits on `release` GPU-side before writing.
//!
//! Every value either side waits on has already been submitted by the other side, so neither a
//! hidden window (nothing rendered) nor an idle producer can deadlock the other.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Slots per CUDA layer. Two would do (the producer writes one while the newest published frame
/// waits in the other); the third lets a producer that is ahead of its own GPU work start the
/// next frame instead of blocking.
pub const CUDA_SLOTS: usize = 3;

/// What `Viewport.create_cuda_surface` / `submit_cuda` get back from the render thread: the
/// exported Vulkan objects for one CUDA layer, plus the slot table shared with it.
///
/// Owns the exported NT handles and closes them on drop. CUDA's import does not take ownership
/// of a handle, so dropping this right after importing is correct.
pub struct CudaExportHandles {
    /// Exported `VkDeviceMemory` holding all slots, `memory_size` bytes.
    pub memory_win32_handle: isize,
    pub memory_size: u64,
    /// The allocation is a dedicated one: CUDA needs `CUDA_EXTERNAL_MEMORY_DEDICATED`.
    pub memory_dedicated: bool,
    /// Byte offset between slots (slot `i` starts at `i * slot_stride`). Each slot holds a
    /// tightly packed `width`×`height` RGBA8 frame (`width * 4`-byte rows).
    pub slot_stride: u64,
    pub width: u32,
    pub height: u32,
    /// Timeline semaphore CUDA signals when a slot's frame is written.
    pub ready_win32_handle: isize,
    /// Timeline semaphore Vulkan signals when it has finished copying a slot out.
    pub release_win32_handle: isize,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`, to find the CUDA device on the same GPU.
    pub device_uuid: [u8; 16],
    pub shared: Arc<CudaLayerShared>,
}

impl Drop for CudaExportHandles {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        for handle in [self.memory_win32_handle, self.ready_win32_handle, self.release_win32_handle] {
            if handle != 0 {
                // SAFETY: each handle came from vkGet*Win32HandleKHR, which hands ownership of a
                // fresh NT handle to the caller; nothing else closes it.
                unsafe { windows_sys::Win32::Foundation::CloseHandle(handle as _) };
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    /// Safe to write once `release` reaches the slot's `release` value.
    Free,
    /// Handed to a producer; GPU writes may be in flight.
    Writing,
    /// Written and newest; waiting for the render thread.
    Published { ready: u64 },
    /// Taken by the render thread for the frame it is recording.
    Copying,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    state: SlotState,
    /// `release` value Vulkan signals after its last copy out of this slot (0: never copied).
    release: u64,
}

#[derive(Debug)]
struct SlotTable {
    slots: [Slot; CUDA_SLOTS],
    closed: bool,
}

/// A slot handed to a producer by [`CudaLayerShared::acquire`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquiredSlot {
    pub index: usize,
    /// Wait (GPU-side) until `release` reaches this before writing; 0 means no wait needed.
    pub wait_release: u64,
}

/// A published slot taken by the render thread with [`CudaLayerShared::take_published`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublishedSlot {
    pub index: usize,
    /// `ready` value to wait on before reading the slot.
    pub ready: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquireError {
    /// The render thread dropped the layer (window closed, viewport removed, replaced by a CPU
    /// frame or a new size). Create a new one.
    Closed,
    /// Every slot is still being written; the producer is too far ahead of its own GPU.
    Timeout,
}

/// The slot table one CUDA layer's producer and render thread share. See the module docs.
#[derive(Debug)]
pub struct CudaLayerShared {
    table: Mutex<SlotTable>,
    changed: Condvar,
    /// Newest `ready` value published.
    last_published: AtomicU64,
}

impl Default for CudaLayerShared {
    fn default() -> Self {
        Self {
            table: Mutex::new(SlotTable {
                slots: [Slot { state: SlotState::Free, release: 0 }; CUDA_SLOTS],
                closed: false,
            }),
            changed: Condvar::new(),
            last_published: AtomicU64::new(0),
        }
    }
}

impl CudaLayerShared {
    fn lock(&self) -> MutexGuard<'_, SlotTable> {
        self.table.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Producer: claim a slot to write the next frame into. Prefers a free slot (the one released
    /// longest ago); otherwise takes back the published-but-not-yet-copied one, dropping that
    /// frame in favour of the newer one about to be written. Blocks only while every slot is
    /// being written, up to `timeout`.
    pub fn acquire(&self, timeout: Duration) -> Result<AcquiredSlot, AcquireError> {
        let deadline = Instant::now() + timeout;
        let mut table = self.lock();
        loop {
            if table.closed {
                return Err(AcquireError::Closed);
            }
            let free = (0..CUDA_SLOTS)
                .filter(|&i| table.slots[i].state == SlotState::Free)
                .min_by_key(|&i| table.slots[i].release);
            let pick = free.or_else(|| {
                (0..CUDA_SLOTS).find(|&i| matches!(table.slots[i].state, SlotState::Published { .. }))
            });
            if let Some(index) = pick {
                let slot = &mut table.slots[index];
                slot.state = SlotState::Writing;
                return Ok(AcquiredSlot { index, wait_release: slot.release });
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(AcquireError::Timeout);
            }
            table = self
                .changed
                .wait_timeout(table, deadline - now)
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .0;
        }
    }

    /// Producer (from a CUDA host callback, once the write and the `ready` signal have completed):
    /// make `index` the newest frame. An older published frame the render thread never took goes
    /// back to free; a newer one wins and this slot is freed instead. A no-op when the slot is no
    /// longer this producer's.
    pub fn publish(&self, index: usize, ready: u64) {
        let mut table = self.lock();
        if table.slots[index].state != SlotState::Writing {
            return;
        }
        let newer_shown = table
            .slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Published { ready: other } if other > ready));
        if newer_shown {
            table.slots[index].state = SlotState::Free;
        } else {
            for slot in table.slots.iter_mut() {
                if matches!(slot.state, SlotState::Published { .. }) {
                    slot.state = SlotState::Free;
                }
            }
            table.slots[index].state = SlotState::Published { ready };
            self.last_published.fetch_max(ready, Ordering::AcqRel);
        }
        drop(table);
        self.changed.notify_all();
    }

    /// Producer: give back a slot without publishing it. Only call once any GPU writes into it
    /// have completed.
    pub fn abandon(&self, index: usize) {
        let mut table = self.lock();
        if table.slots[index].state == SlotState::Writing {
            table.slots[index].state = SlotState::Free;
        }
        drop(table);
        self.changed.notify_all();
    }

    /// Render thread: take the newest published frame, if there is one, to copy out this frame.
    /// Follow with [`Self::copied`] once the submission is queued, or [`Self::copy_failed`].
    pub fn take_published(&self) -> Option<PublishedSlot> {
        let mut table = self.lock();
        let (index, ready) = table.slots.iter().enumerate().find_map(|(i, slot)| match slot.state {
            SlotState::Published { ready } => Some((i, ready)),
            _ => None,
        })?;
        table.slots[index].state = SlotState::Copying;
        Some(PublishedSlot { index, ready })
    }

    /// Render thread: the copy out of `index` is submitted and signals `release = value` when it
    /// completes. The slot is free for producers from now on (they wait on `value` GPU-side).
    pub fn copied(&self, index: usize, release: u64) {
        let mut table = self.lock();
        let slot = &mut table.slots[index];
        if slot.state == SlotState::Copying {
            slot.state = SlotState::Free;
            slot.release = release;
        }
        drop(table);
        self.changed.notify_all();
    }

    /// Render thread: recording or submitting the copy failed, so nothing reads the slot or
    /// signals a new `release`. Free it with its previous `release` value.
    pub fn copy_failed(&self, index: usize) {
        let mut table = self.lock();
        if table.slots[index].state == SlotState::Copying {
            table.slots[index].state = SlotState::Free;
        }
        drop(table);
        self.changed.notify_all();
    }

    /// Render thread: the layer is gone. Producers get [`AcquireError::Closed`] from now on.
    pub fn close(&self) {
        self.lock().closed = true;
        self.changed.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// Newest `ready` value published so far (0: none).
    pub fn last_published(&self) -> u64 {
        self.last_published.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: Duration = Duration::ZERO;

    #[test]
    fn producer_and_renderer_round_trip() {
        let shared = CudaLayerShared::default();
        let a = shared.acquire(NOW).unwrap();
        assert_eq!(a.wait_release, 0, "a never-copied slot needs no wait");
        assert!(shared.take_published().is_none(), "nothing published while writing");
        shared.publish(a.index, 1);
        assert_eq!(shared.last_published(), 1);
        let taken = shared.take_published().unwrap();
        assert_eq!(taken, PublishedSlot { index: a.index, ready: 1 });
        assert!(shared.take_published().is_none(), "a frame is copied once");
        shared.copied(taken.index, 7);
        // All free again; the slot just copied was released last, so it is handed out last.
        let b = shared.acquire(NOW).unwrap();
        let c = shared.acquire(NOW).unwrap();
        assert!(b.index != a.index && c.index != a.index && b.index != c.index);
        assert_eq!((b.wait_release, c.wait_release), (0, 0));
        assert_eq!(shared.acquire(NOW).unwrap(), AcquiredSlot { index: a.index, wait_release: 7 });
    }

    #[test]
    fn newer_publish_replaces_an_uncopied_frame() {
        let shared = CudaLayerShared::default();
        let a = shared.acquire(NOW).unwrap();
        let b = shared.acquire(NOW).unwrap();
        shared.publish(a.index, 1);
        shared.publish(b.index, 2);
        assert_eq!(shared.take_published(), Some(PublishedSlot { index: b.index, ready: 2 }));
        assert!(shared.take_published().is_none(), "frame 1 was dropped, not queued");
    }

    #[test]
    fn late_older_publish_is_dropped() {
        let shared = CudaLayerShared::default();
        let a = shared.acquire(NOW).unwrap();
        let b = shared.acquire(NOW).unwrap();
        shared.publish(b.index, 2);
        shared.publish(a.index, 1);
        assert_eq!(shared.take_published(), Some(PublishedSlot { index: b.index, ready: 2 }));
        // Frame 1's slot went back to free rather than staying stuck in `Writing`.
        assert!(shared.acquire(NOW).is_ok());
        assert!(shared.acquire(NOW).is_ok());
    }

    #[test]
    fn producer_steals_the_published_slot_when_nothing_is_free() {
        let shared = CudaLayerShared::default();
        let a = shared.acquire(NOW).unwrap();
        let _b = shared.acquire(NOW).unwrap();
        let _c = shared.acquire(NOW).unwrap();
        assert_eq!(shared.acquire(NOW), Err(AcquireError::Timeout), "all three being written");
        shared.publish(a.index, 1);
        assert_eq!(shared.acquire(NOW).unwrap().index, a.index, "the unshown frame is overwritten");
        assert!(shared.take_published().is_none());
    }

    #[test]
    fn copying_slot_is_never_handed_out() {
        let shared = CudaLayerShared::default();
        let a = shared.acquire(NOW).unwrap();
        shared.publish(a.index, 1);
        let taken = shared.take_published().unwrap();
        let x = shared.acquire(NOW).unwrap();
        let y = shared.acquire(NOW).unwrap();
        assert!(x.index != taken.index && y.index != taken.index);
        assert_eq!(shared.acquire(NOW), Err(AcquireError::Timeout));
        shared.copy_failed(taken.index);
        assert_eq!(shared.acquire(NOW).unwrap(), AcquiredSlot { index: taken.index, wait_release: 0 });
    }

    #[test]
    fn acquire_wakes_when_a_slot_frees_up() {
        let shared = Arc::new(CudaLayerShared::default());
        let held: Vec<_> = (0..CUDA_SLOTS).map(|_| shared.acquire(NOW).unwrap()).collect();
        let waiter = {
            let shared = shared.clone();
            std::thread::spawn(move || shared.acquire(Duration::from_secs(10)))
        };
        std::thread::sleep(Duration::from_millis(20));
        shared.abandon(held[1].index);
        assert_eq!(waiter.join().unwrap().unwrap().index, held[1].index);
    }

    #[test]
    fn close_fails_waiting_and_later_producers() {
        let shared = Arc::new(CudaLayerShared::default());
        for _ in 0..CUDA_SLOTS {
            shared.acquire(NOW).unwrap();
        }
        let waiter = {
            let shared = shared.clone();
            std::thread::spawn(move || shared.acquire(Duration::from_secs(10)))
        };
        std::thread::sleep(Duration::from_millis(20));
        shared.close();
        assert_eq!(waiter.join().unwrap(), Err(AcquireError::Closed));
        assert_eq!(shared.acquire(NOW), Err(AcquireError::Closed));
        assert!(shared.is_closed());
    }
}
