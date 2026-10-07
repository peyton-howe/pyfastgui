//! CUDA driver API interop for `fastgui`: imports the Vulkan-exported (win32) slot buffer and
//! timeline semaphores of a CUDA `Viewport` layer, and the stream/event/copy plumbing that moves
//! a CUDA array into it in order with the caller's own stream. See `fastgui_app::cuda_handles`
//! for the protocol.
//!
//! # Verification status
//!
//! **This crate has never been run against a real CUDA driver.** It was developed on a machine
//! with no NVIDIA GPU. Struct layouts, enum values and signatures come from NVIDIA's
//! `cuda-python` bindings, and every struct offset is asserted against the layout measured
//! through those bindings (see `sys.rs`), but whether the driver accepts these exact imports and
//! whether the semaphore handshake really orders the two APIs is unverified.
//!
//! Dynamically loads the driver (`nvcuda.dll` / `libcuda.so.1`) on first use rather than
//! linking it, so everything builds and runs without CUDA installed; using this crate there fails
//! cleanly with [`CudaError::Load`].
//!
//! Every call pushes the context it needs and pops it afterwards, so the caller's thread keeps
//! whatever context it had current (a torch/cupy context on another device, say).

mod driver;
mod error;
mod sys;

use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

pub use error::CudaError;

use driver::{driver, CudaDriver};
use sys::*;

/// The CUDA driver library name this platform loads.
pub const DRIVER_LIBRARY: &str = driver::LIBRARY;

/// A `CUstream` handle as an integer, the way Python libraries hand them around. `0` is the
/// legacy default stream; `1` (`CU_STREAM_LEGACY`) and `2` (`CU_STREAM_PER_THREAD`) are the
/// driver's special handles and are passed through as-is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawStream(pub usize);

impl RawStream {
    fn get(self) -> CUstream {
        self.0 as CUstream
    }
}

/// Load the driver (once per process) without doing anything else.
pub fn ensure_loaded() -> Result<(), CudaError> {
    driver().map(|_| ())
}

/// The ordinal of the CUDA device whose UUID is `uuid` (as reported by Vulkan's
/// `VkPhysicalDeviceIDProperties::deviceUUID`), if any.
pub fn device_by_uuid(uuid: &[u8; 16]) -> Result<Option<i32>, CudaError> {
    let d = driver()?;
    unsafe {
        let mut count = 0;
        d.check((d.device_get_count)(&mut count))?;
        for ordinal in 0..count {
            let mut device: CUdevice = 0;
            d.check((d.device_get)(&mut device, ordinal))?;
            let mut id = CUuuid::default();
            d.check((d.device_get_uuid)(&mut id, device))?;
            if id.bytes.map(|b| b as u8) == *uuid {
                return Ok(Some(ordinal));
            }
        }
    }
    Ok(None)
}

/// The ordinal of the device that owns `ptr` (any UVA pointer).
pub fn pointer_device(ptr: u64) -> Result<i32, CudaError> {
    let d = driver()?;
    let query = || unsafe {
        let mut ordinal: i32 = 0;
        let result = (d.pointer_get_attribute)(
            (&mut ordinal as *mut i32).cast(),
            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            ptr,
        );
        (result, ordinal)
    };
    let (mut result, mut ordinal) = query();
    // Older drivers want some context current even for UVA lookups.
    const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
    if result == CUDA_ERROR_INVALID_CONTEXT {
        let ctx = PrimaryContext::new(0)?;
        let _current = ctx.enter()?;
        (result, ordinal) = query();
    }
    d.check(result)?;
    Ok(ordinal)
}

/// A retained primary context (the one the runtime API, torch and cupy use) for one device.
pub struct PrimaryContext {
    driver: &'static CudaDriver,
    device: CUdevice,
    context: CUcontext,
}

// SAFETY: a context handle may be made current on any thread; every use goes through `enter`.
unsafe impl Send for PrimaryContext {}
unsafe impl Sync for PrimaryContext {}

impl PrimaryContext {
    pub fn new(ordinal: i32) -> Result<Arc<Self>, CudaError> {
        let d = driver()?;
        unsafe {
            let mut device: CUdevice = 0;
            d.check((d.device_get)(&mut device, ordinal))?;
            let mut context: CUcontext = std::ptr::null_mut();
            d.check((d.device_primary_ctx_retain)(&mut context, device))?;
            Ok(Arc::new(Self { driver: d, device, context }))
        }
    }

    /// The device ordinal this context belongs to.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Make this context current on this thread until the guard drops (push/pop, so the
    /// thread's previous context comes back).
    pub fn enter(&self) -> Result<ContextGuard<'_>, CudaError> {
        unsafe { self.driver.check((self.driver.ctx_push_current)(self.context))? };
        Ok(ContextGuard { ctx: self, _not_send: std::marker::PhantomData })
    }
}

/// Copy `dst.len()` contiguous bytes from device pointer `src` into `dst`, after the work already
/// enqueued on `producer` (the stream the data was written on; `None` when no ordering is
/// needed). Blocks until the copy is done. Runs on the owning device's primary context, pushed
/// and popped, so the caller's current context is untouched.
pub fn copy_to_host(src: u64, dst: &mut [u8], producer: Option<RawStream>) -> Result<(), CudaError> {
    if dst.is_empty() {
        return Ok(());
    }
    let ctx = PrimaryContext::new(pointer_device(src)?)?;
    let stream = Stream::new(&ctx)?;
    if let Some(producer) = producer {
        stream.wait_for(producer)?;
    }
    // SAFETY: `dst` is borrowed mutably for the whole call and we synchronize before returning.
    unsafe { stream.copy_2d_to_host(src, dst.len(), dst.as_mut_ptr(), dst.len(), dst.len(), 1)? };
    stream.synchronize()
}

impl Drop for PrimaryContext {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.device_primary_ctx_release)(self.device);
        }
    }
}

/// See [`PrimaryContext::enter`].
pub struct ContextGuard<'a> {
    ctx: &'a PrimaryContext,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            let mut popped: CUcontext = std::ptr::null_mut();
            let _ = (self.ctx.driver.ctx_pop_current)(&mut popped);
        }
    }
}

/// A non-blocking stream (no implicit ordering with the legacy default stream; every dependency
/// is an explicit event wait).
pub struct Stream {
    ctx: Arc<PrimaryContext>,
    stream: CUstream,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new(ctx: &Arc<PrimaryContext>) -> Result<Self, CudaError> {
        let _current = ctx.enter()?;
        let d = ctx.driver;
        unsafe {
            let mut stream: CUstream = std::ptr::null_mut();
            d.check((d.stream_create)(&mut stream, CU_STREAM_NON_BLOCKING))?;
            Ok(Self { ctx: ctx.clone(), stream })
        }
    }

    pub fn raw(&self) -> RawStream {
        RawStream(self.stream as usize)
    }

    pub fn context(&self) -> &Arc<PrimaryContext> {
        &self.ctx
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        unsafe { self.ctx.driver.check((self.ctx.driver.stream_synchronize)(self.stream)) }
    }

    /// Make this stream wait for everything enqueued on `other` so far.
    pub fn wait_for(&self, other: RawStream) -> Result<(), CudaError> {
        let event = Event::new(&self.ctx)?;
        event.record(other)?;
        event.wait_on(self.raw())
    }

    /// Make `other` wait for everything enqueued on this stream so far.
    pub fn make_wait(&self, other: RawStream) -> Result<(), CudaError> {
        let event = Event::new(&self.ctx)?;
        event.record(self.raw())?;
        event.wait_on(other)
    }

    /// Copy `height` rows of `width_bytes` from `src` (any UVA pointer) into device memory.
    ///
    /// # Safety
    /// Both ranges must be valid for the copy until it completes on this stream.
    pub unsafe fn copy_2d_to_device(
        &self,
        src: u64,
        src_pitch: usize,
        dst: u64,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), CudaError> {
        self.copy_2d(CudaMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_UNIFIED,
            src_device: src,
            src_pitch,
            dst_memory_type: CU_MEMORYTYPE_DEVICE,
            dst_device: dst,
            dst_pitch,
            ..memcpy_2d(width_bytes, height)
        })
    }

    /// Copy `height` rows of `width_bytes` from `src` (any UVA pointer) into host memory at `dst`.
    ///
    /// # Safety
    /// `src` must stay valid, and `dst` valid and untouched, until the copy completes on this
    /// stream (synchronize before reading `dst`).
    pub unsafe fn copy_2d_to_host(
        &self,
        src: u64,
        src_pitch: usize,
        dst: *mut u8,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
    ) -> Result<(), CudaError> {
        self.copy_2d(CudaMemcpy2D {
            src_memory_type: CU_MEMORYTYPE_UNIFIED,
            src_device: src,
            src_pitch,
            dst_memory_type: CU_MEMORYTYPE_HOST,
            dst_host: dst.cast(),
            dst_pitch,
            ..memcpy_2d(width_bytes, height)
        })
    }

    unsafe fn copy_2d(&self, copy: CudaMemcpy2D) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        self.ctx.driver.check((self.ctx.driver.memcpy_2d_async)(&copy, self.stream))
    }

    /// Run `f` on a driver thread once everything enqueued on this stream so far has completed.
    /// `f` must not call into CUDA. A panic in `f` is caught and dropped.
    pub fn on_complete(&self, f: Box<dyn FnOnce() + Send>) -> Result<(), CudaError> {
        unsafe extern "system" fn trampoline(data: *mut c_void) {
            let f = Box::from_raw(data.cast::<Box<dyn FnOnce() + Send>>());
            let _ = std::panic::catch_unwind(AssertUnwindSafe(f));
        }
        let _current = self.ctx.enter()?;
        let data = Box::into_raw(Box::new(f));
        let result = unsafe { (self.ctx.driver.launch_host_func)(self.stream, trampoline, data.cast()) };
        if result != CUDA_SUCCESS {
            // Never enqueued: reclaim the closure here instead of leaking it.
            drop(unsafe { Box::from_raw(data) });
        }
        self.ctx.driver.check(result)
    }
}

fn memcpy_2d(width_bytes: usize, height: usize) -> CudaMemcpy2D {
    CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: 0,
        src_host: std::ptr::null(),
        src_device: 0,
        src_array: std::ptr::null_mut(),
        src_pitch: 0,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: 0,
        dst_host: std::ptr::null_mut(),
        dst_device: 0,
        dst_array: std::ptr::null_mut(),
        dst_pitch: 0,
        width_in_bytes: width_bytes,
        height,
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(_current) = self.ctx.enter() {
            unsafe {
                let _ = (self.ctx.driver.stream_synchronize)(self.stream);
                let _ = (self.ctx.driver.stream_destroy)(self.stream);
            }
        }
    }
}

/// A timing-disabled event, used only to order one stream after another.
struct Event {
    ctx: Arc<PrimaryContext>,
    event: CUevent,
}

impl Event {
    fn new(ctx: &Arc<PrimaryContext>) -> Result<Self, CudaError> {
        let _current = ctx.enter()?;
        unsafe {
            let mut event: CUevent = std::ptr::null_mut();
            ctx.driver.check((ctx.driver.event_create)(&mut event, CU_EVENT_DISABLE_TIMING))?;
            Ok(Self { ctx: ctx.clone(), event })
        }
    }

    fn record(&self, stream: RawStream) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        unsafe { self.ctx.driver.check((self.ctx.driver.event_record)(self.event, stream.get())) }
    }

    fn wait_on(&self, stream: RawStream) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        unsafe { self.ctx.driver.check((self.ctx.driver.stream_wait_event)(stream.get(), self.event, 0)) }
    }
}

impl Drop for Event {
    // Destroying an event with work still pending on it is fine: the driver releases it once
    // that work completes.
    fn drop(&mut self) {
        if let Ok(_current) = self.ctx.enter() {
            unsafe {
                let _ = (self.ctx.driver.event_destroy)(self.event);
            }
        }
    }
}

/// A Vulkan-exported win32 memory allocation, imported into CUDA and mapped whole as a flat
/// device buffer.
pub struct ExternalMemory {
    ctx: Arc<PrimaryContext>,
    handle: CUexternalMemory,
    device_ptr: CUdeviceptr,
}

unsafe impl Send for ExternalMemory {}
unsafe impl Sync for ExternalMemory {}

impl ExternalMemory {
    /// `win32_handle`, `size` and `dedicated` must describe the exact Vulkan export: the handle
    /// from `vkGetMemoryWin32HandleKHR`, the allocation's `VkMemoryRequirements.size` (not the
    /// logical byte count), and whether it was a dedicated allocation. The handle is not
    /// consumed; the caller still closes it.
    pub fn import_win32(
        ctx: &Arc<PrimaryContext>,
        win32_handle: isize,
        size: u64,
        dedicated: bool,
    ) -> Result<Self, CudaError> {
        let _current = ctx.enter()?;
        let d = ctx.driver;
        unsafe {
            let desc = CudaExternalMemoryHandleDesc {
                ty: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
                handle: CudaExternalMemoryHandle {
                    win32: CudaWin32Handle { handle: win32_handle as *mut c_void, name: std::ptr::null() },
                },
                size,
                flags: if dedicated { CUDA_EXTERNAL_MEMORY_DEDICATED } else { 0 },
                reserved: [0; 16],
            };
            let mut handle: CUexternalMemory = std::ptr::null_mut();
            d.check((d.import_external_memory)(&mut handle, &desc))?;
            let buffer_desc = CudaExternalMemoryBufferDesc { offset: 0, size, flags: 0, reserved: [0; 16] };
            let mut device_ptr: CUdeviceptr = 0;
            if let Err(err) = d.check((d.external_memory_get_mapped_buffer)(&mut device_ptr, handle, &buffer_desc)) {
                let _ = (d.destroy_external_memory)(handle);
                return Err(err);
            }
            Ok(Self { ctx: ctx.clone(), handle, device_ptr })
        }
    }

    /// Device pointer to the start of the allocation.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        if let Ok(_current) = self.ctx.enter() {
            unsafe {
                // A mapped buffer is freed with cuMemFree before the import goes away.
                let _ = (self.ctx.driver.mem_free)(self.device_ptr);
                let _ = (self.ctx.driver.destroy_external_memory)(self.handle);
            }
        }
    }
}

/// A Vulkan-exported win32 *timeline* semaphore imported into CUDA.
pub struct ExternalSemaphore {
    ctx: Arc<PrimaryContext>,
    handle: CUexternalSemaphore,
}

unsafe impl Send for ExternalSemaphore {}
unsafe impl Sync for ExternalSemaphore {}

impl ExternalSemaphore {
    /// `win32_handle` must come from `vkGetSemaphoreWin32HandleKHR` on a semaphore created with
    /// `VkSemaphoreTypeCreateInfo { semaphoreType: TIMELINE }`. Not consumed; the caller closes it.
    pub fn import_win32_timeline(ctx: &Arc<PrimaryContext>, win32_handle: isize) -> Result<Self, CudaError> {
        let _current = ctx.enter()?;
        let d = ctx.driver;
        unsafe {
            let desc = CudaExternalSemaphoreHandleDesc {
                ty: CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
                handle: CudaExternalSemaphoreHandle {
                    win32: CudaWin32Handle { handle: win32_handle as *mut c_void, name: std::ptr::null() },
                },
                flags: 0,
                reserved: [0; 16],
            };
            let mut handle: CUexternalSemaphore = std::ptr::null_mut();
            d.check((d.import_external_semaphore)(&mut handle, &desc))?;
            Ok(Self { ctx: ctx.clone(), handle })
        }
    }

    /// Enqueue on `stream`: set the counter to `value` once earlier work on `stream` is done.
    pub fn signal(&self, stream: &Stream, value: u64) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        let params = [CudaExternalSemaphoreSignalParams::for_timeline_value(value)];
        unsafe {
            self.ctx.driver.check((self.ctx.driver.signal_external_semaphores_async)(
                &self.handle,
                params.as_ptr(),
                1,
                stream.stream,
            ))
        }
    }

    /// Enqueue on `stream`: hold later work on `stream` until the counter reaches `value`.
    pub fn wait(&self, stream: &Stream, value: u64) -> Result<(), CudaError> {
        let _current = self.ctx.enter()?;
        let params = [CudaExternalSemaphoreWaitParams::for_timeline_value(value)];
        unsafe {
            self.ctx.driver.check((self.ctx.driver.wait_external_semaphores_async)(
                &self.handle,
                params.as_ptr(),
                1,
                stream.stream,
            ))
        }
    }
}

impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        if let Ok(_current) = self.ctx.enter() {
            unsafe {
                let _ = (self.ctx.driver.destroy_external_semaphore)(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a machine without CUDA (like the one this was written on) every entry point fails
    /// cleanly with `Load` rather than crashing; with CUDA, loading succeeds.
    #[test]
    fn missing_driver_is_a_clean_error() {
        match ensure_loaded() {
            Ok(()) => {
                assert!(device_by_uuid(&[0; 16]).unwrap().is_none(), "no device has an all-zero UUID");
            }
            Err(CudaError::Load { library, .. }) => {
                assert_eq!(library, DRIVER_LIBRARY);
                assert!(matches!(PrimaryContext::new(0), Err(CudaError::Load { .. })));
                assert!(matches!(pointer_device(0x1000), Err(CudaError::Load { .. })));
            }
            Err(other) => panic!("unexpected error loading the driver: {other}"),
        }
    }
}
