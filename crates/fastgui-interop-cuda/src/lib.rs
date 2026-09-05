//! CUDA driver API interop: imports Vulkan-exported (win32) memory and timeline-semaphore
//! handles into CUDA, so a CUDA kernel can write directly into memory a `fastgui` `Viewport`
//! displays via Vulkan — zero-copy, no CPU round-trip.
//!
//! # Verification status
//!
//! **This crate has never been run against a real CUDA driver.** It was developed on a
//! machine with no NVIDIA GPU; everything here compiles and is written directly against
//! NVIDIA's own struct/function definitions (transcribed from `cuda-python`'s bindings
//! generator source, not from memory), but the actual behavior — does the import succeed, is
//! the struct layout really bit-for-bit correct, does the semaphore wait really prevent a
//! torn read — is unverified. Treat this as a first draft to validate on real hardware, not
//! as a working implementation. See `sys.rs` for exactly which structs are highest-risk.
//!
//! Dynamically loads `nvcuda.dll` at first use (not linked at build time), so the rest of the
//! workspace builds fine on machines without CUDA installed — confirmed on the no-NVIDIA-GPU
//! machine this was developed on: attempting to use this path there fails cleanly with a
//! "failed to load nvcuda.dll" error rather than crashing.

mod driver;
mod error;
mod sys;

use std::sync::Arc;

pub use error::CudaError;

use driver::CudaDriver;
use sys::*;

/// A retained CUDA primary context, current on whatever thread calls into this crate.
/// Cheap and safe to create more than once: `cuDevicePrimaryCtxRetain` is refcounted by the
/// driver and returns the same context handle for repeat calls, so callers don't need to
/// share a single `CudaContext` — each `Viewport.create_cuda_surface()` call can make its own.
pub struct CudaContext {
    driver: Arc<CudaDriver>,
    context: CUcontext,
}

// SAFETY (unverified): CUDA contexts are documented as safe to use from any thread as long as
// `cuCtxSetCurrent` establishes the right context on that thread first, which every method
// here that touches the driver does.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new() -> Result<Self, CudaError> {
        let driver = Arc::new(CudaDriver::load()?);
        unsafe {
            driver.check((driver.init)(0))?;
            let mut device: CUdevice = 0;
            driver.check((driver.device_get)(&mut device, 0))?;
            let mut context: CUcontext = std::ptr::null_mut();
            driver.check((driver.device_primary_ctx_retain)(&mut context, device))?;
            driver.check((driver.ctx_set_current)(context))?;
            Ok(Self { driver, context })
        }
    }

    fn make_current(&self) -> Result<(), CudaError> {
        unsafe { self.driver.check((self.driver.ctx_set_current)(self.context)) }
    }
}

/// A CUDA stream, used to order the semaphore signal issued by `ExternalSemaphore::signal`.
pub struct CudaStream {
    driver: Arc<CudaDriver>,
    stream: CUstream,
}

unsafe impl Send for CudaStream {}

impl CudaStream {
    pub fn new(ctx: &CudaContext) -> Result<Self, CudaError> {
        ctx.make_current()?;
        unsafe {
            let mut stream: CUstream = std::ptr::null_mut();
            ctx.driver.check((ctx.driver.stream_create)(&mut stream, 0))?;
            Ok(Self { driver: ctx.driver.clone(), stream })
        }
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { self.driver.check((self.driver.stream_synchronize)(self.stream)) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.stream_destroy)(self.stream);
        }
    }
}

/// A Vulkan-exported win32 memory allocation, imported into CUDA and mapped as a flat device
/// buffer. `device_ptr()` is the raw pointer a CUDA kernel (or cupy/numba/torch, via whatever
/// `__cuda_array_interface__`-style wrapping the caller does) writes into.
pub struct ExternalMemory {
    driver: Arc<CudaDriver>,
    handle: CUexternalMemory,
    device_ptr: CUdeviceptr,
}

unsafe impl Send for ExternalMemory {}
// `&self` only ever hands back a copy of `device_ptr` (no interior mutability), so sharing a
// `&ExternalMemory` across threads is fine.
unsafe impl Sync for ExternalMemory {}

impl ExternalMemory {
    /// `win32_handle` and `size` must come from the exact Vulkan export this is meant to
    /// import (`vkGetMemoryWin32HandleKHR` output and the memory's `VkMemoryRequirements.size`
    /// — not the logical width*height*4 pixel byte count, which is smaller than the actual
    /// allocation once alignment padding is included).
    pub fn import_win32(ctx: &CudaContext, win32_handle: isize, size: u64) -> Result<Self, CudaError> {
        ctx.make_current()?;
        unsafe {
            let desc = CudaExternalMemoryHandleDesc {
                ty: CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
                handle: CudaExternalMemoryHandle {
                    win32: CudaWin32Handle {
                        handle: win32_handle as *mut std::ffi::c_void,
                        name: std::ptr::null(),
                    },
                },
                size,
                flags: 0,
                reserved: [0; 16],
            };
            let mut handle: CUexternalMemory = std::ptr::null_mut();
            ctx.driver.check((ctx.driver.import_external_memory)(&mut handle, &desc))?;

            let buffer_desc =
                CudaExternalMemoryBufferDesc { offset: 0, size, flags: 0, reserved: [0; 16] };
            let mut device_ptr: CUdeviceptr = 0;
            ctx.driver.check((ctx.driver.external_memory_get_mapped_buffer)(
                &mut device_ptr,
                handle,
                &buffer_desc,
            ))?;

            Ok(Self { driver: ctx.driver.clone(), handle, device_ptr })
        }
    }

    /// Raw CUDA device pointer, valid for `size` bytes as passed to `import_win32`.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }
}

impl Drop for ExternalMemory {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.destroy_external_memory)(self.handle);
        }
    }
}

/// A Vulkan-exported win32 timeline semaphore, imported into CUDA so a kernel's completion can
/// be signalled in a way the Vulkan render thread can wait on before sampling the shared
/// texture — without CUDA and Vulkan ever needing to touch the same queue or block on a CPU
/// round-trip.
pub struct ExternalSemaphore {
    driver: Arc<CudaDriver>,
    handle: CUexternalSemaphore,
}

unsafe impl Send for ExternalSemaphore {}

impl ExternalSemaphore {
    /// `win32_handle` must come from `vkGetSemaphoreWin32HandleKHR` on a semaphore created
    /// with `VkSemaphoreTypeCreateInfo{ semaphoreType: TIMELINE }` — this only imports it as a
    /// *timeline* semaphore, not CUDA's other (binary/D3D12-fence-style) semaphore handle
    /// types.
    pub fn import_win32_timeline(ctx: &CudaContext, win32_handle: isize) -> Result<Self, CudaError> {
        ctx.make_current()?;
        unsafe {
            let desc = CudaExternalSemaphoreHandleDesc {
                ty: CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32,
                handle: CudaExternalSemaphoreHandle {
                    win32: CudaWin32Handle {
                        handle: win32_handle as *mut std::ffi::c_void,
                        name: std::ptr::null(),
                    },
                },
                flags: 0,
                reserved: [0; 16],
            };
            let mut handle: CUexternalSemaphore = std::ptr::null_mut();
            ctx.driver.check((ctx.driver.import_external_semaphore)(&mut handle, &desc))?;
            Ok(Self { driver: ctx.driver.clone(), handle })
        }
    }

    /// Signal the semaphore to `value` on `stream`. Callers wanting this to have definitely
    /// happened before telling anyone else "the frame is ready" should follow this with
    /// `stream.synchronize()` — `signal` alone only *enqueues* the signal.
    pub fn signal(&self, stream: &CudaStream, value: u64) -> Result<(), CudaError> {
        unsafe {
            let params = CudaExternalSemaphoreSignalParams::for_timeline_value(value);
            let semaphores = [self.handle];
            let params_array = [params];
            self.driver.check((self.driver.signal_external_semaphores_async)(
                semaphores.as_ptr(),
                params_array.as_ptr(),
                1,
                stream.stream,
            ))
        }
    }
}

impl Drop for ExternalSemaphore {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.destroy_external_semaphore)(self.handle);
        }
    }
}
