use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

use crate::error::CudaError;
use crate::sys::*;

#[cfg(windows)]
pub(crate) const LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
pub(crate) const LIBRARY: &str = "libcuda.so.1";

/// The CUDA driver library and the entry points this crate uses. Loaded at runtime (not linked at
/// build time) so the rest of the workspace builds and runs fine on machines without CUDA — the
/// same trick `fastgui-render-vk` uses for the Vulkan loader. Loaded once per process (see
/// [`driver`]) and never unloaded.
pub(crate) struct CudaDriver {
    _lib: libloading::Library,
    pub get_error_string: PFN_cuGetErrorString,
    pub get_error_name: PFN_cuGetErrorName,
    pub init: PFN_cuInit,
    pub device_get_count: PFN_cuDeviceGetCount,
    pub device_get: PFN_cuDeviceGet,
    pub device_get_uuid: PFN_cuDeviceGetUuid,
    pub device_primary_ctx_retain: PFN_cuDevicePrimaryCtxRetain,
    pub device_primary_ctx_release: PFN_cuDevicePrimaryCtxRelease,
    pub ctx_push_current: PFN_cuCtxPushCurrent,
    pub ctx_pop_current: PFN_cuCtxPopCurrent,
    pub pointer_get_attribute: PFN_cuPointerGetAttribute,
    pub stream_create: PFN_cuStreamCreate,
    pub stream_synchronize: PFN_cuStreamSynchronize,
    pub stream_destroy: PFN_cuStreamDestroy,
    pub stream_wait_event: PFN_cuStreamWaitEvent,
    pub event_create: PFN_cuEventCreate,
    pub event_record: PFN_cuEventRecord,
    pub event_destroy: PFN_cuEventDestroy,
    pub memcpy_2d_async: PFN_cuMemcpy2DAsync,
    pub mem_free: PFN_cuMemFree,
    pub launch_host_func: PFN_cuLaunchHostFunc,
    pub import_external_memory: PFN_cuImportExternalMemory,
    pub external_memory_get_mapped_buffer: PFN_cuExternalMemoryGetMappedBuffer,
    pub destroy_external_memory: PFN_cuDestroyExternalMemory,
    pub import_external_semaphore: PFN_cuImportExternalSemaphore,
    pub signal_external_semaphores_async: PFN_cuSignalExternalSemaphoresAsync,
    pub wait_external_semaphores_async: PFN_cuWaitExternalSemaphoresAsync,
    pub destroy_external_semaphore: PFN_cuDestroyExternalSemaphore,
}

// SAFETY: plain function pointers into a library that is never unloaded; the driver API itself
// is thread-safe.
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

/// Resolve the first of `names` the library exports. cuda.h `#define`s several APIs to a `_v2`
/// symbol (same signature, newer semantics); the plain name is still exported for old binaries,
/// so list the `_v2` name first.
macro_rules! resolve {
    ($lib:expr, $ty:ty, $($name:literal),+) => {{
        let mut found: Option<$ty> = None;
        $(
            if found.is_none() {
                if let Ok(symbol) = unsafe { $lib.get::<$ty>(concat!($name, "\0").as_bytes()) } {
                    found = Some(*symbol);
                }
            }
        )+
        let first = [$($name),+][0];
        found.ok_or(CudaError::MissingSymbol(first))?
    }};
}

impl CudaDriver {
    fn load() -> Result<Self, CudaError> {
        let lib = unsafe { libloading::Library::new(LIBRARY) }
            .map_err(|err| CudaError::Load { library: LIBRARY, message: err.to_string() })?;
        let driver = Self {
            get_error_string: resolve!(lib, PFN_cuGetErrorString, "cuGetErrorString"),
            get_error_name: resolve!(lib, PFN_cuGetErrorName, "cuGetErrorName"),
            init: resolve!(lib, PFN_cuInit, "cuInit"),
            device_get_count: resolve!(lib, PFN_cuDeviceGetCount, "cuDeviceGetCount"),
            device_get: resolve!(lib, PFN_cuDeviceGet, "cuDeviceGet"),
            device_get_uuid: resolve!(lib, PFN_cuDeviceGetUuid, "cuDeviceGetUuid_v2", "cuDeviceGetUuid"),
            device_primary_ctx_retain: resolve!(lib, PFN_cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"),
            device_primary_ctx_release: resolve!(
                lib,
                PFN_cuDevicePrimaryCtxRelease,
                "cuDevicePrimaryCtxRelease_v2",
                "cuDevicePrimaryCtxRelease"
            ),
            ctx_push_current: resolve!(lib, PFN_cuCtxPushCurrent, "cuCtxPushCurrent_v2", "cuCtxPushCurrent"),
            ctx_pop_current: resolve!(lib, PFN_cuCtxPopCurrent, "cuCtxPopCurrent_v2", "cuCtxPopCurrent"),
            pointer_get_attribute: resolve!(lib, PFN_cuPointerGetAttribute, "cuPointerGetAttribute"),
            stream_create: resolve!(lib, PFN_cuStreamCreate, "cuStreamCreate"),
            stream_synchronize: resolve!(lib, PFN_cuStreamSynchronize, "cuStreamSynchronize"),
            stream_destroy: resolve!(lib, PFN_cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy"),
            stream_wait_event: resolve!(lib, PFN_cuStreamWaitEvent, "cuStreamWaitEvent"),
            event_create: resolve!(lib, PFN_cuEventCreate, "cuEventCreate"),
            event_record: resolve!(lib, PFN_cuEventRecord, "cuEventRecord"),
            event_destroy: resolve!(lib, PFN_cuEventDestroy, "cuEventDestroy_v2", "cuEventDestroy"),
            memcpy_2d_async: resolve!(lib, PFN_cuMemcpy2DAsync, "cuMemcpy2DAsync_v2", "cuMemcpy2DAsync"),
            mem_free: resolve!(lib, PFN_cuMemFree, "cuMemFree_v2", "cuMemFree"),
            launch_host_func: resolve!(lib, PFN_cuLaunchHostFunc, "cuLaunchHostFunc"),
            import_external_memory: resolve!(lib, PFN_cuImportExternalMemory, "cuImportExternalMemory"),
            external_memory_get_mapped_buffer: resolve!(
                lib,
                PFN_cuExternalMemoryGetMappedBuffer,
                "cuExternalMemoryGetMappedBuffer"
            ),
            destroy_external_memory: resolve!(lib, PFN_cuDestroyExternalMemory, "cuDestroyExternalMemory"),
            import_external_semaphore: resolve!(lib, PFN_cuImportExternalSemaphore, "cuImportExternalSemaphore"),
            signal_external_semaphores_async: resolve!(
                lib,
                PFN_cuSignalExternalSemaphoresAsync,
                "cuSignalExternalSemaphoresAsync"
            ),
            wait_external_semaphores_async: resolve!(
                lib,
                PFN_cuWaitExternalSemaphoresAsync,
                "cuWaitExternalSemaphoresAsync"
            ),
            destroy_external_semaphore: resolve!(lib, PFN_cuDestroyExternalSemaphore, "cuDestroyExternalSemaphore"),
            _lib: lib,
        };
        unsafe { driver.check((driver.init)(0))? };
        Ok(driver)
    }

    /// Turn a `CUresult` into `Result<(), CudaError>`, resolving it to a human-readable message
    /// via `cuGetErrorName`/`cuGetErrorString` on failure.
    pub fn check(&self, result: CUresult) -> Result<(), CudaError> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let name = self.error_str(self.get_error_name, result);
        let description = self.error_str(self.get_error_string, result);
        Err(CudaError::Driver {
            code: result,
            message: match (name, description) {
                (Some(n), Some(d)) => format!("{n}: {d}"),
                (Some(n), None) => n,
                (None, Some(d)) => d,
                (None, None) => "unknown error".to_owned(),
            },
        })
    }

    fn error_str(
        &self,
        f: unsafe extern "system" fn(CUresult, *mut *const c_char) -> CUresult,
        result: CUresult,
    ) -> Option<String> {
        unsafe {
            let mut ptr: *const c_char = std::ptr::null();
            if f(result, &mut ptr) != CUDA_SUCCESS || ptr.is_null() {
                return None;
            }
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

/// The process-wide driver, loaded and `cuInit`ed on first use. A failure is remembered: there
/// is no point retrying a missing library on every frame.
pub(crate) fn driver() -> Result<&'static CudaDriver, CudaError> {
    static DRIVER: OnceLock<Result<CudaDriver, CudaError>> = OnceLock::new();
    DRIVER.get_or_init(CudaDriver::load).as_ref().map_err(Clone::clone)
}
