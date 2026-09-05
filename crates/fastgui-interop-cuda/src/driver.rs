use std::ffi::CStr;
use std::os::raw::c_char;

use crate::error::CudaError;
use crate::sys::*;

/// Dynamically loads `nvcuda.dll` and resolves the handful of driver entry points this crate
/// needs. Loaded at runtime (not linked at build time) so the rest of the workspace keeps
/// building fine on machines without CUDA installed — the same trick `fastgui-render-vk` uses
/// for `vulkan-1.dll` via `ash::Entry::load()`. Fails with a clear error only when something
/// actually tries to use CUDA on a machine that doesn't have it.
pub(crate) struct CudaDriver {
    _lib: libloading::Library,
    pub get_error_string: PFN_cuGetErrorString,
    pub get_error_name: PFN_cuGetErrorName,
    pub init: PFN_cuInit,
    pub device_get: PFN_cuDeviceGet,
    pub device_primary_ctx_retain: PFN_cuDevicePrimaryCtxRetain,
    pub ctx_set_current: PFN_cuCtxSetCurrent,
    pub stream_create: PFN_cuStreamCreate,
    pub stream_synchronize: PFN_cuStreamSynchronize,
    pub stream_destroy: PFN_cuStreamDestroy,
    pub import_external_memory: PFN_cuImportExternalMemory,
    pub external_memory_get_mapped_buffer: PFN_cuExternalMemoryGetMappedBuffer,
    pub destroy_external_memory: PFN_cuDestroyExternalMemory,
    pub import_external_semaphore: PFN_cuImportExternalSemaphore,
    pub signal_external_semaphores_async: PFN_cuSignalExternalSemaphoresAsync,
    pub destroy_external_semaphore: PFN_cuDestroyExternalSemaphore,
}

macro_rules! resolve {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let symbol: libloading::Symbol<'_, $ty> = unsafe { $lib.get(concat!($name, "\0").as_bytes()) }
            .map_err(|e| CudaError::missing($name, e))?;
        *symbol
    }};
}

impl CudaDriver {
    pub fn load() -> Result<Self, CudaError> {
        let lib = unsafe { libloading::Library::new("nvcuda.dll")? };
        Ok(Self {
            get_error_string: resolve!(lib, "cuGetErrorString", PFN_cuGetErrorString),
            get_error_name: resolve!(lib, "cuGetErrorName", PFN_cuGetErrorName),
            init: resolve!(lib, "cuInit", PFN_cuInit),
            device_get: resolve!(lib, "cuDeviceGet", PFN_cuDeviceGet),
            device_primary_ctx_retain: resolve!(
                lib,
                "cuDevicePrimaryCtxRetain",
                PFN_cuDevicePrimaryCtxRetain
            ),
            ctx_set_current: resolve!(lib, "cuCtxSetCurrent", PFN_cuCtxSetCurrent),
            stream_create: resolve!(lib, "cuStreamCreate", PFN_cuStreamCreate),
            stream_synchronize: resolve!(lib, "cuStreamSynchronize", PFN_cuStreamSynchronize),
            stream_destroy: resolve!(lib, "cuStreamDestroy", PFN_cuStreamDestroy),
            import_external_memory: resolve!(
                lib,
                "cuImportExternalMemory",
                PFN_cuImportExternalMemory
            ),
            external_memory_get_mapped_buffer: resolve!(
                lib,
                "cuExternalMemoryGetMappedBuffer",
                PFN_cuExternalMemoryGetMappedBuffer
            ),
            destroy_external_memory: resolve!(
                lib,
                "cuDestroyExternalMemory",
                PFN_cuDestroyExternalMemory
            ),
            import_external_semaphore: resolve!(
                lib,
                "cuImportExternalSemaphore",
                PFN_cuImportExternalSemaphore
            ),
            signal_external_semaphores_async: resolve!(
                lib,
                "cuSignalExternalSemaphoresAsync",
                PFN_cuSignalExternalSemaphoresAsync
            ),
            destroy_external_semaphore: resolve!(
                lib,
                "cuDestroyExternalSemaphore",
                PFN_cuDestroyExternalSemaphore
            ),
            _lib: lib,
        })
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
