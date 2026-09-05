//! Raw CUDA driver API ABI: types, structs, and function signatures for exactly the subset
//! we need (external memory/semaphore interop). Field layouts and function signatures are
//! transcribed from NVIDIA's own `cuda-python` bindings generator source
//! (`cuda_bindings/cuda/bindings/cydriver.pxd`, which mirrors `cuda.h` field-for-field) rather
//! than from memory, specifically to minimize the risk of an ABI mismatch in code that can't
//! be run against real CUDA hardware during development. Still: **unverified against a real
//! driver** — see the crate-level docs.

#![allow(non_camel_case_types, non_snake_case)]

use std::os::raw::{c_int, c_uint, c_void};

pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

#[repr(C)]
pub struct CUctx_st {
    _private: [u8; 0],
}
pub type CUcontext = *mut CUctx_st;

#[repr(C)]
pub struct CUstream_st {
    _private: [u8; 0],
}
pub type CUstream = *mut CUstream_st;

#[repr(C)]
pub struct CUextMemory_st {
    _private: [u8; 0],
}
pub type CUexternalMemory = *mut CUextMemory_st;

#[repr(C)]
pub struct CUextSemaphore_st {
    _private: [u8; 0],
}
pub type CUexternalSemaphore = *mut CUextSemaphore_st;

/// `CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: c_uint = 2;

/// `CUexternalSemaphoreHandleType_enum::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32`
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32: c_uint = 10;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaWin32Handle {
    pub handle: *mut c_void,
    pub name: *const c_void,
}

#[repr(C)]
pub union CudaExternalMemoryHandle {
    pub fd: c_int,
    pub win32: CudaWin32Handle,
    pub nv_sci_buf_object: *const c_void,
}

/// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st`
#[repr(C)]
pub struct CudaExternalMemoryHandleDesc {
    pub ty: c_uint,
    pub handle: CudaExternalMemoryHandle,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

/// `CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st`
#[repr(C)]
pub struct CudaExternalMemoryBufferDesc {
    pub offset: u64,
    pub size: u64,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
pub union CudaExternalSemaphoreHandle {
    pub fd: c_int,
    pub win32: CudaWin32Handle,
    pub nv_sci_sync_obj: *const c_void,
}

/// `CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st`
#[repr(C)]
pub struct CudaExternalSemaphoreHandleDesc {
    pub ty: c_uint,
    pub handle: CudaExternalSemaphoreHandle,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaFence {
    pub value: u64,
}

#[repr(C)]
pub union CudaNvSciSync {
    pub fence: *mut c_void,
    pub reserved: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaKeyedMutexSignal {
    pub key: u64,
}

#[repr(C)]
pub struct CudaExternalSemaphoreSignalParamsInner {
    pub fence: CudaFence,
    pub nv_sci_sync: CudaNvSciSync,
    pub keyed_mutex: CudaKeyedMutexSignal,
    pub reserved: [c_uint; 12],
}

/// `CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st`. Only `params.fence.value` is meaningful for our
/// use (signalling a timeline semaphore); every other field must be zeroed.
#[repr(C)]
pub struct CudaExternalSemaphoreSignalParams {
    pub params: CudaExternalSemaphoreSignalParamsInner,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

impl CudaExternalSemaphoreSignalParams {
    pub fn for_timeline_value(value: u64) -> Self {
        Self {
            params: CudaExternalSemaphoreSignalParamsInner {
                fence: CudaFence { value },
                nv_sci_sync: CudaNvSciSync { reserved: 0 },
                keyed_mutex: CudaKeyedMutexSignal { key: 0 },
                reserved: [0; 12],
            },
            flags: 0,
            reserved: [0; 16],
        }
    }
}

// Function pointer types for every entry point we dynamically load from nvcuda.dll. Signatures
// transcribed from cydriver.pxd's `cdef CUresult <name>(...)` declarations.
pub type PFN_cuGetErrorString =
    unsafe extern "system" fn(error: CUresult, pStr: *mut *const std::os::raw::c_char) -> CUresult;
pub type PFN_cuGetErrorName =
    unsafe extern "system" fn(error: CUresult, pStr: *mut *const std::os::raw::c_char) -> CUresult;
pub type PFN_cuInit = unsafe extern "system" fn(flags: c_uint) -> CUresult;
pub type PFN_cuDeviceGet =
    unsafe extern "system" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
pub type PFN_cuDevicePrimaryCtxRetain =
    unsafe extern "system" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
pub type PFN_cuCtxSetCurrent = unsafe extern "system" fn(ctx: CUcontext) -> CUresult;
pub type PFN_cuStreamCreate =
    unsafe extern "system" fn(phStream: *mut CUstream, flags: c_uint) -> CUresult;
pub type PFN_cuStreamSynchronize = unsafe extern "system" fn(hStream: CUstream) -> CUresult;
pub type PFN_cuStreamDestroy = unsafe extern "system" fn(hStream: CUstream) -> CUresult;
pub type PFN_cuImportExternalMemory = unsafe extern "system" fn(
    extMem_out: *mut CUexternalMemory,
    memHandleDesc: *const CudaExternalMemoryHandleDesc,
) -> CUresult;
pub type PFN_cuExternalMemoryGetMappedBuffer = unsafe extern "system" fn(
    devPtr: *mut CUdeviceptr,
    extMem: CUexternalMemory,
    bufferDesc: *const CudaExternalMemoryBufferDesc,
) -> CUresult;
pub type PFN_cuDestroyExternalMemory =
    unsafe extern "system" fn(extMem: CUexternalMemory) -> CUresult;
pub type PFN_cuImportExternalSemaphore = unsafe extern "system" fn(
    extSem_out: *mut CUexternalSemaphore,
    semHandleDesc: *const CudaExternalSemaphoreHandleDesc,
) -> CUresult;
pub type PFN_cuSignalExternalSemaphoresAsync = unsafe extern "system" fn(
    extSemArray: *const CUexternalSemaphore,
    paramsArray: *const CudaExternalSemaphoreSignalParams,
    numExtSems: c_uint,
    stream: CUstream,
) -> CUresult;
pub type PFN_cuDestroyExternalSemaphore =
    unsafe extern "system" fn(extSem: CUexternalSemaphore) -> CUresult;
