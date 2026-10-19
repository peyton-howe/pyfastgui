//! Raw CUDA driver API ABI: types, structs, and function signatures for exactly the subset
//! we need (external memory/semaphore interop, streams/events, 2D copies, host callbacks).
//! Field layouts, enum values and function signatures are transcribed from NVIDIA's own
//! `cuda-python` bindings (`cuda/bindings/cydriver.pxd`, which mirrors `cuda.h` field-for-field;
//! enum values read from `cuda.bindings.driver`, cuda-bindings 13.4.3) rather than from memory,
//! to minimize the risk of an ABI mismatch in code that can't be run against real CUDA hardware
//! during development. The size assertions at the bottom pin the layouts down further. Still:
//! **unverified against a real driver** — see the crate-level docs.

#![allow(non_camel_case_types, non_snake_case)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

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
pub struct CUevent_st {
    _private: [u8; 0],
}
pub type CUevent = *mut CUevent_st;

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

/// `CUuuid_st`
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CUuuid {
    pub bytes: [c_char; 16],
}

/// `CUexternalMemoryHandleType_enum::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`
pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32: c_uint = 2;
/// `CUDA_EXTERNAL_MEMORY_DEDICATED`: the imported memory is a dedicated allocation.
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: c_uint = 1;

/// `CUexternalSemaphoreHandleType_enum::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32`
pub const CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32: c_uint = 10;

/// `CUstream_flags_enum::CU_STREAM_NON_BLOCKING`: no implicit sync with the legacy default stream.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;
/// `CUevent_flags_enum::CU_EVENT_DISABLE_TIMING`
pub const CU_EVENT_DISABLE_TIMING: c_uint = 2;

/// `CUmemorytype_enum`
pub type CUmemorytype = c_uint;
pub const CU_MEMORYTYPE_HOST: CUmemorytype = 1;
pub const CU_MEMORYTYPE_DEVICE: CUmemorytype = 2;
/// Any UVA pointer (device, host-pinned or managed); the driver works out which.
pub const CU_MEMORYTYPE_UNIFIED: CUmemorytype = 4;

/// `CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`
pub const CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: c_uint = 9;

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

/// The anonymous `params` struct of the signal params (`anon_pod22`).
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CudaKeyedMutexWait {
    pub key: u64,
    pub timeout_ms: c_uint,
}

/// The anonymous `params` struct of the wait params (`anon_pod26`): same as the signal one
/// except the keyed mutex carries a timeout, so the padding is two words shorter.
#[repr(C)]
pub struct CudaExternalSemaphoreWaitParamsInner {
    pub fence: CudaFence,
    pub nv_sci_sync: CudaNvSciSync,
    pub keyed_mutex: CudaKeyedMutexWait,
    pub reserved: [c_uint; 10],
}

/// `CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st`. Only `params.fence.value` is meaningful for a
/// timeline semaphore (wait until the counter reaches it).
#[repr(C)]
pub struct CudaExternalSemaphoreWaitParams {
    pub params: CudaExternalSemaphoreWaitParamsInner,
    pub flags: c_uint,
    pub reserved: [c_uint; 16],
}

impl CudaExternalSemaphoreWaitParams {
    pub fn for_timeline_value(value: u64) -> Self {
        Self {
            params: CudaExternalSemaphoreWaitParamsInner {
                fence: CudaFence { value },
                nv_sci_sync: CudaNvSciSync { reserved: 0 },
                keyed_mutex: CudaKeyedMutexWait { key: 0, timeout_ms: 0 },
                reserved: [0; 10],
            },
            flags: 0,
            reserved: [0; 16],
        }
    }
}

#[repr(C)]
pub struct CUarray_st {
    _private: [u8; 0],
}
pub type CUarray = *mut CUarray_st;

/// `CUDA_MEMCPY2D_st` (`CUDA_MEMCPY2D_v2`, the layout `cuMemcpy2DAsync_v2` takes).
#[repr(C)]
pub struct CudaMemcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: CUmemorytype,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: CUarray,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: CUmemorytype,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: CUarray,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}

pub type CUhostFn = unsafe extern "system" fn(user_data: *mut c_void);

// Function pointer types for every entry point we dynamically load. Signatures transcribed from
// cydriver.pxd's `cdef CUresult <name>(...)` declarations; where cuda.h `#define`s a name to a
// `_v2` symbol, the `_v2` export is loaded (see driver.rs) and has the same signature.
pub type PFN_cuGetErrorString = unsafe extern "system" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult;
pub type PFN_cuGetErrorName = unsafe extern "system" fn(error: CUresult, pStr: *mut *const c_char) -> CUresult;
pub type PFN_cuInit = unsafe extern "system" fn(flags: c_uint) -> CUresult;
pub type PFN_cuDeviceGetCount = unsafe extern "system" fn(count: *mut c_int) -> CUresult;
pub type PFN_cuDeviceGet = unsafe extern "system" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult;
pub type PFN_cuDeviceGetUuid = unsafe extern "system" fn(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
pub type PFN_cuDevicePrimaryCtxRetain = unsafe extern "system" fn(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
pub type PFN_cuDevicePrimaryCtxRelease = unsafe extern "system" fn(dev: CUdevice) -> CUresult;
pub type PFN_cuCtxPushCurrent = unsafe extern "system" fn(ctx: CUcontext) -> CUresult;
pub type PFN_cuCtxPopCurrent = unsafe extern "system" fn(pctx: *mut CUcontext) -> CUresult;
pub type PFN_cuPointerGetAttribute =
    unsafe extern "system" fn(data: *mut c_void, attribute: c_uint, ptr: CUdeviceptr) -> CUresult;
pub type PFN_cuStreamCreate = unsafe extern "system" fn(phStream: *mut CUstream, flags: c_uint) -> CUresult;
pub type PFN_cuStreamSynchronize = unsafe extern "system" fn(hStream: CUstream) -> CUresult;
pub type PFN_cuStreamDestroy = unsafe extern "system" fn(hStream: CUstream) -> CUresult;
pub type PFN_cuStreamWaitEvent =
    unsafe extern "system" fn(hStream: CUstream, hEvent: CUevent, flags: c_uint) -> CUresult;
pub type PFN_cuEventCreate = unsafe extern "system" fn(phEvent: *mut CUevent, flags: c_uint) -> CUresult;
pub type PFN_cuEventRecord = unsafe extern "system" fn(hEvent: CUevent, hStream: CUstream) -> CUresult;
pub type PFN_cuEventDestroy = unsafe extern "system" fn(hEvent: CUevent) -> CUresult;
pub type PFN_cuMemcpy2DAsync = unsafe extern "system" fn(pCopy: *const CudaMemcpy2D, hStream: CUstream) -> CUresult;
pub type PFN_cuMemFree = unsafe extern "system" fn(dptr: CUdeviceptr) -> CUresult;
pub type PFN_cuLaunchHostFunc =
    unsafe extern "system" fn(hStream: CUstream, func: CUhostFn, userData: *mut c_void) -> CUresult;
pub type PFN_cuImportExternalMemory = unsafe extern "system" fn(
    extMem_out: *mut CUexternalMemory,
    memHandleDesc: *const CudaExternalMemoryHandleDesc,
) -> CUresult;
pub type PFN_cuExternalMemoryGetMappedBuffer = unsafe extern "system" fn(
    devPtr: *mut CUdeviceptr,
    extMem: CUexternalMemory,
    bufferDesc: *const CudaExternalMemoryBufferDesc,
) -> CUresult;
pub type PFN_cuDestroyExternalMemory = unsafe extern "system" fn(extMem: CUexternalMemory) -> CUresult;
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
pub type PFN_cuWaitExternalSemaphoresAsync = unsafe extern "system" fn(
    extSemArray: *const CUexternalSemaphore,
    paramsArray: *const CudaExternalSemaphoreWaitParams,
    numExtSems: c_uint,
    stream: CUstream,
) -> CUresult;
pub type PFN_cuDestroyExternalSemaphore = unsafe extern "system" fn(extSem: CUexternalSemaphore) -> CUresult;

// Layout checks (64-bit). Every offset below was measured on the real `cuda.h` layout by writing
// a field through cuda-bindings 13.4.3 (compiled against cuda.h) and finding it in the struct's
// raw bytes. The signal and wait params are both 144 bytes: cuda.h pads their `params` structs
// to the same 72 bytes. CUDA_MEMCPY2D is 16 eight-byte slots (each 4-byte enum is padded by the
// pointer after it).
#[cfg(target_pointer_width = "64")]
const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<CudaExternalSemaphoreSignalParamsInner>() == 72);
    assert!(size_of::<CudaExternalSemaphoreWaitParamsInner>() == 72);
    assert!(size_of::<CudaExternalSemaphoreSignalParams>() == 144);
    assert!(size_of::<CudaExternalSemaphoreWaitParams>() == 144);
    assert!(offset_of!(CudaExternalSemaphoreSignalParams, flags) == 72);
    assert!(offset_of!(CudaExternalSemaphoreWaitParams, flags) == 72);
    assert!(offset_of!(CudaExternalSemaphoreWaitParamsInner, keyed_mutex) + offset_of!(CudaKeyedMutexWait, timeout_ms) == 24);
    assert!(size_of::<CudaMemcpy2D>() == 128);
    assert!(offset_of!(CudaMemcpy2D, src_host) == 24);
    assert!(offset_of!(CudaMemcpy2D, dst_memory_type) == 72);
    assert!(offset_of!(CudaMemcpy2D, dst_pitch) == 104);
    assert!(offset_of!(CudaMemcpy2D, height) == 120);
    assert!(offset_of!(CudaExternalMemoryHandleDesc, size) == 24);
    assert!(offset_of!(CudaExternalMemoryHandleDesc, flags) == 32);
    assert!(size_of::<CudaExternalMemoryHandleDesc>() == 104);
    assert!(offset_of!(CudaExternalMemoryBufferDesc, size) == 8);
    assert!(offset_of!(CudaExternalMemoryBufferDesc, flags) == 16);
    assert!(size_of::<CudaExternalMemoryBufferDesc>() == 88);
    assert!(offset_of!(CudaExternalSemaphoreHandleDesc, flags) == 24);
    assert!(size_of::<CudaExternalSemaphoreHandleDesc>() == 96);
    assert!(size_of::<CUuuid>() == 16 && align_of::<CUuuid>() == 1);
};
