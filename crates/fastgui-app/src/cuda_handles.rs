use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Win32 handles exported for CUDA external-memory import (Vulkan backend only).
///
/// Kept in `fastgui-app` so [`crate::Command::CreateCudaSurface`] does not depend on the
/// Vulkan crate. Metal never constructs this command (macOS stubs `create_cuda_surface` in
/// Python before anything is sent).
pub struct CudaExportHandles {
    pub memory_win32_handle: isize,
    pub memory_size: u64,
    pub semaphore_win32_handle: isize,
    pub row_pitch: u64,
    pub width: u32,
    pub height: u32,
    /// Highest timeline value a CUDA producer has signalled; polled each frame on the render thread.
    pub target_value: Arc<AtomicU64>,
}
