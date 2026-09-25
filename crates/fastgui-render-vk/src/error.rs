use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum VkRendererError {
    #[error("failed to load the Vulkan library: {0}")]
    Load(#[from] ash::LoadingError),
    #[error("Vulkan call failed: {0}")]
    Vk(#[from] vk::Result),
    #[error("failed to obtain a window/display handle: {0}")]
    Handle(#[from] raw_window_handle::HandleError),
    #[error(
        "no Vulkan physical device supports graphics + presentation to this surface and \
         Vulkan 1.3 dynamic rendering"
    )]
    NoSuitablePhysicalDevice,
    #[error("surface reported no supported formats")]
    NoSurfaceFormat,
    #[error("no host-visible memory type available for the viewport texture")]
    NoHostVisibleTextureMemory,
    #[error("no host-visible memory type available for the chrome instance buffer")]
    NoHostVisibleBufferMemory,
    #[error("no device-local memory type available for the CUDA-shared texture")]
    NoDeviceLocalTextureMemory,
    #[error("failed to read shader SPIR-V: {0}")]
    ShaderCode(#[from] std::io::Error),
    #[error(
        "this GPU/driver doesn't support the win32 external memory/semaphore extensions \
         CUDA interop needs (VK_KHR_external_memory_win32, VK_KHR_external_semaphore_win32)"
    )]
    CudaInteropUnsupported,
    #[error(
        "CUDA interop is not implemented on Linux yet -- only the Windows (win32 handle) \
         path exists. Use Viewport.submit_frame() (CPU copy) instead."
    )]
    CudaInteropNotImplemented,
    #[error("the render thread hasn't finished starting up yet")]
    RendererNotReady,
}
