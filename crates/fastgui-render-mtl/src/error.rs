use thiserror::Error;

#[derive(Debug, Error)]
pub enum MtlRendererError {
    #[error("failed to obtain a window/display handle: {0}")]
    Handle(#[from] raw_window_handle::HandleError),
    #[error("the window's raw handle is not an AppKit (NSView) handle")]
    NotAppKit,
    #[error("no Metal-capable GPU is available on this Mac")]
    NoDevice,
    #[error("failed to create a Metal command queue")]
    NoCommandQueue,
    #[error("shader compilation failed: {0}")]
    ShaderCompile(String),
    #[error("shader is missing the expected function '{0}'")]
    MissingFunction(&'static str),
    #[error("failed to create the render pipeline state: {0}")]
    PipelineState(String),
    #[error("failed to allocate a Metal texture")]
    NoTexture,
    #[error("failed to create a Metal command buffer")]
    NoCommandBuffer,
    #[error("failed to create a Metal render command encoder")]
    NoEncoder,
    #[error("GPU command buffer execution failed: {0}")]
    CommandBufferFailed(String),
    #[error("the render thread hasn't finished starting up yet")]
    RendererNotReady,
}
