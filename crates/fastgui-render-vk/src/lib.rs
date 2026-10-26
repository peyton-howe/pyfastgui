//! Vulkan rendering backend (Windows + Linux) for fastgui.

mod app;
mod cuda_texture;
mod adapter;
mod d3d_import;
#[cfg(windows)]
mod d3d_probe;
mod dmabuf;
#[cfg(all(test, windows))]
mod cuda_test;
mod error;
mod pipeline;
mod quad;
#[cfg(test)]
mod quad_test;
mod renderer;
mod texture;
mod upload_bench;

pub use adapter::adapter_luid;
pub use app::{run, RunError};
pub use cuda_texture::CudaExportHandles;
pub use error::VkRendererError;
pub use fastgui_app::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
pub use renderer::VulkanRenderer;
pub use upload_bench::run_upload_bench;
