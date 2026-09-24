//! Metal rendering backend (macOS) for fastgui.

mod app;
mod error;
mod pipeline;
mod renderer;
mod texture;

pub use app::{run, RunError};
pub use error::MtlRendererError;
pub use fastgui_app::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
pub use renderer::MetalRenderer;
