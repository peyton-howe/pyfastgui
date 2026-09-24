//! Metal rendering backend (macOS) for fastgui.

mod app;
mod command;
mod error;
mod pipeline;
mod renderer;
mod texture;

pub use app::{run, RunError};
pub use command::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
pub use error::MtlRendererError;
pub use renderer::MetalRenderer;
