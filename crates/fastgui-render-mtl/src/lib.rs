//! Metal rendering backend (macOS) for fast-gui.
//!
//! No CUDA interop path -- Apple hasn't shipped an NVIDIA GPU since ~2019, so there is no
//! zero-copy CUDA<->Metal surface to build (see `command::Command`'s doc comment).
//! `Viewport.submit_frame()`'s CPU-copy path works unchanged.

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
