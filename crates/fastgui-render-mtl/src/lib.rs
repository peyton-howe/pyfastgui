//! Metal rendering backend (macOS) for fastgui.
//!
//! Empty on every other target (its dependencies are macOS-only too — see Cargo.toml), so a
//! plain `cargo test --workspace` / `cargo clippy --workspace` works on Windows and Linux.
#![cfg(target_os = "macos")]

mod app;
mod error;
mod import;
mod pipeline;
#[cfg(test)]
mod quad_test;
mod renderer;
mod texture;

pub use app::{run, RunError};
pub use error::MtlRendererError;
pub use import::prepare_metal_import;
pub use renderer::run_import_window_bench;
pub use fastgui_app::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
pub use renderer::MetalRenderer;
