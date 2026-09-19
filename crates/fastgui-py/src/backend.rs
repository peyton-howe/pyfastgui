//! Picks `fastgui-render-mtl` on macOS and `fastgui-render-vk` elsewhere so the rest of this
//! crate can name `Command`, `CommandDispatch`, `RenderThreadHandles`, and `run` without a
//! `cfg` at every call site.

#[cfg(target_os = "macos")]
pub use fastgui_render_mtl::{run, Command, CommandDispatch, EventWaker, RenderThreadHandles};

#[cfg(not(target_os = "macos"))]
pub use fastgui_render_vk::{run, Command, CommandDispatch, EventWaker, RenderThreadHandles};
