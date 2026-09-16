//! Picks the platform-specific render-backend crate (`fastgui-render-vk` on Windows/Linux,
//! `fastgui-render-mtl` on macOS) so the rest of this crate can refer to `Command`,
//! `CommandDispatch`, `RenderThreadHandles`, and `run` without a `cfg` at every call site.
//!
//! The two backends' `Command` enums differ (only `fastgui-render-vk`'s has a
//! `CreateCudaSurface` variant -- macOS has no zero-copy CUDA<->Metal path to expose, see
//! `fastgui_render_mtl::Command`'s doc comment); that variant is only ever referenced from
//! `Viewport::create_cuda_surface`'s own `#[cfg(not(target_os = "macos"))]` arm, so the mismatch
//! never surfaces here.

#[cfg(target_os = "macos")]
pub use fastgui_render_mtl::{run, Command, CommandDispatch, EventWaker, RenderThreadHandles};

#[cfg(not(target_os = "macos"))]
pub use fastgui_render_vk::{run, Command, CommandDispatch, EventWaker, RenderThreadHandles};
