//! Shared window ApplicationHandler for Metal and Vulkan backends.
//!
//! Backends implement [`surface::SurfaceBackend`] and call [`run`]; dock/float/ghost input and
//! the command queue live here once so the two `app.rs` files cannot drift.

mod app;
mod cloak;
mod command;
mod constants;
mod coords;
mod cuda_handles;
mod ghost;
mod resize_edge;
mod surface;

pub use app::{run, RunError};
pub use command::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
pub use constants::*;
pub use coords::scale_rect;
pub use cuda_handles::CudaExportHandles;
pub use ghost::build_tear_ghost_tree;
pub use resize_edge::{classify_float_resize_edge, resize_edge_cursor, ResizeEdge};
pub use surface::{MainResizePolicy, SurfaceBackend};
