//! Cross-thread command queue, cached-readback, CPU-frame-mailbox, and widget-tree primitives.
//!
//! M1 added the generic command queue + readback plumbing. M2 added `FrameSlot`/`CpuFrame`
//! for the Viewport's CPU texture upload path — a different sharing pattern (latest-wins,
//! drops stale frames) than the command queue (in-order, never drops). M4 adds the retained
//! widget tree (`widget` module), backed by `taffy` for layout; `fastgui-chrome` rasterizes it
//! and `fastgui-render-vk` displays the result the same way it displays a CPU-uploaded
//! `Viewport` frame. See ROADMAP.md for the milestone breakdown.

mod frame;
pub mod plot;
mod queue;
mod readback;
pub mod text_edit;
pub mod theme;
pub mod widget;

pub use frame::{
    AtlasUpload, ChromeFrame, ChromeQuad, ChromeQuads, CpuFrame, FrameSlot, PixelFormat, PixelRect,
    MAX_CPU_FRAME_EXTENT, QUAD_CIRCLE, QUAD_CIRCLE_CLIPPED, QUAD_SOLID, QUAD_SPRITE,
};
pub use plot::{raster_heatmap, raster_line, raster_scatter, AxisRange, Colormap, PlotStyle};
pub use queue::{command_channel, oneshot_channel, CommandReceiver, CommandSender, OneshotReceiver, OneshotSender};
pub use readback::Readback;

pub use taffy;
