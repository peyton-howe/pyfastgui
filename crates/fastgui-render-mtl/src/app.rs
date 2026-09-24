use fastgui_app::{MainResizePolicy, RenderThreadHandles, RunError as AppRunError};

use crate::error::MtlRendererError;
use crate::renderer::MetalRenderer;

pub type RunError = AppRunError<MtlRendererError>;

/// Open a Metal window at logical `width`×`height` and block until it closes.
pub fn run(
    title: &str,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
) -> Result<(), RunError> {
    fastgui_app::run::<MetalRenderer>(
        title,
        width,
        height,
        clear_color,
        handles,
        MainResizePolicy::Immediate,
    )
}
