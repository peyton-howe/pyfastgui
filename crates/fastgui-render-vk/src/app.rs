use fastgui_app::{MainResizePolicy, RenderThreadHandles, RunError as AppRunError};

use crate::error::VkRendererError;
use crate::renderer::VulkanRenderer;

pub type RunError = AppRunError<VkRendererError>;

/// Open a Vulkan window at logical `width`×`height` and block until it closes.
pub fn run(
    title: &str,
    width: u32,
    height: u32,
    clear_color: [f32; 4],
    handles: RenderThreadHandles,
) -> Result<(), RunError> {
    fastgui_app::run::<VulkanRenderer>(
        title,
        width,
        height,
        clear_color,
        handles,
        MainResizePolicy::Debounced,
    )
}
