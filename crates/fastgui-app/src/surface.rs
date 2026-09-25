use std::time::Duration;

use fastgui_core::widget::Rect;
use fastgui_core::{ChromeFrame, CpuFrame};
use winit::window::Window;

use crate::constants::RESIZE_DEBOUNCE;

/// How the shared app reacts to main-window `Resized` events.
#[derive(Clone, Copy, Debug)]
pub enum MainResizePolicy {
    /// Metal: resize the surface and redraw immediately.
    Immediate,
    /// Vulkan: update bookkeeping; recreate the swapchain after idle debounce.
    Debounced,
}

impl MainResizePolicy {
    pub fn debounce(self) -> Option<Duration> {
        match self {
            MainResizePolicy::Immediate => None,
            MainResizePolicy::Debounced => Some(RESIZE_DEBOUNCE),
        }
    }
}

/// GPU surface bound to one winit `Window` (main, floater, or tear ghost).
pub trait SurfaceBackend: Sized {
    type Error: std::error::Error + Send + Sync + 'static;

    fn new(window: &Window, physical_width: u32, physical_height: u32) -> Result<Self, Self::Error>;

    fn resize(&mut self, physical_width: u32, physical_height: u32) -> Result<(), Self::Error>;

    /// Update the chrome texture. When the texture already holds the previous frame at this
    /// size, only `frame.damage` needs copying; otherwise (or when `damage` is `None`) upload
    /// all of `frame.data`.
    fn set_chrome_frame(&mut self, frame: &ChromeFrame<'_>) -> Result<(), Self::Error>;

    fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Self::Error>;

    fn retain_layers(&mut self, live_ids: &[u64]);

    /// `draws` rects are always in **physical** pixels.
    fn render_frame(
        &mut self,
        clear: [f32; 4],
        draw_chrome: bool,
        draws: &[(u64, Rect)],
    ) -> Result<(), Self::Error>;

    /// Optional CUDA path — Vulkan implements; Metal returns an error string.
    fn create_cuda_surface(
        &mut self,
        _viewport_id: u64,
        _width: u32,
        _height: u32,
    ) -> Result<crate::CudaExportHandles, String> {
        Err("CUDA interop is not available on this backend".into())
    }
}
