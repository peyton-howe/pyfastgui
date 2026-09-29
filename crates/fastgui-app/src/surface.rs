use std::time::Duration;

use fastgui_core::widget::Rect;
use fastgui_core::{ChromeFrame, ChromeQuads, CpuFrame};
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

/// One `Viewport` widget's frame to composite over the chrome, in **physical** pixels. `rect` is
/// where the whole frame goes — it may reach past the window's edges when the viewport is
/// scrolled partly out of sight — and `visible` is the part of it that may actually be drawn
/// (inside every enclosing `ScrollArea`).
#[derive(Clone, Copy, Debug)]
pub struct ViewportDraw {
    pub viewport_id: u64,
    pub rect: Rect,
    pub visible: Rect,
}

impl ViewportDraw {
    /// The GPU viewport (`[x, y, width, height]`, the whole frame, whole pixels, possibly
    /// off-target) and scissor (`[x, y, width, height]`, the visible part within a
    /// `target_width`×`target_height` render target), or `None` when nothing is visible.
    /// Clipping with the scissor instead of shrinking the viewport keeps a partly hidden frame
    /// at its size, cut off rather than squashed.
    pub fn viewport_and_scissor(&self, target_width: u32, target_height: u32) -> Option<([f32; 4], [u32; 4])> {
        let (x, y) = (self.rect.x.round(), self.rect.y.round());
        let (w, h) = (self.rect.width.round(), self.rect.height.round());
        if w < 1.0 || h < 1.0 {
            return None;
        }
        let target = Rect { x: 0.0, y: 0.0, width: target_width as f32, height: target_height as f32 };
        let shown = Rect { x, y, width: w, height: h }.intersect(&self.visible).intersect(&target);
        let (sx0, sy0) = (shown.x.round(), shown.y.round());
        let (sx1, sy1) = ((shown.x + shown.width).round(), (shown.y + shown.height).round());
        if sx1 <= sx0 || sy1 <= sy0 {
            return None;
        }
        Some(([x, y, w, h], [sx0 as u32, sy0 as u32, (sx1 - sx0) as u32, (sy1 - sy0) as u32]))
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

    /// Take the chrome as GPU quads + atlas uploads instead (see
    /// `fastgui_chrome::ChromeRenderer::build_quads`); replaces any chrome frame, and vice versa.
    fn set_chrome_quads(&mut self, frame: &ChromeQuads<'_>) -> Result<(), Self::Error>;

    fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Self::Error>;

    fn retain_layers(&mut self, live_ids: &[u64]);

    /// The size of what `render_frame` actually draws into, in physical pixels. It can differ
    /// from the last size passed to `resize`: a swapchain rebuilt after `OUT_OF_DATE` takes the
    /// surface's current extent (X11 reports the new size mid-resize), so the app remaps
    /// viewport rects against this rather than what it last asked for.
    fn surface_size(&self) -> (u32, u32);

    fn render_frame(&mut self, clear: [f32; 4], draw_chrome: bool, draws: &[ViewportDraw]) -> Result<(), Self::Error>;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolled_viewport_is_cut_not_squashed() {
        let rect = Rect { x: 10.0, y: -40.0, width: 200.0, height: 100.0 };
        let visible = Rect { x: 0.0, y: 0.0, width: 150.0, height: 500.0 };
        let draw = ViewportDraw { viewport_id: 1, rect, visible };
        let (viewport, scissor) = draw.viewport_and_scissor(800, 600).unwrap();
        assert_eq!(viewport, [10.0, -40.0, 200.0, 100.0], "full size, partly above the window");
        assert_eq!(scissor, [10, 0, 140, 60], "only the part inside the clip and the target");
        let hidden = ViewportDraw { visible: Rect { x: 300.0, ..visible }, ..draw };
        assert!(hidden.viewport_and_scissor(800, 600).is_none());
    }
}
