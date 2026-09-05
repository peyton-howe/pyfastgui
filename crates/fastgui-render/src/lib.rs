//! Backend-agnostic rendering traits. `fastgui-render-vk` (Vulkan) and, later,
//! `fastgui-render-mtl` (Metal) implement `Renderer` against a native window handle.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub trait Renderer: Sized {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a renderer targeting the given window surface at its initial pixel size.
    fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Result<Self, Self::Error>;

    /// Notify the renderer that the surface's pixel size changed (e.g. on window resize).
    fn resize(&mut self, width: u32, height: u32) -> Result<(), Self::Error>;

    /// Render one frame: clears the surface to `color` (linear RGBA), then draws whatever
    /// backend-specific content (e.g. a Vulkan `Viewport` texture) the implementor is
    /// currently holding on top of it.
    fn render_frame(&mut self, color: [f32; 4]) -> Result<(), Self::Error>;
}
