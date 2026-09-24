use fastgui_core::widget::Rect;

/// Main-window (or floater) size in layout units — always **logical points** inside `fastgui-app`.
#[derive(Clone, Copy, Debug)]
pub struct LayoutSize {
    pub width: f32,
    pub height: f32,
}

/// Scale a layout (logical) rect into physical pixels for GPU draws / chrome.
pub fn scale_rect(rect: Rect, scale: f32) -> Rect {
    Rect {
        x: rect.x * scale,
        y: rect.y * scale,
        width: rect.width * scale,
        height: rect.height * scale,
    }
}

pub fn physical_to_logical(x: f32, y: f32, scale: f64) -> (f32, f32) {
    let s = scale.max(0.01) as f32;
    (x / s, y / s)
}

pub fn logical_to_physical_rect(rect: Rect, scale: f64) -> Rect {
    scale_rect(rect, scale.max(0.01) as f32)
}
