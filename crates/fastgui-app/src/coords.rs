use fastgui_core::widget::Rect;

/// Scale a layout (logical) rect into physical pixels for GPU draws / chrome.
pub fn scale_rect(rect: Rect, scale: f32) -> Rect {
    Rect {
        x: rect.x * scale,
        y: rect.y * scale,
        width: rect.width * scale,
        height: rect.height * scale,
    }
}
