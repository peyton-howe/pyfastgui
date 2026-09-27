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

/// Map a physical-pixel rect laid out for a `layout`-sized window onto a `surface`-sized one,
/// the same linear stretch chrome gets while a debounced resize hasn't reached the swapchain
/// yet (`quad.frag`'s remap / the full-surface chrome texture). Identity once they match.
pub fn remap_rect(rect: Rect, layout: (u32, u32), surface: (u32, u32)) -> Rect {
    if layout == surface || layout.0 == 0 || layout.1 == 0 {
        return rect;
    }
    let sx = surface.0 as f32 / layout.0 as f32;
    let sy = surface.1 as f32 / layout.1 as f32;
    Rect { x: rect.x * sx, y: rect.y * sy, width: rect.width * sx, height: rect.height * sy }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_rect_stretches_layout_onto_surface() {
        let fields = |r: Rect| (r.x, r.y, r.width, r.height);
        let rect = Rect { x: 100.0, y: 50.0, width: 400.0, height: 300.0 };
        assert_eq!(fields(remap_rect(rect, (800, 600), (800, 600))), fields(rect));
        assert_eq!(fields(remap_rect(rect, (1600, 1200), (800, 600))), (50.0, 25.0, 200.0, 150.0));
    }
}
