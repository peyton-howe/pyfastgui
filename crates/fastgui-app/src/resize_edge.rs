use winit::window::CursorIcon;

use crate::constants::FLOAT_RESIZE_MARGIN;

/// Which edge/corner of a floating OS window is being resized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeEdge {
    Left,
    Right,
    Top,
    Bottom,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub fn classify_float_resize_edge(width: f32, height: f32, x: f32, y: f32) -> Option<ResizeEdge> {
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    let left = x <= FLOAT_RESIZE_MARGIN;
    let right = x >= width - FLOAT_RESIZE_MARGIN;
    let top = y <= FLOAT_RESIZE_MARGIN;
    let bottom = y >= height - FLOAT_RESIZE_MARGIN;
    match (left, right, top, bottom) {
        (true, _, true, _) => Some(ResizeEdge::TopLeft),
        (_, true, true, _) => Some(ResizeEdge::TopRight),
        (true, _, _, true) => Some(ResizeEdge::BottomLeft),
        (_, true, _, true) => Some(ResizeEdge::BottomRight),
        (true, _, _, _) => Some(ResizeEdge::Left),
        (_, true, _, _) => Some(ResizeEdge::Right),
        (_, _, true, _) => Some(ResizeEdge::Top),
        (_, _, _, true) => Some(ResizeEdge::Bottom),
        _ => None,
    }
}

pub fn resize_edge_cursor(edge: ResizeEdge) -> CursorIcon {
    match edge {
        ResizeEdge::Left | ResizeEdge::Right => CursorIcon::ColResize,
        ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::RowResize,
        ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
        ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
    }
}
