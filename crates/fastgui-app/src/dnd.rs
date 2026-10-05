//! The drag gesture shared by dock drags and widget drag-and-drop, and OS file drops.
//!
//! Dock drags (panel title bars, tabs) and widget drags (`WidgetTree::set_drag_source`) start the
//! same way — a press that moves past [`past_drag_threshold`] — and preview the same way: the
//! app hands chrome one drop-preview rect per frame, whichever kind of drag produced it. The
//! dock keeps its own region/zone/tear-off logic; widget drags are the [`WidgetDrag`] state
//! machine below, with targets resolved by `WidgetTree::drop_target_at`.

use std::path::PathBuf;

use fastgui_core::dnd::{DragOrigin, DropCallback, DropEvent, DropHit};
use fastgui_core::widget::{Rect, WidgetId, WidgetTree};

use crate::constants::TEAR_GHOST_THRESHOLD;

/// Whether the cursor has moved far enough from `press` for a press to count as a drag. Below
/// it a click that jitters a pixel stays a click.
pub fn past_drag_threshold(press: (f32, f32), cursor: (f32, f32)) -> bool {
    let (dx, dy) = (cursor.0 - press.0, cursor.1 - press.1);
    dx * dx + dy * dy >= TEAR_GHOST_THRESHOLD * TEAR_GHOST_THRESHOLD
}

/// A widget drag: nothing, pressed on a drag source but not moved far enough yet, or under way.
#[derive(Default)]
pub(crate) enum WidgetDrag {
    #[default]
    Idle,
    Pending { source: WidgetId, origin: DragOrigin, press: (f32, f32) },
    Active { tag: String, data: Vec<u8>, hit: Option<DropHit> },
}

impl WidgetDrag {
    /// A press at `cursor`: arm a drag if it landed on an (enabled) drag source.
    pub fn press(tree: &WidgetTree, cursor: (f32, f32)) -> Self {
        match tree.drag_origin_at(cursor.0, cursor.1) {
            Some((source, origin, _)) => WidgetDrag::Pending { source, origin, press: cursor },
            None => WidgetDrag::Idle,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, WidgetDrag::Active { .. })
    }

    /// The cursor moved (with the button held). A pending drag starts once past the threshold,
    /// unless `other_drag` (a slider, text selection... the press also started) has it; the
    /// source's data callback runs then, and returning `None` cancels. Returns whether the drop
    /// preview changed.
    pub fn moved(&mut self, tree: &WidgetTree, cursor: (f32, f32), other_drag: bool) -> bool {
        match self {
            WidgetDrag::Idle => false,
            WidgetDrag::Pending { source, origin, press } => {
                if other_drag {
                    *self = WidgetDrag::Idle;
                    return false;
                }
                if !past_drag_threshold(*press, cursor) {
                    return false;
                }
                let started = tree.drag_origin_at(press.0, press.1).filter(|(id, ..)| id == source).and_then(
                    |(_, _, drag_source)| Some((drag_source.tag.clone(), (drag_source.data)(origin)?)),
                );
                *self = match started {
                    Some((tag, data)) => {
                        let hit = tree.drop_target_at(cursor.0, cursor.1, &tag);
                        WidgetDrag::Active { tag, data, hit }
                    }
                    None => WidgetDrag::Idle,
                };
                true
            }
            WidgetDrag::Active { tag, hit, .. } => {
                let new_hit = tree.drop_target_at(cursor.0, cursor.1, tag);
                let changed = new_hit.as_ref().map(|h| h.preview) != hit.as_ref().map(|h| h.preview);
                *hit = new_hit;
                changed
            }
        }
    }

    /// The button came up: end the drag. Returns the drop to deliver, if it was over a target.
    pub fn release(&mut self, tree: &WidgetTree) -> Option<(DropCallback, DropEvent)> {
        let WidgetDrag::Active { tag, data, hit: Some(hit) } = std::mem::take(self) else { return None };
        let target = tree.drop_target(hit.target)?;
        Some((target.on_drop.clone(), DropEvent { tag, data, position: hit.position }))
    }

    /// Escape: drop the drag without delivering it. Returns whether one was under way.
    pub fn cancel(&mut self) -> bool {
        let was_active = self.is_active();
        *self = WidgetDrag::Idle;
        was_active
    }

    /// The rect the drop preview highlights, while over a target.
    pub fn preview(&self) -> Option<Rect> {
        match self {
            WidgetDrag::Active { hit: Some(hit), .. } => Some(hit.preview),
            _ => None,
        }
    }

    /// Over a target that would take the drop.
    pub fn has_target(&self) -> bool {
        self.preview().is_some()
    }
}

/// Files from one OS drop gesture arrive one `DroppedFile` event each; collect them and deliver
/// them together once the event burst is over.
#[derive(Default)]
pub(crate) struct FileDrops {
    paths: Vec<PathBuf>,
}

impl FileDrops {
    pub fn push(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Deliver the collected files to the file-drop target at `cursor`. Returns whether a
    /// target took them; files over no target are dropped.
    pub fn flush(&mut self, tree: &WidgetTree, cursor: (f32, f32)) -> bool {
        if self.paths.is_empty() {
            return false;
        }
        let paths = std::mem::take(&mut self.paths);
        match tree.file_drop_at(cursor.0, cursor.1) {
            Some((callback, x, y)) => {
                callback(paths, x, y);
                true
            }
            None => false,
        }
    }
}

/// The cursor in the window's logical coordinates, from the OS rather than the last
/// `CursorMoved`: during an OS drag-and-drop the window gets no cursor events, so the last one
/// is wherever the cursor entered. `None` where the platform can't say (then use the last).
#[cfg(target_os = "windows")]
pub(crate) fn os_cursor(window: &winit::window::Window, scale_factor: f64) -> Option<(f32, f32)> {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;
    let mut point = POINT { x: 0, y: 0 };
    // SAFETY: plain out-parameter call.
    if unsafe { GetCursorPos(&mut point) } == 0 {
        return None;
    }
    let origin = window.inner_position().ok()?;
    let scale = scale_factor.max(0.01);
    Some((((point.x - origin.x) as f64 / scale) as f32, ((point.y - origin.y) as f64 / scale) as f32))
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn os_cursor(_window: &winit::window::Window, _scale_factor: f64) -> Option<(f32, f32)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_is_a_radius() {
        assert!(!past_drag_threshold((10.0, 10.0), (13.0, 13.0)), "~4.2px is still a click");
        assert!(past_drag_threshold((10.0, 10.0), (16.0, 10.0)));
        assert!(past_drag_threshold((10.0, 10.0), (10.0, 4.0)));
    }
}
