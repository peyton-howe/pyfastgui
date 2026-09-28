//! Click / keyboard / drag handling for M7 form controls (checkbox, radio, toggle, spin, scrub).

use fastgui_core::widget::{
    spin_down_rect, spin_up_rect, WidgetId, WidgetKind, WidgetTree,
};

/// Toggle a `Checkbox` or `Toggle` and fire `on_change`.
pub fn toggle_bool(tree: &mut WidgetTree, id: WidgetId) -> bool {
    let mut new_value = None;
    tree.mutate_kind(id, |kind| match kind {
        WidgetKind::Checkbox { checked, .. } | WidgetKind::Toggle { checked, .. } => {
            *checked = !*checked;
            new_value = Some(*checked);
        }
        _ => {}
    });
    let Some(value) = new_value else { return false };
    if let Some(WidgetKind::Checkbox { on_change: Some(callback), .. } | WidgetKind::Toggle { on_change: Some(callback), .. }) =
        tree.kind(id)
    {
        callback(value);
    }
    true
}

/// Select a `Radio`, clear peers that share its `group_id`, and fire `on_select` if it changed.
pub fn select_radio(tree: &mut WidgetTree, id: WidgetId) -> bool {
    let Some(WidgetKind::Radio { selected, group_id, .. }) = tree.kind(id) else {
        return false;
    };
    if *selected {
        return false;
    }
    let group_id = *group_id;
    let peers: Vec<_> = tree
        .walk()
        .filter(|&peer| {
            matches!(tree.kind(peer), Some(WidgetKind::Radio { group_id: g, .. }) if *g == group_id)
        })
        .collect();
    for peer in peers {
        tree.mutate_kind(peer, |kind| {
            if let WidgetKind::Radio { selected, .. } = kind {
                *selected = peer == id;
            }
        });
    }
    if let Some(WidgetKind::Radio { on_select: Some(callback), .. }) = tree.kind(id) {
        callback();
    }
    true
}

/// Set a `SpinBox` or `NumericScrub` value (clamped), update its mirror, fire `on_change`.
pub fn set_numeric(tree: &mut WidgetTree, id: WidgetId, new_value: f32) -> bool {
    let mut changed = false;
    let mut published = None;
    tree.mutate_kind(id, |kind| match kind {
        WidgetKind::SpinBox { value, min, max, mirror, .. } => {
            let clamped = new_value.clamp((*min).min(*max), (*max).max(*min));
            changed = *value != clamped;
            *value = clamped;
            if let Some(mirror) = mirror {
                mirror.set(clamped);
            }
            published = Some(clamped);
        }
        WidgetKind::NumericScrub { value, min, max, mirror, .. } => {
            let clamped = new_value.clamp((*min).min(*max), (*max).max(*min));
            changed = *value != clamped;
            *value = clamped;
            if let Some(mirror) = mirror {
                mirror.set(clamped);
            }
            published = Some(clamped);
        }
        _ => {}
    });
    if !changed {
        return false;
    }
    if let Some(value) = published {
        if let Some(
            WidgetKind::SpinBox { on_change: Some(callback), .. }
            | WidgetKind::NumericScrub { on_change: Some(callback), .. },
        ) = tree.kind(id)
        {
            callback(value);
        }
    }
    true
}

/// Click handling for form controls. `x` is only needed for SpinBox button hit-testing.
pub fn handle_press(tree: &mut WidgetTree, id: WidgetId, x: f32, y: f32) -> PressResult {
    match tree.kind(id) {
        Some(WidgetKind::Checkbox { .. } | WidgetKind::Toggle { .. }) => {
            PressResult::Handled(toggle_bool(tree, id))
        }
        Some(WidgetKind::Radio { .. }) => PressResult::Handled(select_radio(tree, id)),
        Some(WidgetKind::SpinBox { .. }) => {
            let Some(rect) = tree.absolute_rect(id) else {
                return PressResult::Ignored;
            };
            if spin_up_rect(rect).contains(x, y) {
                let step = match tree.kind(id) {
                    Some(WidgetKind::SpinBox { step, .. }) => *step,
                    _ => return PressResult::Ignored,
                };
                PressResult::Handled(set_numeric_delta(tree, id, step))
            } else if spin_down_rect(rect).contains(x, y) {
                let step = match tree.kind(id) {
                    Some(WidgetKind::SpinBox { step, .. }) => *step,
                    _ => return PressResult::Ignored,
                };
                PressResult::Handled(set_numeric_delta(tree, id, -step))
            } else {
                // Value area: treat like a scrub start (caller tracks drag).
                PressResult::StartScrub
            }
        }
        Some(WidgetKind::NumericScrub { .. }) => PressResult::StartScrub,
        _ => PressResult::Ignored,
    }
}

fn set_numeric_delta(tree: &mut WidgetTree, id: WidgetId, delta: f32) -> bool {
    let current = match tree.kind(id) {
        Some(WidgetKind::SpinBox { value, .. } | WidgetKind::NumericScrub { value, .. }) => *value,
        _ => return false,
    };
    set_numeric(tree, id, current + delta)
}

/// Apply a horizontal scrub drag: `dx` logical pixels since press (or last update).
pub fn scrub_by(tree: &mut WidgetTree, id: WidgetId, dx: f32) -> bool {
    let (value, speed) = match tree.kind(id) {
        Some(WidgetKind::NumericScrub { value, speed, .. }) => (*value, *speed),
        Some(WidgetKind::SpinBox { value, step, .. }) => (*value, *step * 0.1),
        _ => return false,
    };
    set_numeric(tree, id, value + dx * speed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressResult {
    Ignored,
    Handled(bool),
    /// Caller should track drag and call `scrub_by`.
    StartScrub,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use fastgui_core::taffy::prelude::*;
    use fastgui_core::widget::{Color, WidgetKind};

    use super::*;

    fn fixed(w: f32, h: f32) -> Style {
        Style { size: Size { width: Dimension::length(w), height: Dimension::length(h) }, ..Default::default() }
    }

    #[test]
    fn checkbox_toggles_and_fires() {
        let seen = Arc::new(AtomicBool::new(false));
        let sink = seen.clone();
        let mut tree = WidgetTree::new();
        let id = tree.new_node(
            fixed(100.0, 20.0),
            WidgetKind::Checkbox {
                checked: false,
                label: "x".into(),
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                box_color: Color::TRANSPARENT,
                check_color: Color::TRANSPARENT,
                on_change: Some(Arc::new(move |v| sink.store(v, Ordering::SeqCst))),
            },
        );
        tree.add_child(tree.root(), id);
        assert!(toggle_bool(&mut tree, id));
        assert!(seen.load(Ordering::SeqCst));
        let Some(WidgetKind::Checkbox { checked: true, .. }) = tree.kind(id) else { panic!() };
    }

    #[test]
    fn radio_group_is_exclusive() {
        let clicks = Arc::new(AtomicUsize::new(0));
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let make = |tree: &mut WidgetTree, selected: bool, counter: Arc<AtomicUsize>| {
            let id = tree.new_node(
                fixed(80.0, 20.0),
                WidgetKind::Radio {
                    selected,
                    label: "r".into(),
                    group_id: 7,
                    font_size: 14.0,
                    text_color: Color::TRANSPARENT,
                    box_color: Color::TRANSPARENT,
                    dot_color: Color::TRANSPARENT,
                    on_select: Some(Arc::new(move || {
                        counter.fetch_add(1, Ordering::SeqCst);
                    })),
                },
            );
            tree.add_child(root, id);
            id
        };
        let a = make(&mut tree, true, clicks.clone());
        let b = make(&mut tree, false, clicks.clone());
        assert!(select_radio(&mut tree, b));
        assert!(matches!(tree.kind(a), Some(WidgetKind::Radio { selected: false, .. })));
        assert!(matches!(tree.kind(b), Some(WidgetKind::Radio { selected: true, .. })));
        assert_eq!(clicks.load(Ordering::SeqCst), 1);
        assert!(!select_radio(&mut tree, b), "already selected is a no-op");
    }

    #[test]
    fn spin_clamps_and_mirrors() {
        let mirror = fastgui_core::Readback::new(0.0);
        let mut tree = WidgetTree::new();
        let id = tree.new_node(
            fixed(100.0, 28.0),
            WidgetKind::SpinBox {
                value: 5.0,
                min: 0.0,
                max: 10.0,
                step: 1.0,
                decimals: 0,
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                button_color: Color::TRANSPARENT,
                on_change: None,
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(tree.root(), id);
        tree.compute_layout(200.0, 200.0);
        assert!(set_numeric(&mut tree, id, 99.0));
        assert_eq!(mirror.get(), 10.0);
        assert!(!set_numeric(&mut tree, id, 10.0));
    }
}
