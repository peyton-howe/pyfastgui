//! Click / keyboard / drag handling for M7 form controls (checkbox, radio, toggle, spin, scrub, combo).

use std::cell::RefCell;
use std::sync::Arc;

use fastgui_core::taffy::prelude::*;
use fastgui_core::widget::{
    clamp_range, decimal_quantum, round_to_decimals, spin_down_rect, spin_up_rect, PopupAnchor,
    PopupSide, WidgetId, WidgetKind, WidgetTree,
};

/// A change to the tree queued by a widget callback (see [`defer`]).
type TreeJob = Box<dyn FnOnce(&mut WidgetTree)>;

thread_local! {
    /// Tree changes queued by callbacks that fire while the tree is borrowed (a combo dropdown's
    /// `ListView` pick, a popup's `on_dismiss`), applied by [`with_tree`] once it's free.
    static DEFERRED: RefCell<Vec<TreeJob>> = const { RefCell::new(Vec::new()) };
}

/// Run input handling `f` on `tree`, then apply every tree change a callback queued meanwhile
/// (with [`defer`]). Callbacks fire from inside tree methods — `list_select`, `dismiss_popup` —
/// while the tree is borrowed, so they can't touch it directly; they queue the change instead,
/// and it lands before this returns, i.e. within the same input event.
pub fn with_tree<R>(tree: &mut WidgetTree, f: impl FnOnce(&mut WidgetTree) -> R) -> R {
    let result = f(tree);
    loop {
        let jobs = DEFERRED.with(|queue| std::mem::take(&mut *queue.borrow_mut()));
        if jobs.is_empty() {
            return result;
        }
        for job in jobs {
            job(tree);
        }
    }
}

/// Queue a tree change from inside a widget callback; the enclosing [`with_tree`] applies it.
fn defer(job: impl FnOnce(&mut WidgetTree) + 'static) {
    DEFERRED.with(|queue| queue.borrow_mut().push(Box::new(job)));
}

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
    if !set_radio_selected(tree, id, true) {
        return false;
    }
    if let Some(WidgetKind::Radio { on_select: Some(callback), .. }) = tree.kind(id) {
        callback();
    }
    true
}

/// Set a radio's selection without firing `on_select` (Python setters never fire callbacks).
/// Selecting clears peers in the same group; deselecting only clears this radio.
pub fn set_radio_selected(tree: &mut WidgetTree, id: WidgetId, selected: bool) -> bool {
    let Some(WidgetKind::Radio { selected: was, group_id, .. }) = tree.kind(id) else {
        return false;
    };
    if *was == selected {
        return false;
    }
    if !selected {
        tree.mutate_kind(id, |kind| {
            if let WidgetKind::Radio { selected, mirror, .. } = kind {
                *selected = false;
                if let Some(mirror) = mirror {
                    mirror.set(false);
                }
            }
        });
        return true;
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
            if let WidgetKind::Radio { selected, mirror, .. } = kind {
                *selected = peer == id;
                if let Some(mirror) = mirror {
                    mirror.set(*selected);
                }
            }
        });
    }
    true
}

/// Radios in `group_id` in tree-walk order (for arrow-key navigation).
pub fn radios_in_group(tree: &WidgetTree, group_id: u64) -> Vec<WidgetId> {
    tree.walk()
        .filter(|&id| matches!(tree.kind(id), Some(WidgetKind::Radio { group_id: g, .. }) if *g == group_id))
        .collect()
}

/// Set a `SpinBox` or `NumericScrub` value (clamped + rounded), update its mirror, fire `on_change`.
pub fn set_numeric(tree: &mut WidgetTree, id: WidgetId, new_value: f32) -> bool {
    let mut changed = false;
    let mut published = None;
    tree.mutate_kind(id, |kind| match kind {
        WidgetKind::SpinBox { value, min, max, decimals, mirror, .. } => {
            let next = round_to_decimals(clamp_range(new_value, *min, *max), *decimals);
            changed = *value != next;
            *value = next;
            if let Some(mirror) = mirror {
                mirror.set(next);
            }
            published = Some(next);
        }
        WidgetKind::NumericScrub { value, min, max, decimals, mirror, .. } => {
            let next = round_to_decimals(clamp_range(new_value, *min, *max), *decimals);
            changed = *value != next;
            *value = next;
            if let Some(mirror) = mirror {
                mirror.set(next);
            }
            published = Some(next);
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

/// Toggle the dropdown for `ComboBox` `combo_id`: close if open, otherwise open a non-modal
/// popup anchored below with a `ListView` of the combo's items.
pub fn open_combo(tree: &mut WidgetTree, combo_id: WidgetId) {
    let Some(WidgetKind::ComboBox { popup_id, .. }) = tree.kind(combo_id) else {
        return;
    };
    if let Some(popup) = *popup_id {
        if matches!(tree.kind(popup), Some(WidgetKind::Popup { .. })) {
            tree.close_popup(popup);
            tree.mutate_kind(combo_id, |kind| {
                if let WidgetKind::ComboBox { popup_id, .. } = kind {
                    *popup_id = None;
                }
            });
            return;
        }
        tree.mutate_kind(combo_id, |kind| {
            if let WidgetKind::ComboBox { popup_id, .. } = kind {
                *popup_id = None;
            }
        });
    }

    let (
        items,
        selected,
        font_size,
        text_color,
        background,
        border,
        selection_color,
    ) = match tree.kind(combo_id) {
        Some(WidgetKind::ComboBox {
            items,
            selected,
            font_size,
            text_color,
            background,
            border,
            selection_color,
            ..
        }) => (
            items.clone(),
            *selected,
            *font_size,
            *text_color,
            *background,
            *border,
            *selection_color,
        ),
        _ => return,
    };

    let width = tree.absolute_rect(combo_id).map(|r| r.width).unwrap_or(160.0);
    let row_height = (font_size * 1.25).max(18.0);
    // Open with the selected row in view (at the top when the list can scroll that far); the
    // dropdown shows up to 8 rows (`ListView`'s intrinsic height).
    let visible = items.len().clamp(1, 8);
    let scroll = selected.map_or(0.0, |i| i.min(items.len().saturating_sub(visible)) as f32 * row_height);

    let pick = Arc::new(move |index: usize| {
        defer(move |tree| apply_combo_pick(tree, combo_id, index));
    });
    let on_dismiss = Arc::new(move || {
        defer(move |tree| {
            tree.mutate_kind(combo_id, |kind| {
                if let WidgetKind::ComboBox { popup_id, .. } = kind {
                    *popup_id = None;
                }
            });
        });
    });

    let popup_kind = WidgetKind::Popup {
        anchor: PopupAnchor::Widget(combo_id, PopupSide::Below),
        modal: false,
        background,
        border,
        on_dismiss: Some(on_dismiss),
        restore_focus: Some(combo_id),
        click_through: false,
        open: None,
    };
    let list_style = Style {
        size: Size { width: Dimension::length(width), height: Dimension::auto() },
        ..Default::default()
    };
    let list_kind = WidgetKind::ListView {
        items,
        row_height,
        font_size,
        scroll,
        selected,
        text_color,
        background,
        selection_color,
        on_select: Some(pick.clone()),
        on_activate: Some(pick),
        mirror: None,
    };

    let popup = tree.open_popup(popup_kind, |tree, popup| {
        let list = tree.new_node(list_style, list_kind);
        tree.add_child(popup, list);
    });
    tree.mutate_kind(combo_id, |kind| {
        if let WidgetKind::ComboBox { popup_id, .. } = kind {
            *popup_id = Some(popup);
        }
    });
}

/// Apply a dropdown pick: update the combo, fire `on_change`, close its popup.
pub fn apply_combo_pick(tree: &mut WidgetTree, combo_id: WidgetId, index: usize) {
    let popup = match tree.kind(combo_id) {
        Some(WidgetKind::ComboBox { popup_id, items, .. }) => {
            if index >= items.len() {
                return;
            }
            *popup_id
        }
        _ => return,
    };
    let mut callback = None;
    tree.mutate_kind(combo_id, |kind| {
        if let WidgetKind::ComboBox { selected, mirror, on_change, popup_id, .. } = kind {
            *selected = Some(index);
            if let Some(mirror) = mirror {
                mirror.set(Some(index));
            }
            *popup_id = None;
            callback = on_change.clone();
        }
    });
    if let Some(popup) = popup {
        if matches!(tree.kind(popup), Some(WidgetKind::Popup { .. })) {
            tree.close_popup(popup);
        }
    }
    if let Some(callback) = callback {
        callback(index);
    }
}

/// The `ComboBox` that owns dropdown list `list_id`, if any.
pub fn combo_for_list(tree: &WidgetTree, list_id: WidgetId) -> Option<WidgetId> {
    let popup = tree.popup_of(list_id)?;
    let anchor = match tree.kind(popup)? {
        WidgetKind::Popup { anchor: PopupAnchor::Widget(id, _), .. } => *id,
        _ => return None,
    };
    match tree.kind(anchor) {
        Some(WidgetKind::ComboBox { popup_id: Some(open), .. }) if *open == popup => Some(anchor),
        _ => None,
    }
}

/// Add `delta` to a spin/scrub value. If rounding to `decimals` would leave the stored value
/// unchanged (e.g. `step=0.4`, `decimals=0`), bump by at least one display unit in `delta`'s
/// direction so arrows / ± buttons always move.
pub fn set_numeric_delta(tree: &mut WidgetTree, id: WidgetId, delta: f32) -> bool {
    let (current, min, max, decimals) = match tree.kind(id) {
        Some(WidgetKind::SpinBox { value, min, max, decimals, .. }) => (*value, *min, *max, *decimals),
        Some(WidgetKind::NumericScrub { value, min, max, decimals, .. }) => (*value, *min, *max, *decimals),
        _ => return false,
    };
    let mut target = current + delta;
    let next = round_to_decimals(clamp_range(target, min, max), decimals);
    if next == current && delta != 0.0 {
        target = current + decimal_quantum(decimals).copysign(delta);
    }
    set_numeric(tree, id, target)
}

/// Apply a horizontal scrub from the press origin: `start_value + dx * speed`.
/// Scrubbing from the origin (instead of per-event deltas) keeps slow drags moving after
/// `round_to_decimals` would otherwise swallow tiny steps.
pub fn scrub_from(tree: &mut WidgetTree, id: WidgetId, start_value: f32, dx: f32) -> bool {
    let speed = match tree.kind(id) {
        Some(WidgetKind::NumericScrub { speed, .. }) => *speed,
        Some(WidgetKind::SpinBox { step, .. }) => *step * 0.1,
        _ => return false,
    };
    set_numeric(tree, id, start_value + dx * speed)
}

/// Current numeric value of a `SpinBox` / `NumericScrub`, for recording a scrub press origin.
pub fn numeric_value(tree: &WidgetTree, id: WidgetId) -> Option<f32> {
    match tree.kind(id) {
        Some(WidgetKind::SpinBox { value, .. } | WidgetKind::NumericScrub { value, .. }) => Some(*value),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressResult {
    Ignored,
    Handled(bool),
    /// Caller should track drag and call `scrub_from`.
    StartScrub,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

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
                    mirror: Some(fastgui_core::Readback::new(selected)),
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
    fn selecting_a_radio_clears_peer_mirrors() {
        // Python's `Radio.selected` reads the mirror, and a rebuild describes the radio from it:
        // a peer cleared without its mirror came back selected after `set_theme`.
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let mirrors: Vec<_> = [true, false, false].into_iter().map(fastgui_core::Readback::new).collect();
        let ids: Vec<_> = mirrors
            .iter()
            .map(|mirror| {
                let id = tree.new_node(
                    fixed(80.0, 20.0),
                    WidgetKind::Radio {
                        selected: mirror.get(),
                        label: "r".into(),
                        group_id: 3,
                        font_size: 14.0,
                        text_color: Color::TRANSPARENT,
                        box_color: Color::TRANSPARENT,
                        dot_color: Color::TRANSPARENT,
                        on_select: None,
                        mirror: Some(mirror.clone()),
                    },
                );
                tree.add_child(root, id);
                id
            })
            .collect();
        assert!(select_radio(&mut tree, ids[2]));
        assert_eq!(mirrors.iter().map(|m| m.get()).collect::<Vec<_>>(), [false, false, true]);
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

    #[test]
    fn spin_delta_steps_at_least_one_display_unit() {
        // step=0.4 with decimals=0 used to round 0+0.4 back to 0 and stick.
        let mirror = fastgui_core::Readback::new(0.0);
        let mut tree = WidgetTree::new();
        let id = tree.new_node(
            fixed(100.0, 28.0),
            WidgetKind::SpinBox {
                value: 0.0,
                min: 0.0,
                max: 10.0,
                step: 0.4,
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
        assert!(set_numeric_delta(&mut tree, id, 0.4));
        assert_eq!(mirror.get(), 1.0);
        assert!(set_numeric_delta(&mut tree, id, 0.4));
        assert_eq!(mirror.get(), 2.0);

        // decimals=1, step below one tenth: bump by 0.1.
        let mirror = fastgui_core::Readback::new(0.0);
        let id = tree.new_node(
            fixed(100.0, 28.0),
            WidgetKind::SpinBox {
                value: 0.0,
                min: 0.0,
                max: 10.0,
                step: 0.04,
                decimals: 1,
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                button_color: Color::TRANSPARENT,
                on_change: None,
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(tree.root(), id);
        assert!(set_numeric_delta(&mut tree, id, 0.04));
        assert!((mirror.get() - 0.1).abs() < 1e-5);

        // step=0.6, decimals=0 already rounds to a 1-unit move.
        let mirror = fastgui_core::Readback::new(0.0);
        let id = tree.new_node(
            fixed(100.0, 28.0),
            WidgetKind::SpinBox {
                value: 0.0,
                min: 0.0,
                max: 10.0,
                step: 0.6,
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
        assert!(set_numeric_delta(&mut tree, id, 0.6));
        assert_eq!(mirror.get(), 1.0);
    }

    fn combo_kind(items: Vec<String>, selected: Option<usize>, on_change: Option<fastgui_core::widget::IndexCallback>) -> WidgetKind {
        WidgetKind::ComboBox {
            items,
            selected,
            placeholder: "Choose…".into(),
            font_size: 14.0,
            text_color: Color::TRANSPARENT,
            placeholder_color: Color::TRANSPARENT,
            background: Color([0.16, 0.17, 0.2, 1.0]),
            border: Color([0.32, 0.35, 0.42, 1.0]),
            selection_color: Color([0.25, 0.45, 0.80, 0.6]),
            on_change,
            mirror: None,
            popup_id: None,
        }
    }

    #[test]
    fn open_combo_sets_popup_and_pick_closes() {
        let seen = Arc::new(AtomicUsize::new(99));
        let sink = seen.clone();
        let mut tree = WidgetTree::new();
        let combo = tree.new_node(
            fixed(160.0, 28.0),
            combo_kind(
                vec!["alpha".into(), "beta".into(), "gamma".into()],
                None,
                Some(Arc::new(move |i| sink.store(i, Ordering::SeqCst))),
            ),
        );
        tree.add_child(tree.root(), combo);
        tree.compute_layout(400.0, 400.0);
        open_combo(&mut tree, combo);
        let popup = match tree.kind(combo) {
            Some(WidgetKind::ComboBox { popup_id: Some(id), .. }) => *id,
            _ => panic!("expected open popup"),
        };
        assert!(matches!(tree.kind(popup), Some(WidgetKind::Popup { .. })));
        let list = tree
            .walk()
            .find(|&id| {
                matches!(tree.kind(id), Some(WidgetKind::ListView { .. })) && tree.popup_of(id) == Some(popup)
            })
            .expect("list inside popup");
        with_tree(&mut tree, |tree| {
            tree.list_select(list, Some(1));
        });
        assert!(tree.topmost_popup().is_none(), "pick closes the popup");
        assert!(matches!(
            tree.kind(combo),
            Some(WidgetKind::ComboBox { selected: Some(1), popup_id: None, .. })
        ));
        assert_eq!(seen.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn open_combo_shows_the_selected_row() {
        let items: Vec<String> = (0..200).map(|i| format!("row {i}")).collect();
        let mut tree = WidgetTree::new();
        let combo = tree.new_node(fixed(160.0, 28.0), combo_kind(items, Some(150), None));
        tree.add_child(tree.root(), combo);
        tree.compute_layout(400.0, 800.0);
        open_combo(&mut tree, combo);
        tree.compute_layout(400.0, 800.0);
        let list = tree.walk().find(|&id| matches!(tree.kind(id), Some(WidgetKind::ListView { .. }))).unwrap();
        assert!(tree.list_visible_rows(list).contains(&150), "rows {:?}", tree.list_visible_rows(list));
        tree.list_highlight(list, 170);
        tree.compute_layout(400.0, 800.0);
        assert!(tree.list_visible_rows(list).contains(&170), "highlight scrolled into view: {:?}", tree.list_visible_rows(list));
    }

    #[test]
    fn open_combo_toggles_closed() {
        let mut tree = WidgetTree::new();
        let combo = tree.new_node(fixed(160.0, 28.0), combo_kind(vec!["a".into()], Some(0), None));
        tree.add_child(tree.root(), combo);
        tree.compute_layout(400.0, 400.0);
        open_combo(&mut tree, combo);
        assert!(matches!(tree.kind(combo), Some(WidgetKind::ComboBox { popup_id: Some(_), .. })));
        open_combo(&mut tree, combo);
        assert!(matches!(tree.kind(combo), Some(WidgetKind::ComboBox { popup_id: None, .. })));
        assert!(tree.topmost_popup().is_none());
    }
}
