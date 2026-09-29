//! Keyboard focus traversal and activation, shared by the main window and floaters (each has
//! its own `WidgetTree`, so its own focus).

use fastgui_core::text_edit::TextMeasure;
use fastgui_core::widget::{WidgetKind, WidgetTree};
use winit::keyboard::{Key, NamedKey};

use crate::text_input::{self, Clipboard, KeyPress};

/// Fraction of a slider's range one arrow press moves it; PageUp/PageDown move ten steps.
const SLIDER_STEP: f32 = 0.01;
const SLIDER_PAGE: f32 = 0.1;

/// Apply one key press to `tree`: Tab / Shift+Tab move focus, Escape clears it, and the
/// focused widget handles the rest (Enter/Space click a `Button`; arrows, PageUp/PageDown and
/// Home/End move a `Slider`; a `TextInput` / `TextArea` edits — see `text_input::handle_key`). Returns
/// whether anything happened, i.e. whether to redraw.
pub fn handle_key(
    tree: &mut WidgetTree,
    press: &KeyPress<'_>,
    measure: &mut dyn TextMeasure,
    clipboard: &mut dyn Clipboard,
) -> bool {
    crate::forms::with_tree(tree, |tree| handle_key_inner(tree, press, measure, clipboard))
}

fn handle_key_inner(
    tree: &mut WidgetTree,
    press: &KeyPress<'_>,
    measure: &mut dyn TextMeasure,
    clipboard: &mut dyn Clipboard,
) -> bool {
    let key = press.key;
    match key {
        Key::Named(NamedKey::Tab) => {
            tree.focus_next(press.modifiers.shift_key());
            if let Some(id) = tree.focused() {
                tree.scroll_into_view(id);
                text_input::scroll_caret_into_view(tree, id, measure);
            }
            return true;
        }
        // Escape closes the topmost popup first (menus, dialogs, tooltips), then clears focus.
        Key::Named(NamedKey::Escape) if tree.dismiss_popup() => return true,
        Key::Named(NamedKey::Escape) if tree.focused().is_some() => {
            tree.set_focus(None);
            return true;
        }
        _ => {}
    }
    // Menu / MenuBar shortcuts — before the focused widget eats the key (text fields still
    // keep A/C/V/X/Z; see `accel::try_fire`).
    if crate::accel::try_fire(tree, press) {
        return true;
    }
    let Some(id) = tree.focused() else { return false };
    match tree.kind(id) {
        Some(WidgetKind::Button { on_click, .. }) => {
            let activate = matches!(key, Key::Named(NamedKey::Enter | NamedKey::Space))
                || matches!(key, Key::Character(c) if c == " ");
            if activate {
                if let Some(callback) = on_click.clone() {
                    callback();
                }
            }
            activate
        }
        Some(&WidgetKind::Slider { value, min, max, .. }) => {
            let range = max - min;
            let target = match key {
                Key::Named(NamedKey::ArrowLeft | NamedKey::ArrowDown) => value - range * SLIDER_STEP,
                Key::Named(NamedKey::ArrowRight | NamedKey::ArrowUp) => value + range * SLIDER_STEP,
                Key::Named(NamedKey::PageDown) => value - range * SLIDER_PAGE,
                Key::Named(NamedKey::PageUp) => value + range * SLIDER_PAGE,
                Key::Named(NamedKey::Home) => min,
                Key::Named(NamedKey::End) => max,
                _ => return false,
            };
            set_slider_value(tree, id, target.clamp(min.min(max), max.max(min)));
            true
        }
        Some(WidgetKind::TextInput { .. } | WidgetKind::TextArea { .. }) => {
            text_input::handle_key(tree, id, press, measure, clipboard)
        }
        Some(WidgetKind::ListView { items, selected, on_activate, .. }) => {
            let combo = crate::forms::combo_for_list(tree, id);
            let (count, current) = (items.len(), *selected);
            let page = tree.list_page_rows(id);
            let last = count.saturating_sub(1);
            // Combo dropdown: arrows move the highlight without firing on_select (which closes).
            if combo.is_some() {
                let activate = matches!(key, Key::Named(NamedKey::Enter | NamedKey::Space))
                    || matches!(key, Key::Character(c) if c == " ");
                if activate {
                    if let (Some(callback), Some(row)) = (on_activate.clone(), current) {
                        callback(row);
                    }
                    return current.is_some();
                }
                let target = match key {
                    Key::Named(NamedKey::ArrowDown) => current.map_or(0, |i| (i + 1).min(last)),
                    Key::Named(NamedKey::ArrowUp) => current.map_or(0, |i| i.saturating_sub(1)),
                    Key::Named(NamedKey::PageDown) => current.map_or(0, |i| (i + page).min(last)),
                    Key::Named(NamedKey::PageUp) => current.map_or(0, |i| i.saturating_sub(page)),
                    Key::Named(NamedKey::Home) => 0,
                    Key::Named(NamedKey::End) => last,
                    _ => return false,
                };
                if count == 0 {
                    return false;
                }
                // Scrolls too: past the 8 visible rows the highlight used to move out of sight.
                tree.list_highlight(id, target);
                return true;
            }
            let target = match key {
                Key::Named(NamedKey::Enter) => {
                    if let (Some(callback), Some(row)) = (on_activate.clone(), current) {
                        callback(row);
                    }
                    return current.is_some();
                }
                Key::Named(NamedKey::ArrowDown) => current.map_or(0, |i| (i + 1).min(last)),
                Key::Named(NamedKey::ArrowUp) => current.map_or(0, |i| i.saturating_sub(1)),
                Key::Named(NamedKey::PageDown) => current.map_or(0, |i| (i + page).min(last)),
                Key::Named(NamedKey::PageUp) => current.map_or(0, |i| i.saturating_sub(page)),
                Key::Named(NamedKey::Home) => 0,
                Key::Named(NamedKey::End) => last,
                _ => return false,
            };
            tree.list_select(id, Some(target));
            true
        }
        Some(WidgetKind::Checkbox { .. } | WidgetKind::Toggle { .. }) => {
            let activate = matches!(key, Key::Named(NamedKey::Enter | NamedKey::Space))
                || matches!(key, Key::Character(c) if c == " ");
            if activate {
                crate::forms::toggle_bool(tree, id);
            }
            activate
        }
        Some(WidgetKind::Radio { group_id, .. }) => {
            let group_id = *group_id;
            let activate = matches!(key, Key::Named(NamedKey::Enter | NamedKey::Space))
                || matches!(key, Key::Character(c) if c == " ");
            if activate {
                crate::forms::select_radio(tree, id);
                return true;
            }
            let step = match key {
                Key::Named(NamedKey::ArrowUp | NamedKey::ArrowLeft) => -1isize,
                Key::Named(NamedKey::ArrowDown | NamedKey::ArrowRight) => 1,
                _ => return false,
            };
            let peers = crate::forms::radios_in_group(tree, group_id);
            let Some(index) = peers.iter().position(|&peer| peer == id) else {
                return false;
            };
            let next = index as isize + step;
            if next < 0 || next as usize >= peers.len() {
                return false;
            }
            let target = peers[next as usize];
            crate::forms::select_radio(tree, target);
            tree.set_focus(Some(target));
            true
        }
        Some(WidgetKind::SpinBox { min, max, step, .. }) => {
            let (min, max, step) = (*min, *max, *step);
            match key {
                Key::Named(NamedKey::ArrowUp | NamedKey::ArrowRight) => {
                    crate::forms::set_numeric_delta(tree, id, step)
                }
                Key::Named(NamedKey::ArrowDown | NamedKey::ArrowLeft) => {
                    crate::forms::set_numeric_delta(tree, id, -step)
                }
                Key::Named(NamedKey::PageUp) => crate::forms::set_numeric_delta(tree, id, step * 10.0),
                Key::Named(NamedKey::PageDown) => crate::forms::set_numeric_delta(tree, id, -step * 10.0),
                Key::Named(NamedKey::Home) => crate::forms::set_numeric(tree, id, min),
                Key::Named(NamedKey::End) => crate::forms::set_numeric(tree, id, max),
                _ => false,
            }
        }
        Some(WidgetKind::NumericScrub { min, max, speed, .. }) => {
            let (min, max, speed) = (*min, *max, *speed);
            match key {
                Key::Named(NamedKey::ArrowUp | NamedKey::ArrowRight) => {
                    crate::forms::set_numeric_delta(tree, id, speed)
                }
                Key::Named(NamedKey::ArrowDown | NamedKey::ArrowLeft) => {
                    crate::forms::set_numeric_delta(tree, id, -speed)
                }
                Key::Named(NamedKey::PageUp) => crate::forms::set_numeric_delta(tree, id, speed * 10.0),
                Key::Named(NamedKey::PageDown) => crate::forms::set_numeric_delta(tree, id, -speed * 10.0),
                Key::Named(NamedKey::Home) => crate::forms::set_numeric(tree, id, min),
                Key::Named(NamedKey::End) => crate::forms::set_numeric(tree, id, max),
                _ => false,
            }
        }
        Some(WidgetKind::ComboBox { .. }) => {
            let open = matches!(key, Key::Named(NamedKey::Enter | NamedKey::Space | NamedKey::ArrowDown))
                || matches!(key, Key::Character(c) if c == " ");
            if open {
                crate::forms::open_combo(tree, id);
            }
            open
        }
        _ => false,
    }
}

/// Set a slider's value and fire its `on_change` — unless the value didn't move (already at
/// an end), so holding an arrow key against the stop doesn't spam the callback.
fn set_slider_value(tree: &mut WidgetTree, id: fastgui_core::widget::WidgetId, new_value: f32) {
    let mut changed = false;
    tree.mutate_kind(id, |kind| {
        if let WidgetKind::Slider { value, .. } = kind {
            changed = *value != new_value;
            *value = new_value;
        }
    });
    if !changed {
        return;
    }
    if let Some(WidgetKind::Slider { on_change: Some(callback), .. }) = tree.kind(id) {
        callback(new_value);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use fastgui_core::taffy::prelude::*;
    use fastgui_core::widget::{Color, WidgetId};
    use winit::keyboard::ModifiersState;

    use super::*;

    struct NoMeasure;

    impl TextMeasure for NoMeasure {
        fn caret_x(&mut self, _: &str, _: f32, _: usize) -> f32 {
            0.0
        }

        fn index_at(&mut self, _: &str, _: f32, _: f32) -> usize {
            0
        }
    }

    struct NoClipboard;

    impl Clipboard for NoClipboard {
        fn get(&mut self) -> Option<String> {
            None
        }

        fn set(&mut self, _: String) {}
    }

    fn press(tree: &mut WidgetTree, key: Key, shift: bool) -> bool {
        let modifiers = if shift { ModifiersState::SHIFT } else { ModifiersState::empty() };
        handle_key(tree, &KeyPress { key: &key, text: None, modifiers }, &mut NoMeasure, &mut NoClipboard)
    }

    fn fixed(w: f32, h: f32) -> Style {
        Style { size: Size { width: Dimension::length(w), height: Dimension::length(h) }, ..Default::default() }
    }

    fn tree_with(kinds: Vec<WidgetKind>) -> (WidgetTree, Vec<WidgetId>) {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let ids: Vec<_> = kinds.into_iter().map(|k| tree.new_node(fixed(100.0, 20.0), k)).collect();
        for &id in &ids {
            tree.add_child(root, id);
        }
        tree.compute_layout(300.0, 300.0);
        (tree, ids)
    }

    fn slider(value: f32, on_change: Option<fastgui_core::widget::ChangeCallback>) -> WidgetKind {
        WidgetKind::Slider {
            value,
            min: 0.0,
            max: 2.0,
            track_color: Color::TRANSPARENT,
            thumb_color: Color::TRANSPARENT,
            on_change,
        }
    }

    #[test]
    fn enter_and_space_click_focused_button() {
        let clicks = Arc::new(AtomicUsize::new(0));
        let counter = clicks.clone();
        let (mut tree, ids) = tree_with(vec![WidgetKind::Button {
            text: "Go".into(),
            font_size: 12.0,
            text_color: Color::TRANSPARENT,
            background: Color::TRANSPARENT,
            flat: false,
            on_click: Some(Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            })),
        }]);
        assert!(!press(&mut tree, Key::Named(NamedKey::Enter), false), "nothing focused");
        tree.set_focus(Some(ids[0]));
        assert!(press(&mut tree, Key::Named(NamedKey::Enter), false));
        assert!(press(&mut tree, Key::Named(NamedKey::Space), false));
        assert!(!press(&mut tree, Key::Character("a".into()), false));
        assert_eq!(clicks.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn arrows_step_slider_and_stop_at_ends() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let (mut tree, ids) = tree_with(vec![slider(1.0, Some(Arc::new(move |v| sink.lock().unwrap().push(v))))]);
        tree.set_focus(Some(ids[0]));
        press(&mut tree, Key::Named(NamedKey::ArrowRight), false);
        press(&mut tree, Key::Named(NamedKey::PageDown), false);
        press(&mut tree, Key::Named(NamedKey::End), false);
        press(&mut tree, Key::Named(NamedKey::ArrowUp), false);
        let values = seen.lock().unwrap().clone();
        assert_eq!(values.len(), 3, "the press against the max stop fires nothing: {values:?}");
        assert!((values[0] - 1.02).abs() < 1e-5 && (values[1] - 0.82).abs() < 1e-5 && values[2] == 2.0);
    }

    #[test]
    fn escape_closes_the_popup_before_clearing_focus() {
        use fastgui_core::widget::{PopupAnchor, PopupSide};
        let (mut tree, ids) = tree_with(vec![slider(0.0, None)]);
        tree.set_focus(Some(ids[0]));
        let popup_kind = WidgetKind::Popup {
            anchor: PopupAnchor::Widget(ids[0], PopupSide::Below),
            modal: false,
            background: Color::TRANSPARENT,
            border: Color::TRANSPARENT,
            on_dismiss: None,
            restore_focus: None,
            open: None,
        };
        let inner = std::cell::Cell::new(None);
        tree.open_popup(popup_kind, |tree, popup| {
            let s = tree.new_node(fixed(80.0, 20.0), slider(0.0, None));
            tree.add_child(popup, s);
            inner.set(Some(s));
        });
        tree.compute_layout(300.0, 300.0);
        assert_eq!(tree.focused(), inner.get(), "focus moves into the popup");
        assert!(press(&mut tree, Key::Named(NamedKey::Escape), false));
        assert!(tree.topmost_popup().is_none());
        assert_eq!(tree.focused(), Some(ids[0]), "focus comes back to where it was");
        assert!(press(&mut tree, Key::Named(NamedKey::Escape), false));
        assert_eq!(tree.focused(), None, "a second Escape clears focus");
    }

    #[test]
    fn list_keys_move_selection_and_enter_activates() {
        let activated = Arc::new(Mutex::new(Vec::new()));
        let sink = activated.clone();
        let (mut tree, ids) = tree_with(vec![WidgetKind::ListView {
            items: (0..30).map(|i| i.to_string()).collect(),
            row_height: 5.0,
            font_size: 12.0,
            scroll: 0.0,
            selected: None,
            text_color: Color::TRANSPARENT,
            background: Color::TRANSPARENT,
            selection_color: Color::TRANSPARENT,
            on_select: None,
            on_activate: Some(Arc::new(move |i| sink.lock().unwrap().push(i))),
            mirror: None,
        }]);
        tree.set_focus(Some(ids[0]));
        let selected = |tree: &WidgetTree| match tree.kind(ids[0]) {
            Some(WidgetKind::ListView { selected, .. }) => *selected,
            _ => panic!(),
        };
        assert!(!press(&mut tree, Key::Named(NamedKey::Enter), false), "nothing selected to activate");
        press(&mut tree, Key::Named(NamedKey::ArrowDown), false);
        assert_eq!(selected(&tree), Some(0), "the first Down selects the first row");
        press(&mut tree, Key::Named(NamedKey::PageDown), false);
        assert_eq!(selected(&tree), Some(4), "a page is the 20-tall list's 4 rows");
        press(&mut tree, Key::Named(NamedKey::End), false);
        press(&mut tree, Key::Named(NamedKey::ArrowDown), false);
        assert_eq!(selected(&tree), Some(29), "stops at the last row");
        press(&mut tree, Key::Named(NamedKey::Enter), false);
        assert_eq!(*activated.lock().unwrap(), [29]);
    }

    #[test]
    fn tab_and_escape_move_and_clear_focus() {
        let (mut tree, ids) = tree_with(vec![slider(0.0, None), slider(0.0, None)]);
        assert!(press(&mut tree, Key::Named(NamedKey::Tab), false));
        assert_eq!(tree.focused(), Some(ids[0]));
        press(&mut tree, Key::Named(NamedKey::Tab), true);
        assert_eq!(tree.focused(), Some(ids[1]));
        assert!(press(&mut tree, Key::Named(NamedKey::Escape), false));
        assert_eq!(tree.focused(), None);
        assert!(!press(&mut tree, Key::Named(NamedKey::Escape), false));
    }
}
