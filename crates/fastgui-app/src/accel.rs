//! Keyboard accelerator parsing helpers and firing against a `WidgetTree` table.

use fastgui_core::widget::{Accel, AccelKey, AccelNamed, WidgetKind, WidgetTree};
use winit::keyboard::{Key, ModifiersState, NamedKey};

use crate::text_input::KeyPress;

/// Cmd on macOS, Ctrl elsewhere.
pub fn primary_down(modifiers: ModifiersState) -> bool {
    if cfg!(target_os = "macos") {
        modifiers.super_key()
    } else {
        modifiers.control_key()
    }
}

/// Whether `press` matches `accel` (platform primary for both Ctrl and Cmd in the string).
pub fn matches(accel: &Accel, key: &Key, modifiers: ModifiersState) -> bool {
    if accel.primary != primary_down(modifiers) {
        return false;
    }
    if accel.shift != modifiers.shift_key() {
        return false;
    }
    if accel.alt != modifiers.alt_key() {
        return false;
    }
    // When primary is required, ignore the non-primary of Ctrl/Super so Ctrl+S on macOS
    // (without Cmd) does not match a primary shortcut.
    if accel.primary {
        let other = if cfg!(target_os = "macos") {
            modifiers.control_key()
        } else {
            modifiers.super_key()
        };
        if other {
            return false;
        }
    }
    match (&accel.key, key) {
        (AccelKey::Char(want), Key::Character(got)) => {
            let mut chars = got.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => c.to_ascii_lowercase() == *want,
                _ => false,
            }
        }
        (AccelKey::F(n), Key::Named(named)) => f_key(*n) == Some(*named),
        (AccelKey::Named(want), Key::Named(named)) => named_key(*want) == *named,
        (AccelKey::Named(AccelNamed::Space), Key::Character(got)) => got.as_str() == " ",
        _ => false,
    }
}

fn named_key(key: AccelNamed) -> NamedKey {
    match key {
        AccelNamed::Delete => NamedKey::Delete,
        AccelNamed::Backspace => NamedKey::Backspace,
        AccelNamed::Enter => NamedKey::Enter,
        AccelNamed::Tab => NamedKey::Tab,
        AccelNamed::Space => NamedKey::Space,
        AccelNamed::Insert => NamedKey::Insert,
        AccelNamed::Home => NamedKey::Home,
        AccelNamed::End => NamedKey::End,
        AccelNamed::PageUp => NamedKey::PageUp,
        AccelNamed::PageDown => NamedKey::PageDown,
        AccelNamed::Up => NamedKey::ArrowUp,
        AccelNamed::Down => NamedKey::ArrowDown,
        AccelNamed::Left => NamedKey::ArrowLeft,
        AccelNamed::Right => NamedKey::ArrowRight,
    }
}

fn f_key(n: u8) -> Option<NamedKey> {
    Some(match n {
        1 => NamedKey::F1,
        2 => NamedKey::F2,
        3 => NamedKey::F3,
        4 => NamedKey::F4,
        5 => NamedKey::F5,
        6 => NamedKey::F6,
        7 => NamedKey::F7,
        8 => NamedKey::F8,
        9 => NamedKey::F9,
        10 => NamedKey::F10,
        11 => NamedKey::F11,
        12 => NamedKey::F12,
        _ => return None,
    })
}

/// Standard text-edit chords that stay with a focused `TextInput` / `TextArea`.
fn is_text_edit_chord(accel: &Accel) -> bool {
    accel.primary
        && !accel.alt
        && matches!(accel.key, AccelKey::Char(c) if matches!(c, 'a' | 'c' | 'v' | 'x' | 'z'))
}

/// Fire the first matching accelerator. When a text field is focused, still allow primary
/// shortcuts (e.g. Cmd+S) but leave unmodified typing and A/C/V/X/Z edit chords alone.
pub fn try_fire(tree: &WidgetTree, press: &KeyPress<'_>) -> bool {
    let text_focused = tree.focused().is_some_and(|id| {
        matches!(tree.kind(id), Some(WidgetKind::TextInput { .. } | WidgetKind::TextArea { .. }))
    });
    for (accel, callback) in tree.accelerators() {
        if !matches(accel, press.plain.unwrap_or(press.key), press.modifiers) {
            continue;
        }
        if text_focused && is_text_edit_chord(accel) {
            continue;
        }
        // Unmodified / non-primary keys while typing stay with the field — except F1–F12,
        // which never type text (F5 Refresh should still work from a focused field).
        if text_focused && !accel.primary && !accel.alt && !matches!(accel.key, AccelKey::F(_)) {
            continue;
        }
        callback();
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use fastgui_core::widget::{Color, WidgetKind};
    use winit::keyboard::Key;

    fn press_char(c: &str, primary: bool, shift: bool) -> (Key, ModifiersState) {
        let mut mods = ModifiersState::empty();
        if primary {
            if cfg!(target_os = "macos") {
                mods |= ModifiersState::SUPER;
            } else {
                mods |= ModifiersState::CONTROL;
            }
        }
        if shift {
            mods |= ModifiersState::SHIFT;
        }
        (Key::Character(c.into()), mods)
    }

    #[test]
    fn matches_primary_and_shift() {
        let accel = Accel::parse("Ctrl+S").unwrap();
        let (key, mods) = press_char("s", true, false);
        assert!(matches(&accel, &key, mods));
        let (key, mods) = press_char("s", false, false);
        assert!(!matches(&accel, &key, mods));

        let accel = Accel::parse("Shift+Cmd+N").unwrap();
        let (key, mods) = press_char("n", true, true);
        assert!(matches(&accel, &key, mods));
        let (key, mods) = press_char("n", true, false);
        assert!(!matches(&accel, &key, mods));
    }

    #[test]
    fn shortcuts_match_the_unmodified_key() {
        // Shift+1 types "!" (and macOS Option+letter a special character): the shortcut must
        // match on the key itself, which winit reports as `key_without_modifiers`.
        let fired = Arc::new(AtomicUsize::new(0));
        let counter = fired.clone();
        let mut tree = WidgetTree::new();
        tree.register_accelerator(
            Accel::parse("Ctrl+Shift+1").unwrap(),
            Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let (typed, mods) = press_char("!", true, true);
        let plain = Key::Character("1".into());
        assert!(!try_fire(&tree, &KeyPress { key: &typed, plain: None, text: None, modifiers: mods }), "the typed \"!\" alone can't match");
        assert!(try_fire(&tree, &KeyPress { key: &typed, plain: Some(&plain), text: None, modifiers: mods }));
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn try_fire_skips_text_edit_chords_when_typing() {
        let fired = Arc::new(AtomicUsize::new(0));
        let counter = fired.clone();
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let input = tree.new_node(
            Default::default(),
            WidgetKind::TextInput {
                edit: fastgui_core::text_edit::TextEdit::new(""),
                placeholder: String::new(),
                font_size: 12.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                scroll: 0.0,
                preedit: None,
                on_change: None,
                on_submit: None,
                mirror: None,
            },
        );
        tree.add_child(root, input);
        tree.set_focus(Some(input));
        tree.register_accelerator(
            Accel::parse("Ctrl+S").unwrap(),
            Arc::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );
        tree.register_accelerator(
            Accel::parse("Ctrl+C").unwrap(),
            Arc::new(|| panic!("Ctrl+C must stay with the text field")),
        );

        let (key, mods) = press_char("c", true, false);
        let press = KeyPress { key: &key, plain: None, text: None, modifiers: mods };
        assert!(!try_fire(&tree, &press));

        let (key, mods) = press_char("s", true, false);
        let press = KeyPress { key: &key, plain: None, text: None, modifiers: mods };
        assert!(try_fire(&tree, &press));
        assert_eq!(fired.load(Ordering::SeqCst), 1);
    }
}
