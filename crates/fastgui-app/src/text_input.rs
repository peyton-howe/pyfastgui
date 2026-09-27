//! `TextInput` editing: keys, mouse, IME and clipboard on top of `fastgui_core::text_edit`,
//! for whichever window's tree the event arrived in. Text is measured through `TextMeasure`
//! (the window's `ChromeRenderer`) so click-to-caret and scrolling match what's drawn.

use fastgui_core::text_edit::{TextEdit, TextMeasure};
use fastgui_core::widget::{Rect, WidgetId, WidgetKind, WidgetTree, TEXT_INPUT_PADDING};
use winit::event::Ime;
use winit::keyboard::{Key, ModifiersState, NamedKey};

/// Room kept right of the caret when scrolling to it, so the caret itself stays visible.
const CARET_SLACK: f32 = 2.0;

/// One key press, as far as widgets care.
pub struct KeyPress<'a> {
    pub key: &'a Key,
    /// The text the press types, if any (winit's `KeyEvent::text`).
    pub text: Option<&'a str>,
    pub modifiers: ModifiersState,
}

impl KeyPress<'_> {
    /// Cmd on macOS, Ctrl elsewhere: select-all, clipboard, undo.
    fn command(&self) -> bool {
        let primary = if cfg!(target_os = "macos") { self.modifiers.super_key() } else { self.modifiers.control_key() };
        // AltGr arrives as Ctrl+Alt on Windows and must still type its character.
        primary && !self.modifiers.alt_key()
    }

    /// Option on macOS, Ctrl elsewhere: move/delete by word.
    fn word(&self) -> bool {
        if cfg!(target_os = "macos") { self.modifiers.alt_key() } else { self.modifiers.control_key() }
    }

    /// Cmd+arrow / Cmd+Backspace on macOS mean "to the start/end of the line".
    fn line(&self) -> bool {
        cfg!(target_os = "macos") && self.modifiers.super_key()
    }
}

pub trait Clipboard {
    fn get(&mut self) -> Option<String>;
    fn set(&mut self, text: String);
}

/// The OS clipboard, opened on first use (so a window that never copies never touches it).
#[derive(Default)]
pub struct SystemClipboard(Option<arboard::Clipboard>);

impl SystemClipboard {
    fn open(&mut self) -> Option<&mut arboard::Clipboard> {
        if self.0.is_none() {
            self.0 = arboard::Clipboard::new().ok();
        }
        self.0.as_mut()
    }
}

impl Clipboard for SystemClipboard {
    fn get(&mut self) -> Option<String> {
        self.open()?.get_text().ok()
    }

    fn set(&mut self, text: String) {
        if let Some(clipboard) = self.open() {
            let _ = clipboard.set_text(text);
        }
    }
}

/// Edit the `TextInput` `id` with `f`, which returns whether its text changed. Fires
/// `on_change` / updates the Python mirror on a change, and scrolls the caret into view.
fn edit(tree: &mut WidgetTree, id: WidgetId, measure: &mut dyn TextMeasure, f: impl FnOnce(&mut TextEdit) -> bool) {
    let mut changed = false;
    tree.mutate_kind(id, |kind| {
        if let WidgetKind::TextInput { edit, .. } = kind {
            changed = f(edit);
        }
    });
    if changed {
        notify_change(tree, id);
    }
    scroll_caret_into_view(tree, id, measure);
}

fn notify_change(tree: &WidgetTree, id: WidgetId) {
    let Some(WidgetKind::TextInput { edit, on_change, mirror, .. }) = tree.kind(id) else { return };
    let text = edit.text().to_owned();
    if let Some(mirror) = mirror {
        mirror.set(text.clone());
    }
    if let Some(callback) = on_change.clone() {
        callback(text);
    }
}

/// Adjust the field's horizontal scroll so the caret is inside it, and never scrolled past
/// the end of the text (deleting from a scrolled line pulls the text back in).
pub fn scroll_caret_into_view(tree: &mut WidgetTree, id: WidgetId, measure: &mut dyn TextMeasure) {
    let Some(rect) = tree.absolute_rect(id) else { return };
    let Some(WidgetKind::TextInput { edit, font_size, scroll, preedit, .. }) = tree.kind(id) else { return };
    let (display, caret, _) = edit.composed(preedit.as_ref());
    let visible = (rect.width - 2.0 * TEXT_INPUT_PADDING - CARET_SLACK).max(0.0);
    let caret_x = measure.caret_x(&display, *font_size, caret);
    let line_width = measure.caret_x(&display, *font_size, display.len());
    let mut target = *scroll;
    if caret_x - target > visible {
        target = caret_x - visible;
    }
    if caret_x < target {
        target = caret_x;
    }
    target = target.clamp(0.0, (line_width - visible).max(0.0));
    if target != *scroll {
        tree.mutate_kind(id, |kind| {
            if let WidgetKind::TextInput { scroll, .. } = kind {
                *scroll = target;
            }
        });
    }
}

/// A key press in the focused `TextInput` `id`. Returns whether it was handled.
pub fn handle_key(
    tree: &mut WidgetTree,
    id: WidgetId,
    press: &KeyPress<'_>,
    measure: &mut dyn TextMeasure,
    clipboard: &mut dyn Clipboard,
) -> bool {
    let shift = press.modifiers.shift_key();
    let (word, line) = (press.word(), press.line());
    match press.key {
        Key::Named(NamedKey::Enter) => {
            if let Some(WidgetKind::TextInput { edit, on_submit: Some(callback), .. }) = tree.kind(id) {
                let (callback, text) = (callback.clone(), edit.text().to_owned());
                callback(text);
            }
        }
        Key::Named(NamedKey::ArrowLeft) if line => move_caret(tree, id, measure, |e| e.home(shift)),
        Key::Named(NamedKey::ArrowRight) if line => move_caret(tree, id, measure, |e| e.end(shift)),
        Key::Named(NamedKey::ArrowLeft) => move_caret(tree, id, measure, |e| e.move_left(shift, word)),
        Key::Named(NamedKey::ArrowRight) => move_caret(tree, id, measure, |e| e.move_right(shift, word)),
        Key::Named(NamedKey::ArrowUp | NamedKey::Home) => move_caret(tree, id, measure, |e| e.home(shift)),
        Key::Named(NamedKey::ArrowDown | NamedKey::End) => move_caret(tree, id, measure, |e| e.end(shift)),
        Key::Named(NamedKey::Backspace) if line => edit(tree, id, measure, |e| {
            if e.selection().is_empty() {
                e.home(true);
            }
            e.backspace(false)
        }),
        Key::Named(NamedKey::Backspace) => edit(tree, id, measure, |e| e.backspace(word)),
        Key::Named(NamedKey::Delete) => edit(tree, id, measure, |e| e.delete(word)),
        Key::Character(c) if press.command() => match c.to_lowercase().as_str() {
            "a" => move_caret(tree, id, measure, |e| e.select_all()),
            "c" => {
                if let Some(WidgetKind::TextInput { edit, .. }) = tree.kind(id) {
                    if !edit.selection().is_empty() {
                        clipboard.set(edit.selected_text().to_owned());
                    }
                }
            }
            "x" => edit(tree, id, measure, |e| match e.cut() {
                Some(text) => {
                    clipboard.set(text);
                    true
                }
                None => false,
            }),
            "v" => {
                let Some(text) = clipboard.get() else { return true };
                edit(tree, id, measure, |e| e.insert(&text));
            }
            "z" if shift => edit(tree, id, measure, TextEdit::redo),
            "z" => edit(tree, id, measure, TextEdit::undo),
            "y" if !cfg!(target_os = "macos") => edit(tree, id, measure, TextEdit::redo),
            _ => return false,
        },
        _ => {
            let typed: String = press.text.unwrap_or_default().chars().filter(|c| !c.is_control()).collect();
            if typed.is_empty() || press.command() {
                return false;
            }
            edit(tree, id, measure, |e| e.insert(&typed));
        }
    }
    true
}

/// A caret/selection change in `id` that leaves the text alone.
fn move_caret(tree: &mut WidgetTree, id: WidgetId, measure: &mut dyn TextMeasure, f: impl FnOnce(&mut TextEdit)) {
    edit(tree, id, measure, |e| {
        f(e);
        false
    });
}

/// Mouse press at window x `cursor_x` in `TextInput` `id`: place the caret there (extending
/// the selection with `extend`), or select the word under it on a double-click.
pub fn handle_press(tree: &mut WidgetTree, id: WidgetId, cursor_x: f32, extend: bool, double: bool, measure: &mut dyn TextMeasure) {
    let Some(index) = index_at_cursor(tree, id, cursor_x, measure) else { return };
    edit(tree, id, measure, |e| {
        if double {
            e.select_word_at(index);
        } else {
            e.move_to(index, extend);
        }
        false
    });
}

/// Mouse drag after a press in `id`: extend the selection to the cursor (scrolling the field
/// when dragged past either edge, since the caret follows the cursor).
pub fn handle_drag(tree: &mut WidgetTree, id: WidgetId, cursor_x: f32, measure: &mut dyn TextMeasure) {
    let Some(index) = index_at_cursor(tree, id, cursor_x, measure) else { return };
    move_caret(tree, id, measure, |e| e.move_to(index, true));
}

fn index_at_cursor(tree: &WidgetTree, id: WidgetId, cursor_x: f32, measure: &mut dyn TextMeasure) -> Option<usize> {
    let rect = tree.absolute_rect(id)?;
    let Some(WidgetKind::TextInput { edit, font_size, scroll, .. }) = tree.kind(id) else { return None };
    let x = cursor_x - rect.x - TEXT_INPUT_PADDING + scroll;
    Some(measure.index_at(edit.text(), *font_size, x))
}

/// An IME event for the focused `TextInput` `id`.
pub fn handle_ime(tree: &mut WidgetTree, id: WidgetId, ime: &Ime, measure: &mut dyn TextMeasure) {
    let set_preedit = |tree: &mut WidgetTree, preedit: Option<(String, Option<(usize, usize)>)>| {
        tree.mutate_kind(id, |kind| {
            if let WidgetKind::TextInput { preedit: current, .. } = kind {
                *current = preedit;
            }
        });
    };
    match ime {
        Ime::Preedit(text, cursor) => {
            set_preedit(tree, (!text.is_empty()).then(|| (text.clone(), *cursor)));
            scroll_caret_into_view(tree, id, measure);
        }
        Ime::Commit(text) => {
            set_preedit(tree, None);
            edit(tree, id, measure, |e| e.insert(text));
        }
        Ime::Disabled => set_preedit(tree, None),
        Ime::Enabled => {}
    }
}

/// Where the focused `TextInput`'s caret is (window coordinates, layout units), for placing
/// the IME candidate window next to it. `None` when no text input has focus.
pub fn ime_cursor_area(tree: &WidgetTree, measure: &mut dyn TextMeasure) -> Option<Rect> {
    let id = tree.focused()?;
    let rect = tree.absolute_rect(id)?;
    let Some(WidgetKind::TextInput { edit, font_size, scroll, preedit, .. }) = tree.kind(id) else { return None };
    let (display, caret, _) = edit.composed(preedit.as_ref());
    let x = rect.x + TEXT_INPUT_PADDING + measure.caret_x(&display, *font_size, caret) - scroll;
    Some(Rect { x, y: rect.y, width: 1.0, height: rect.height })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use fastgui_core::taffy::prelude::*;
    use fastgui_core::widget::Color;
    use fastgui_core::Readback;

    use super::*;

    /// Every grapheme is 10 units wide.
    struct Mono;

    impl TextMeasure for Mono {
        fn caret_x(&mut self, text: &str, _: f32, index: usize) -> f32 {
            text[..index].chars().count() as f32 * 10.0
        }

        fn index_at(&mut self, text: &str, _: f32, x: f32) -> usize {
            let n = ((x / 10.0).round().max(0.0) as usize).min(text.chars().count());
            text.char_indices().nth(n).map_or(text.len(), |(i, _)| i)
        }
    }

    #[derive(Default)]
    struct FakeClipboard(Option<String>);

    impl Clipboard for FakeClipboard {
        fn get(&mut self) -> Option<String> {
            self.0.clone()
        }

        fn set(&mut self, text: String) {
            self.0 = Some(text);
        }
    }

    struct Field {
        tree: WidgetTree,
        id: WidgetId,
        changes: Arc<Mutex<Vec<String>>>,
        submits: Arc<Mutex<Vec<String>>>,
        mirror: Readback<String>,
        clipboard: FakeClipboard,
    }

    /// A 116-wide field at x=0: 100 units (10 characters) visible after padding.
    fn field(text: &str) -> Field {
        let changes = Arc::new(Mutex::new(Vec::new()));
        let submits = Arc::new(Mutex::new(Vec::new()));
        let mirror = Readback::new(text.to_owned());
        let (c, s) = (changes.clone(), submits.clone());
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            Style {
                size: Size { width: Dimension::length(100.0 + 2.0 * TEXT_INPUT_PADDING + CARET_SLACK), height: Dimension::length(30.0) },
                ..Default::default()
            },
            WidgetKind::TextInput {
                edit: TextEdit::new(text),
                placeholder: String::new(),
                font_size: 16.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                scroll: 0.0,
                preedit: None,
                on_change: Some(Arc::new(move |t| c.lock().unwrap().push(t))),
                on_submit: Some(Arc::new(move |t| s.lock().unwrap().push(t))),
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 100.0);
        tree.set_focus(Some(id));
        Field { tree, id, changes, submits, mirror, clipboard: FakeClipboard::default() }
    }

    impl Field {
        fn key(&mut self, key: Key, text: Option<&str>, modifiers: ModifiersState) -> bool {
            let press = KeyPress { key: &key, text, modifiers };
            handle_key(&mut self.tree, self.id, &press, &mut Mono, &mut self.clipboard)
        }

        fn typed(&mut self, text: &str) {
            for c in text.chars() {
                let s = c.to_string();
                self.key(Key::Character(s.as_str().into()), Some(&s), ModifiersState::empty());
            }
        }

        fn state(&self) -> (String, usize, f32) {
            let Some(WidgetKind::TextInput { edit, scroll, .. }) = self.tree.kind(self.id) else { panic!() };
            (edit.text().to_owned(), edit.caret(), *scroll)
        }
    }

    const CMD: ModifiersState = if cfg!(target_os = "macos") { ModifiersState::SUPER } else { ModifiersState::CONTROL };

    #[test]
    fn typing_fires_on_change_and_updates_mirror() {
        let mut f = field("");
        f.typed("hi");
        assert_eq!(f.state().0, "hi");
        assert_eq!(*f.changes.lock().unwrap(), ["h", "hi"]);
        assert_eq!(f.mirror.get(), "hi");
        f.key(Key::Named(NamedKey::Enter), Some("\r"), ModifiersState::empty());
        assert_eq!(*f.submits.lock().unwrap(), ["hi"]);
        assert_eq!(f.state().0, "hi", "Enter doesn't type a newline");
        // Control characters and command-modified keys never type.
        assert!(!f.key(Key::Character("q".into()), Some("q"), CMD));
        assert_eq!(f.state().0, "hi");
    }

    #[test]
    fn clipboard_and_undo_shortcuts() {
        let mut f = field("hello");
        f.key(Key::Character("a".into()), Some("a"), CMD);
        f.key(Key::Character("x".into()), Some("x"), CMD);
        assert_eq!((f.state().0.as_str(), f.clipboard.0.as_deref()), ("", Some("hello")));
        f.key(Key::Character("v".into()), Some("v"), CMD);
        f.key(Key::Character("v".into()), Some("v"), CMD);
        assert_eq!(f.state().0, "hellohello");
        f.key(Key::Character("z".into()), Some("z"), CMD);
        assert_eq!(f.state().0, "hello");
        f.key(Key::Character("Z".into()), Some("Z"), CMD | ModifiersState::SHIFT);
        assert_eq!(f.state().0, "hellohello");
        assert_eq!(f.mirror.get(), "hellohello", "undo/redo also reach the mirror");
    }

    #[test]
    fn caret_scrolls_into_view_and_back() {
        let mut f = field("");
        f.typed("0123456789abcde");
        // 15 chars = 150 units in a 100-unit view: the caret sits at the right edge.
        assert_eq!(f.state(), ("0123456789abcde".into(), 15, 50.0));
        f.key(Key::Named(NamedKey::Home), None, ModifiersState::empty());
        assert_eq!(f.state().2, 0.0);
        f.key(Key::Named(NamedKey::End), None, ModifiersState::empty());
        for _ in 0..10 {
            f.key(Key::Named(NamedKey::Backspace), None, ModifiersState::empty());
        }
        assert_eq!(f.state(), ("01234".into(), 5, 0.0), "deleting pulls a scrolled line back");
    }

    #[test]
    fn click_places_caret_accounting_for_padding_and_scroll() {
        let mut f = field("0123456789abcde");
        f.key(Key::Named(NamedKey::End), None, ModifiersState::empty());
        let scroll = f.state().2;
        handle_press(&mut f.tree, f.id, TEXT_INPUT_PADDING + 31.0, false, false, &mut Mono);
        assert_eq!(f.state().1, ((31.0 + scroll) / 10.0).round() as usize);
        handle_drag(&mut f.tree, f.id, 500.0, &mut Mono);
        let Some(WidgetKind::TextInput { edit, .. }) = f.tree.kind(f.id) else { panic!() };
        assert_eq!(edit.selection().end, 15, "dragging past the edge selects to the end");
    }

    #[test]
    fn ime_preedit_then_commit() {
        let mut f = field("ab");
        handle_ime(&mut f.tree, f.id, &Ime::Preedit("にほ".into(), Some((6, 6))), &mut Mono);
        assert_eq!(f.state().0, "ab", "composition isn't part of the text");
        let area = ime_cursor_area(&f.tree, &mut Mono).unwrap();
        assert_eq!(area.x, TEXT_INPUT_PADDING + 40.0, "caret after the composed text");
        handle_ime(&mut f.tree, f.id, &Ime::Commit("日本".into()), &mut Mono);
        assert_eq!(f.state().0, "ab日本");
        let Some(WidgetKind::TextInput { preedit, .. }) = f.tree.kind(f.id) else { panic!() };
        assert!(preedit.is_none());
        assert_eq!(*f.changes.lock().unwrap(), ["ab日本"]);
    }
}
