//! `TextInput` / `TextArea` editing: keys, mouse, IME and clipboard on top of
//! `fastgui_core::text_edit`, for whichever window's tree the event arrived in. Text is measured
//! through `TextMeasure` (the window's `ChromeRenderer`) so click-to-caret and scrolling match
//! what's drawn.

use fastgui_core::text_edit::{TextEdit, TextMeasure};
use fastgui_core::widget::{
    Rect, WidgetId, WidgetKind, WidgetTree, LINE_HEIGHT_RATIO, TEXT_INPUT_PADDING,
    TEXT_INPUT_VERTICAL_PADDING,
};
use winit::event::Ime;
use winit::keyboard::{Key, ModifiersState, NamedKey};

/// Room kept right of the caret when scrolling to it, so the caret itself stays visible.
const CARET_SLACK: f32 = 2.0;

/// One key press, as far as widgets care.
pub struct KeyPress<'a> {
    pub key: &'a Key,
    /// The same key with no modifiers applied (`Shift+1` is `1` here, `!` in `key`; macOS
    /// `Option+letter` is the letter). Shortcuts match against it. `None`: same as `key`.
    pub plain: Option<&'a Key>,
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

fn is_text_field(kind: &WidgetKind) -> bool {
    matches!(kind, WidgetKind::TextInput { .. } | WidgetKind::TextArea { .. })
}

/// Edit the text field `id` with `f`, which returns whether its text changed. Fires
/// `on_change` / updates the Python mirror on a change, and scrolls the caret into view.
fn edit(tree: &mut WidgetTree, id: WidgetId, measure: &mut dyn TextMeasure, f: impl FnOnce(&mut TextEdit) -> bool) {
    let mut changed = false;
    tree.mutate_kind(id, |kind| match kind {
        WidgetKind::TextInput { edit, .. } | WidgetKind::TextArea { edit, .. } => {
            changed = f(edit);
        }
        _ => {}
    });
    if changed {
        notify_change(tree, id);
    }
    scroll_caret_into_view(tree, id, measure);
}

fn notify_change(tree: &WidgetTree, id: WidgetId) {
    let (text, on_change, mirror) = match tree.kind(id) {
        Some(WidgetKind::TextInput { edit, on_change, mirror, .. } | WidgetKind::TextArea { edit, on_change, mirror, .. }) => {
            (edit.text().to_owned(), on_change.clone(), mirror.clone())
        }
        _ => return,
    };
    if let Some(mirror) = mirror {
        mirror.set(text.clone());
    }
    if let Some(callback) = on_change {
        callback(text);
    }
}

/// Adjust the field's scroll so the caret is inside it.
pub fn scroll_caret_into_view(tree: &mut WidgetTree, id: WidgetId, measure: &mut dyn TextMeasure) {
    let Some(rect) = tree.absolute_rect(id) else { return };
    match tree.kind(id) {
        Some(WidgetKind::TextInput { edit, font_size, scroll, preedit, .. }) => {
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
        Some(WidgetKind::TextArea { edit, font_size, scroll_x, scroll_y, preedit, .. }) => {
            let (display, caret, _) = edit.composed(preedit.as_ref());
            let size = *font_size;
            let line_height = size * LINE_HEIGHT_RATIO;
            let visible_h = (rect.height - 2.0 * TEXT_INPUT_VERTICAL_PADDING).max(0.0);
            let visible_w = (rect.width - 2.0 * TEXT_INPUT_PADDING - CARET_SLACK).max(0.0);
            let line_index = display[..caret].bytes().filter(|&b| b == b'\n').count() as f32;
            let caret_top = line_index * line_height;
            let caret_bottom = caret_top + line_height;
            let mut target_y = *scroll_y;
            if caret_bottom - target_y > visible_h {
                target_y = caret_bottom - visible_h;
            }
            if caret_top < target_y {
                target_y = caret_top;
            }
            let total_lines = display.split('\n').count().max(1) as f32;
            let content_h = total_lines * line_height;
            target_y = target_y.clamp(0.0, (content_h - visible_h).max(0.0));

            let line_start = display[..caret].rfind('\n').map_or(0, |i| i + 1);
            let line_end = display[line_start..].find('\n').map_or(display.len(), |i| line_start + i);
            let line = &display[line_start..line_end];
            let local = caret.saturating_sub(line_start).min(line.len());
            let caret_x = measure.caret_x(line, size, local);
            let mut max_line_w = 0.0f32;
            for row in display.split('\n') {
                max_line_w = max_line_w.max(measure.caret_x(row, size, row.len()));
            }
            let mut target_x = *scroll_x;
            if caret_x - target_x > visible_w {
                target_x = caret_x - visible_w;
            }
            if caret_x < target_x {
                target_x = caret_x;
            }
            target_x = target_x.clamp(0.0, (max_line_w - visible_w).max(0.0));

            if target_x != *scroll_x || target_y != *scroll_y {
                tree.mutate_kind(id, |kind| {
                    if let WidgetKind::TextArea { scroll_x, scroll_y, .. } = kind {
                        *scroll_x = target_x;
                        *scroll_y = target_y;
                    }
                });
            }
        }
        _ => {}
    }
}

/// A key press in the focused `TextInput` / `TextArea` `id`. Returns whether it was handled.
pub fn handle_key(
    tree: &mut WidgetTree,
    id: WidgetId,
    press: &KeyPress<'_>,
    measure: &mut dyn TextMeasure,
    clipboard: &mut dyn Clipboard,
) -> bool {
    let multiline = matches!(tree.kind(id), Some(WidgetKind::TextArea { .. }));
    if !matches!(tree.kind(id), Some(k) if is_text_field(k)) {
        return false;
    }
    let shift = press.modifiers.shift_key();
    let (word, line) = (press.word(), press.line());
    match press.key {
        Key::Named(NamedKey::Enter) if multiline && press.command() => {
            if let Some(WidgetKind::TextArea { edit, on_submit: Some(callback), .. }) = tree.kind(id) {
                let (callback, text) = (callback.clone(), edit.text().to_owned());
                callback(text);
            }
        }
        Key::Named(NamedKey::Enter) if multiline => {
            edit(tree, id, measure, |e| e.insert("\n"));
        }
        Key::Named(NamedKey::Enter) => {
            if let Some(WidgetKind::TextInput { edit, on_submit: Some(callback), .. }) = tree.kind(id) {
                let (callback, text) = (callback.clone(), edit.text().to_owned());
                callback(text);
            }
        }
        Key::Named(NamedKey::ArrowLeft) if line && multiline => {
            move_caret(tree, id, measure, |e| e.line_home(shift));
        }
        Key::Named(NamedKey::ArrowRight) if line && multiline => {
            move_caret(tree, id, measure, |e| e.line_end_move(shift));
        }
        Key::Named(NamedKey::ArrowLeft) if line => move_caret(tree, id, measure, |e| e.home(shift)),
        Key::Named(NamedKey::ArrowRight) if line => move_caret(tree, id, measure, |e| e.end(shift)),
        Key::Named(NamedKey::ArrowLeft) => move_caret(tree, id, measure, |e| e.move_left(shift, word)),
        Key::Named(NamedKey::ArrowRight) => move_caret(tree, id, measure, |e| e.move_right(shift, word)),
        Key::Named(NamedKey::ArrowUp) if multiline => {
            move_caret(tree, id, measure, |e| e.move_line_up(shift));
        }
        Key::Named(NamedKey::ArrowDown) if multiline => {
            move_caret(tree, id, measure, |e| e.move_line_down(shift));
        }
        Key::Named(NamedKey::Home) if multiline && (press.command() || line) => {
            move_caret(tree, id, measure, |e| e.home(shift));
        }
        Key::Named(NamedKey::End) if multiline && (press.command() || line) => {
            move_caret(tree, id, measure, |e| e.end(shift));
        }
        Key::Named(NamedKey::Home) if multiline => {
            move_caret(tree, id, measure, |e| e.line_home(shift));
        }
        Key::Named(NamedKey::End) if multiline => {
            move_caret(tree, id, measure, |e| e.line_end_move(shift));
        }
        Key::Named(NamedKey::ArrowUp | NamedKey::Home) => move_caret(tree, id, measure, |e| e.home(shift)),
        Key::Named(NamedKey::ArrowDown | NamedKey::End) => move_caret(tree, id, measure, |e| e.end(shift)),
        Key::Named(NamedKey::Backspace) if line => edit(tree, id, measure, |e| {
            if e.selection().is_empty() {
                if multiline {
                    e.line_home(true);
                } else {
                    e.home(true);
                }
            }
            e.backspace(false)
        }),
        Key::Named(NamedKey::Backspace) => edit(tree, id, measure, |e| e.backspace(word)),
        Key::Named(NamedKey::Delete) => edit(tree, id, measure, |e| e.delete(word)),
        Key::Character(c) if press.command() => match c.to_lowercase().as_str() {
            "a" => move_caret(tree, id, measure, |e| e.select_all()),
            "c" => {
                if let Some(WidgetKind::TextInput { edit, .. } | WidgetKind::TextArea { edit, .. }) = tree.kind(id) {
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

/// Mouse press at window coords in text field `id`: place the caret (extending with `extend`),
/// or select the word under it on a double-click.
pub fn handle_press(
    tree: &mut WidgetTree,
    id: WidgetId,
    cursor_x: f32,
    cursor_y: f32,
    extend: bool,
    double: bool,
    measure: &mut dyn TextMeasure,
) {
    let Some(index) = index_at_cursor(tree, id, cursor_x, cursor_y, measure) else { return };
    edit(tree, id, measure, |e| {
        if double {
            e.select_word_at(index);
        } else {
            e.move_to(index, extend);
        }
        false
    });
}

/// Mouse drag after a press in `id`: extend the selection to the cursor.
pub fn handle_drag(tree: &mut WidgetTree, id: WidgetId, cursor_x: f32, cursor_y: f32, measure: &mut dyn TextMeasure) {
    let Some(index) = index_at_cursor(tree, id, cursor_x, cursor_y, measure) else { return };
    move_caret(tree, id, measure, |e| e.move_to(index, true));
}

fn index_at_cursor(
    tree: &WidgetTree,
    id: WidgetId,
    cursor_x: f32,
    cursor_y: f32,
    measure: &mut dyn TextMeasure,
) -> Option<usize> {
    let rect = tree.absolute_rect(id)?;
    match tree.kind(id) {
        Some(WidgetKind::TextInput { edit, font_size, scroll, .. }) => {
            let x = cursor_x - rect.x - TEXT_INPUT_PADDING + scroll;
            Some(measure.index_at(edit.text(), *font_size, x))
        }
        Some(WidgetKind::TextArea { edit, font_size, scroll_x, scroll_y, .. }) => {
            let text = edit.text();
            let line_height = *font_size * LINE_HEIGHT_RATIO;
            let y = cursor_y - rect.y - TEXT_INPUT_VERTICAL_PADDING + *scroll_y;
            let line_index = if line_height <= 0.0 {
                0
            } else {
                (y / line_height).floor().max(0.0) as usize
            };
            let mut start = 0usize;
            let mut lines = text.split('\n').peekable();
            let mut i = 0usize;
            while let Some(line) = lines.next() {
                let end = start + line.len();
                let last = lines.peek().is_none();
                if i == line_index || last {
                    let x = cursor_x - rect.x - TEXT_INPUT_PADDING + scroll_x;
                    let local = measure.index_at(line, *font_size, x);
                    return Some(start + local);
                }
                start = end + 1; // skip '\n'
                i += 1;
            }
            Some(text.len())
        }
        _ => None,
    }
}

/// An IME event for the focused text field `id`.
pub fn handle_ime(tree: &mut WidgetTree, id: WidgetId, ime: &Ime, measure: &mut dyn TextMeasure) {
    let set_preedit = |tree: &mut WidgetTree, preedit: Option<(String, Option<(usize, usize)>)>| {
        tree.mutate_kind(id, |kind| match kind {
            WidgetKind::TextInput { preedit: current, .. } | WidgetKind::TextArea { preedit: current, .. } => {
                *current = preedit;
            }
            _ => {}
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

/// Where the focused text field's caret is (window coordinates, layout units), for placing
/// the IME candidate window next to it. `None` when no text field has focus.
pub fn ime_cursor_area(tree: &WidgetTree, measure: &mut dyn TextMeasure) -> Option<Rect> {
    let id = tree.focused()?;
    let rect = tree.absolute_rect(id)?;
    match tree.kind(id) {
        Some(WidgetKind::TextInput { edit, font_size, scroll, preedit, .. }) => {
            let (display, caret, _) = edit.composed(preedit.as_ref());
            let x = rect.x + TEXT_INPUT_PADDING + measure.caret_x(&display, *font_size, caret) - scroll;
            Some(Rect { x, y: rect.y, width: 1.0, height: rect.height })
        }
        Some(WidgetKind::TextArea { edit, font_size, scroll_x, scroll_y, preedit, .. }) => {
            let (display, caret, _) = edit.composed(preedit.as_ref());
            let caret = caret.min(display.len());
            let line_height = *font_size * LINE_HEIGHT_RATIO;
            let line_start = display[..caret].rfind('\n').map_or(0, |i| i + 1);
            let line_end = display[line_start..].find('\n').map_or(display.len(), |i| line_start + i);
            let line = &display[line_start..line_end];
            let local_caret = caret.saturating_sub(line_start).min(line.len());
            let line_index = display[..line_start].bytes().filter(|&b| b == b'\n').count() as f32;
            let x = rect.x + TEXT_INPUT_PADDING + measure.caret_x(line, *font_size, local_caret) - scroll_x;
            let y = rect.y + TEXT_INPUT_VERTICAL_PADDING + line_index * line_height - *scroll_y;
            let _ = edit;
            Some(Rect { x, y, width: 1.0, height: line_height })
        }
        _ => None,
    }
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
                size: Size {
                    width: Dimension::length(100.0 + 2.0 * TEXT_INPUT_PADDING + CARET_SLACK),
                    height: Dimension::length(30.0),
                },
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

    fn area(text: &str) -> Field {
        let changes = Arc::new(Mutex::new(Vec::new()));
        let submits = Arc::new(Mutex::new(Vec::new()));
        let mirror = Readback::new(text.to_owned());
        let (c, s) = (changes.clone(), submits.clone());
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            Style {
                size: Size {
                    width: Dimension::length(200.0),
                    height: Dimension::length(16.0 * LINE_HEIGHT_RATIO * 4.0 + 2.0 * TEXT_INPUT_VERTICAL_PADDING),
                },
                ..Default::default()
            },
            WidgetKind::TextArea {
                edit: TextEdit::new_multiline(text),
                placeholder: String::new(),
                font_size: 16.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                scroll_x: 0.0,
                scroll_y: 0.0,
                preedit: None,
                on_change: Some(Arc::new(move |t| c.lock().unwrap().push(t))),
                on_submit: Some(Arc::new(move |t| s.lock().unwrap().push(t))),
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 300.0);
        tree.set_focus(Some(id));
        Field { tree, id, changes, submits, mirror, clipboard: FakeClipboard::default() }
    }

    /// A narrow multiline field (~10 mono characters visible) for horizontal scroll tests.
    fn narrow_area(text: &str) -> Field {
        let changes = Arc::new(Mutex::new(Vec::new()));
        let submits = Arc::new(Mutex::new(Vec::new()));
        let mirror = Readback::new(text.to_owned());
        let (c, s) = (changes.clone(), submits.clone());
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            Style {
                size: Size {
                    width: Dimension::length(100.0 + 2.0 * TEXT_INPUT_PADDING + CARET_SLACK),
                    height: Dimension::length(16.0 * LINE_HEIGHT_RATIO * 2.0 + 2.0 * TEXT_INPUT_VERTICAL_PADDING),
                },
                ..Default::default()
            },
            WidgetKind::TextArea {
                edit: TextEdit::new_multiline(text),
                placeholder: String::new(),
                font_size: 16.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                scroll_x: 0.0,
                scroll_y: 0.0,
                preedit: None,
                on_change: Some(Arc::new(move |t| c.lock().unwrap().push(t))),
                on_submit: Some(Arc::new(move |t| s.lock().unwrap().push(t))),
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 300.0);
        tree.set_focus(Some(id));
        Field { tree, id, changes, submits, mirror, clipboard: FakeClipboard::default() }
    }

    impl Field {
        fn key(&mut self, key: Key, text: Option<&str>, modifiers: ModifiersState) -> bool {
            let press = KeyPress { key: &key, plain: None, text, modifiers };
            handle_key(&mut self.tree, self.id, &press, &mut Mono, &mut self.clipboard)
        }

        fn typed(&mut self, text: &str) {
            for c in text.chars() {
                let s = c.to_string();
                self.key(Key::Character(s.as_str().into()), Some(&s), ModifiersState::empty());
            }
        }

        fn state(&self) -> (String, usize, f32) {
            match self.tree.kind(self.id) {
                Some(WidgetKind::TextInput { edit, scroll, .. }) => (edit.text().to_owned(), edit.caret(), *scroll),
                Some(WidgetKind::TextArea { edit, scroll_y, .. }) => (edit.text().to_owned(), edit.caret(), *scroll_y),
                _ => panic!(),
            }
        }

        fn scroll_x(&self) -> f32 {
            match self.tree.kind(self.id) {
                Some(WidgetKind::TextArea { scroll_x, .. }) => *scroll_x,
                _ => panic!(),
            }
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
        handle_press(&mut f.tree, f.id, TEXT_INPUT_PADDING + 31.0, 15.0, false, false, &mut Mono);
        assert_eq!(f.state().1, ((31.0 + scroll) / 10.0).round() as usize);
        handle_drag(&mut f.tree, f.id, 500.0, 15.0, &mut Mono);
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

    #[test]
    fn textarea_enter_inserts_newline_and_cmd_enter_submits() {
        let mut f = area("hi");
        f.key(Key::Named(NamedKey::Enter), Some("\r"), ModifiersState::empty());
        assert_eq!(f.state().0, "hi\n");
        assert!(f.submits.lock().unwrap().is_empty(), "plain Enter does not submit");
        f.typed("there");
        assert_eq!(f.state().0, "hi\nthere");
        f.key(Key::Named(NamedKey::Enter), Some("\r"), CMD);
        assert_eq!(*f.submits.lock().unwrap(), ["hi\nthere"]);
        assert_eq!(f.state().0, "hi\nthere", "Cmd/Ctrl+Enter does not insert another newline");
    }

    #[test]
    fn textarea_arrows_move_by_line() {
        let mut f = area("ab\ncde\nf");
        f.key(Key::Named(NamedKey::Home), None, ModifiersState::empty());
        // End of document first, then Home = line start of last line ("f").
        assert_eq!(f.state().1, 7);
        f.key(Key::Named(NamedKey::ArrowUp), None, ModifiersState::empty());
        assert_eq!(f.state().1, 3);
        f.key(Key::Named(NamedKey::End), None, ModifiersState::empty());
        assert_eq!(f.state().1, 6);
        f.key(Key::Named(NamedKey::Home), None, CMD);
        assert_eq!(f.state().1, 0);
    }

    #[test]
    fn textarea_caret_scrolls_horizontally_into_view_and_home_resets() {
        let mut f = narrow_area("");
        f.typed("0123456789abcde");
        // 15 chars = 150 units in a 100-unit view: the caret sits at the right edge.
        assert_eq!(f.scroll_x(), 50.0);
        f.key(Key::Named(NamedKey::Home), None, ModifiersState::empty());
        assert_eq!(f.scroll_x(), 0.0);
    }
}
