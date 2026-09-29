//! Editable single-line text: the string, caret and selection, and undo history, with no
//! knowledge of fonts. Positions are byte offsets that always sit on grapheme-cluster
//! boundaries, so an emoji or an accented letter moves and deletes as one unit.
//! `fastgui-app` drives it from keys and clicks; `fastgui-chrome` draws it.

use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

/// Undo entries kept per field; the oldest is dropped past this.
const UNDO_LIMIT: usize = 200;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot {
    text: String,
    caret: usize,
    anchor: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EditKind {
    Typing,
    Deleting,
}

#[derive(Clone, Debug, Default)]
pub struct TextEdit {
    text: String,
    /// Where the caret is. The selection runs between `anchor` and `caret` (either order);
    /// they're equal when nothing is selected.
    caret: usize,
    anchor: usize,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    /// The kind of the last edit, while consecutive edits of that kind still merge into one
    /// undo step (typing a word undoes as a whole). Any caret move ends the run.
    run: Option<EditKind>,
}

impl TextEdit {
    /// A field holding `text`, caret at the end.
    pub fn new(text: &str) -> Self {
        let text = single_line(text);
        let end = text.len();
        Self { text, caret: end, anchor: end, ..Default::default() }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn caret(&self) -> usize {
        self.caret
    }

    pub fn anchor(&self) -> usize {
        self.anchor
    }

    /// The selected byte range (start <= end); empty when nothing is selected.
    pub fn selection(&self) -> Range<usize> {
        self.caret.min(self.anchor)..self.caret.max(self.anchor)
    }

    pub fn selected_text(&self) -> &str {
        &self.text[self.selection()]
    }

    /// Replace the contents from code (not the user): caret to the end, history cleared.
    pub fn set_text(&mut self, text: &str) {
        *self = Self::new(text);
    }

    /// Put the caret at `pos` (snapped to a grapheme boundary). With `extend`, the anchor
    /// stays put and the selection grows or shrinks; otherwise the selection collapses.
    pub fn move_to(&mut self, pos: usize, extend: bool) {
        self.caret = self.snap(pos);
        if !extend {
            self.anchor = self.caret;
        }
        self.run = None;
    }

    /// Left/right by one grapheme, or by word with `word`. Without `extend`, a selection
    /// collapses to its near edge instead of moving past it (as in every native text field).
    pub fn move_left(&mut self, extend: bool, word: bool) {
        let selection = self.selection();
        let target = if !extend && !selection.is_empty() {
            selection.start
        } else if word {
            self.prev_word(self.caret)
        } else {
            self.prev_grapheme(self.caret)
        };
        self.move_to(target, extend);
    }

    pub fn move_right(&mut self, extend: bool, word: bool) {
        let selection = self.selection();
        let target = if !extend && !selection.is_empty() {
            selection.end
        } else if word {
            self.next_word(self.caret)
        } else {
            self.next_grapheme(self.caret)
        };
        self.move_to(target, extend);
    }

    pub fn home(&mut self, extend: bool) {
        self.move_to(0, extend);
    }

    pub fn end(&mut self, extend: bool) {
        self.move_to(self.text.len(), extend);
    }

    pub fn select_all(&mut self) {
        self.anchor = 0;
        self.caret = self.text.len();
        self.run = None;
    }

    /// Select the word (or run of spaces/punctuation) containing `pos` — double-click.
    pub fn select_word_at(&mut self, pos: usize) {
        let pos = self.snap(pos);
        let segment = self
            .text
            .split_word_bound_indices()
            .map(|(start, word)| start..start + word.len())
            .find(|range| range.contains(&pos) || (pos == self.text.len() && range.end == pos));
        if let Some(range) = segment {
            self.anchor = range.start;
            self.caret = range.end;
            self.run = None;
        }
    }

    /// Type or paste `text` over the selection. Newlines become spaces (single-line field).
    /// Returns whether the contents changed.
    pub fn insert(&mut self, text: &str) -> bool {
        let text = single_line(text);
        if text.is_empty() && self.selection().is_empty() {
            return false;
        }
        // A paste, or typing over a selection, is its own undo step; plain typing merges.
        let kind = (text.graphemes(true).count() == 1 && self.selection().is_empty()).then_some(EditKind::Typing);
        self.checkpoint(kind);
        let selection = self.selection();
        self.text.replace_range(selection.clone(), &text);
        self.caret = selection.start + text.len();
        self.anchor = self.caret;
        self.run = kind;
        true
    }

    /// Backspace: delete the selection, else the grapheme (or word) before the caret.
    pub fn backspace(&mut self, word: bool) -> bool {
        let selection = self.selection();
        let range = if !selection.is_empty() {
            selection
        } else if word {
            self.prev_word(self.caret)..self.caret
        } else {
            self.prev_grapheme(self.caret)..self.caret
        };
        self.delete_range(range)
    }

    /// Forward delete: the selection, else the grapheme (or word) after the caret.
    pub fn delete(&mut self, word: bool) -> bool {
        let selection = self.selection();
        let range = if !selection.is_empty() {
            selection
        } else if word {
            self.caret..self.next_word(self.caret)
        } else {
            self.caret..self.next_grapheme(self.caret)
        };
        self.delete_range(range)
    }

    /// Remove and return the selection (Cut), or `None` when nothing is selected.
    pub fn cut(&mut self) -> Option<String> {
        let selection = self.selection();
        if selection.is_empty() {
            return None;
        }
        let text = self.text[selection.clone()].to_owned();
        self.checkpoint(None);
        self.remove(selection);
        Some(text)
    }

    pub fn undo(&mut self) -> bool {
        let Some(previous) = self.undo.pop() else { return false };
        self.redo.push(self.snapshot());
        self.restore(previous);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else { return false };
        self.undo.push(self.snapshot());
        self.restore(next);
        true
    }

    /// What the field shows while an IME composition `preedit` (text, cursor range within it)
    /// is in progress: the composition inserted at the caret. Returns that text, the caret's
    /// position in it, and the composition's byte range; without a composition, just the text.
    pub fn composed(&self, preedit: Option<&(String, Option<(usize, usize)>)>) -> (String, usize, Option<Range<usize>>) {
        match preedit {
            Some((text, cursor)) if !text.is_empty() => {
                let at = self.caret;
                let composed = format!("{}{}{}", &self.text[..at], text, &self.text[at..]);
                let caret = at + cursor.map_or(text.len(), |(_, end)| end.min(text.len()));
                (composed, caret, Some(at..at + text.len()))
            }
            _ => (self.text.clone(), self.caret, None),
        }
    }

    fn delete_range(&mut self, range: Range<usize>) -> bool {
        if range.is_empty() {
            return false;
        }
        let kind = (self.selection().is_empty()).then_some(EditKind::Deleting);
        self.checkpoint(kind);
        self.remove(range);
        self.run = kind;
        true
    }

    fn remove(&mut self, range: Range<usize>) {
        self.text.replace_range(range.clone(), "");
        self.caret = range.start;
        self.anchor = range.start;
    }

    /// Save the state before an edit, unless it continues the current run of `kind` edits.
    fn checkpoint(&mut self, kind: Option<EditKind>) {
        if kind.is_some() && kind == self.run {
            return;
        }
        self.undo.push(self.snapshot());
        if self.undo.len() > UNDO_LIMIT {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot { text: self.text.clone(), caret: self.caret, anchor: self.anchor }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.text = snapshot.text;
        self.caret = snapshot.caret;
        self.anchor = snapshot.anchor;
        self.run = None;
    }

    /// The nearest grapheme boundary at or before `pos` (clamped to the text).
    fn snap(&self, pos: usize) -> usize {
        if pos >= self.text.len() {
            return self.text.len();
        }
        self.text
            .grapheme_indices(true)
            .map(|(start, _)| start)
            .take_while(|&start| start <= pos)
            .last()
            .unwrap_or(0)
    }

    fn prev_grapheme(&self, pos: usize) -> usize {
        self.text[..pos].grapheme_indices(true).next_back().map_or(0, |(start, _)| start)
    }

    fn next_grapheme(&self, pos: usize) -> usize {
        self.text[pos..].graphemes(true).next().map_or(pos, |g| pos + g.len())
    }

    /// Start of the word before `pos`, skipping any spaces/punctuation in between.
    fn prev_word(&self, pos: usize) -> usize {
        self.text[..pos]
            .split_word_bound_indices()
            .rev()
            .find(|(_, segment)| is_word(segment))
            .map_or(0, |(start, _)| start)
    }

    /// End of the word after `pos`, skipping any spaces/punctuation in between.
    fn next_word(&self, pos: usize) -> usize {
        self.text[pos..]
            .split_word_bound_indices()
            .find(|(_, segment)| is_word(segment))
            .map_or(self.text.len(), |(start, segment)| pos + start + segment.len())
    }
}

fn is_word(segment: &str) -> bool {
    segment.chars().any(char::is_alphanumeric)
}

fn single_line(text: &str) -> String {
    text.replace("\r\n", " ").replace(['\n', '\r'], " ")
}

/// Where text lands horizontally, for a font the caller owns — `fastgui-chrome` implements it
/// with real shaping. Everything is in the same units as `font_size`, measured from the start
/// of the unscrolled line.
pub trait TextMeasure {
    /// X of the caret placed before byte `index` of `text`.
    fn caret_x(&mut self, text: &str, font_size: f32, index: usize) -> f32;
    /// The byte offset whose caret position is nearest `x` (click-to-caret).
    fn index_at(&mut self, text: &str, font_size: f32, x: f32) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_and_backspace_respect_graphemes() {
        let mut edit = TextEdit::new("");
        for c in ["h", "é", "👍🏽"] {
            edit.insert(c);
        }
        assert_eq!(edit.text(), "hé👍🏽");
        assert!(edit.backspace(false));
        assert_eq!(edit.text(), "hé", "the skin-tone emoji goes as one unit");
        edit.move_left(false, false);
        assert_eq!(edit.caret(), 1);
        edit.delete(false);
        assert_eq!(edit.text(), "h");
    }

    #[test]
    fn word_motion_skips_spaces_and_punctuation() {
        let mut edit = TextEdit::new("hello,  big world");
        edit.move_left(false, true);
        assert_eq!(&edit.text()[edit.caret()..], "world");
        edit.move_left(false, true);
        assert_eq!(&edit.text()[edit.caret()..], "big world");
        edit.move_left(false, true);
        assert_eq!(edit.caret(), 0);
        edit.move_right(false, true);
        assert_eq!(&edit.text()[..edit.caret()], "hello");
        edit.move_right(false, true);
        assert_eq!(&edit.text()[..edit.caret()], "hello,  big");
        assert!(edit.backspace(true));
        assert_eq!(edit.text(), "hello,   world");
    }

    #[test]
    fn selection_extend_collapse_and_replace() {
        let mut edit = TextEdit::new("abcdef");
        edit.move_left(true, false);
        edit.move_left(true, false);
        assert_eq!(edit.selected_text(), "ef");
        edit.move_right(false, false);
        assert_eq!((edit.caret(), edit.selection().is_empty()), (6, true), "collapses to the right edge");
        edit.home(true);
        assert_eq!(edit.selected_text(), "abcdef");
        edit.insert("X");
        assert_eq!(edit.text(), "X");
        edit.select_all();
        assert_eq!(edit.cut().as_deref(), Some("X"));
        assert_eq!(edit.text(), "");
        assert_eq!(edit.cut(), None);
    }

    #[test]
    fn undo_merges_typing_runs_and_redo_restores() {
        let mut edit = TextEdit::new("");
        for c in "hi".chars() {
            edit.insert(&c.to_string());
        }
        edit.move_left(false, false);
        edit.move_right(false, false);
        for c in " there".chars() {
            edit.insert(&c.to_string());
        }
        assert_eq!(edit.text(), "hi there");
        assert!(edit.undo());
        assert_eq!(edit.text(), "hi", "the second typing run undoes as one step");
        assert!(edit.undo());
        assert_eq!(edit.text(), "");
        assert!(!edit.undo());
        assert!(edit.redo());
        assert_eq!((edit.text(), edit.caret()), ("hi", 2));
        edit.insert("!");
        assert!(!edit.redo(), "a new edit clears redo");
    }

    #[test]
    fn newlines_flatten_and_double_click_selects_word() {
        let mut edit = TextEdit::new("");
        edit.insert("one\ntwo\r\nthree");
        assert_eq!(edit.text(), "one two three");
        edit.select_word_at(5);
        assert_eq!(edit.selected_text(), "two");
        edit.select_word_at(edit.text().len());
        assert_eq!(edit.selected_text(), "three", "past the end picks the last word");
        edit.move_to(2, false);
        edit.move_to(99, true);
        assert_eq!(edit.selected_text(), "e two three", "out-of-range positions clamp");
    }
}
