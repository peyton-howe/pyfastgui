//! The colors chrome draws on its own — not taken from any widget's fields: the window
//! background, the keyboard focus ring and text caret, the dimming behind a modal popup, the
//! dock drop preview, and scrollbars on widgets without their own bar color. One per process
//! (like Qt's application palette), set from Python's `fg.set_theme`; widget colors come from
//! the same Python `Theme` when widgets are described.

use std::sync::RwLock;

use crate::widget::Color;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromeTheme {
    pub background: Color,
    /// Focus ring and text caret.
    pub accent: Color,
    pub scrim: Color,
    pub drop_indicator: Color,
    pub scrollbar: Color,
}

impl ChromeTheme {
    pub const DARK: ChromeTheme = ChromeTheme {
        background: Color([0.10, 0.11, 0.13, 1.0]),
        accent: Color([0.4, 0.7, 1.0, 1.0]),
        scrim: Color([0.0, 0.0, 0.0, 0.45]),
        drop_indicator: Color([0.40, 0.65, 1.0, 0.35]),
        scrollbar: Color([1.0, 1.0, 1.0, 0.35]),
    };
}

impl Default for ChromeTheme {
    fn default() -> Self {
        Self::DARK
    }
}

static CURRENT: RwLock<ChromeTheme> = RwLock::new(ChromeTheme::DARK);

/// The current chrome theme. Cheap; chrome reads it once per frame build.
pub fn chrome_theme() -> ChromeTheme {
    *CURRENT.read().unwrap_or_else(|p| p.into_inner())
}

/// Replace the chrome theme. Windows pick it up the next time their chrome is rebuilt (Python's
/// `Window.set_theme` rebuilds the content, which forces that).
pub fn set_chrome_theme(theme: ChromeTheme) {
    *CURRENT.write().unwrap_or_else(|p| p.into_inner()) = theme;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chrome_theme_round_trips_and_defaults_to_dark() {
        assert_eq!(ChromeTheme::default(), ChromeTheme::DARK);
        let custom = ChromeTheme { accent: Color([1.0, 0.5, 0.0, 1.0]), ..ChromeTheme::DARK };
        set_chrome_theme(custom);
        assert_eq!(chrome_theme(), custom);
        set_chrome_theme(ChromeTheme::DARK);
        assert_eq!(chrome_theme(), ChromeTheme::DARK);
    }
}
