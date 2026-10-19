//! The colors chrome draws on its own — not taken from any widget's fields: the window
//! background, the keyboard focus ring and text caret, the dimming behind a modal popup, the
//! dock drop preview, and scrollbars on widgets without their own bar color — plus the font
//! family all chrome text is shaped in. One per process
//! (like Qt's application palette), set from Python's `fg.set_theme`; widget colors come from
//! the same Python `Theme` when widgets are described.

use std::sync::{Arc, RwLock};

use crate::widget::Color;

#[derive(Clone, Debug, PartialEq)]
pub struct ChromeTheme {
    pub background: Color,
    /// Focus ring and text caret.
    pub accent: Color,
    pub scrim: Color,
    pub drop_indicator: Color,
    pub scrollbar: Color,
    /// State layer drawn over a hovered control (translucent; menu rows and flat buttons
    /// included) — from Python `Theme.hover`.
    pub hover: Color,
    /// Veil drawn over a disabled widget's subtree (translucent) — from Python `Theme.disabled`.
    pub disabled: Color,
    /// Raised chrome (tooltips, popup chrome) — from Python `Theme.surface_alt`.
    pub surface_alt: Color,
    /// Primary chrome text — from Python `Theme.text`.
    pub text: Color,
    /// Chrome outlines — from Python `Theme.border`.
    pub border: Color,
    /// Font family name for all text (`None`: the system's default sans-serif). A name the
    /// system doesn't have falls back to that default.
    pub font_family: Option<Arc<str>>,
}

impl ChromeTheme {
    pub const DARK: ChromeTheme = ChromeTheme {
        background: Color([0.10, 0.11, 0.13, 1.0]),
        accent: Color([0.4, 0.7, 1.0, 1.0]),
        scrim: Color([0.0, 0.0, 0.0, 0.45]),
        drop_indicator: Color([0.40, 0.65, 1.0, 0.35]),
        scrollbar: Color([1.0, 1.0, 1.0, 0.35]),
        hover: Color([1.0, 1.0, 1.0, 0.08]),
        disabled: Color([0.10, 0.11, 0.13, 0.55]),
        surface_alt: Color([0.16, 0.17, 0.20, 1.0]),
        text: Color([0.92, 0.93, 0.95, 1.0]),
        border: Color([0.32, 0.35, 0.42, 1.0]),
        font_family: None,
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
    CURRENT.read().unwrap_or_else(|p| p.into_inner()).clone()
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
        set_chrome_theme(custom.clone());
        assert_eq!(chrome_theme(), custom);
        set_chrome_theme(ChromeTheme::DARK);
        assert_eq!(chrome_theme(), ChromeTheme::DARK);
    }
}
