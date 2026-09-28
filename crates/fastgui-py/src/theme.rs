//! `fg.Theme`: the named colors widgets take when their own color arguments are left out, plus
//! the colors chrome draws itself (pushed to `fastgui_core::theme`). One current theme per
//! process, like Qt's application palette; `fg.set_theme` swaps it for widgets described from
//! then on, and `Window.set_theme` also rebuilds a window so it shows at once.

use std::sync::RwLock;

use fastgui_core::theme::{set_chrome_theme, ChromeTheme};
use fastgui_core::widget::Color;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Rgba = (f32, f32, f32, f32);

macro_rules! theme {
    ($($(#[doc = $doc:literal])* $name:ident),* $(,)?) => {
        /// A set of named colors. Build one with `Theme.dark()` / `Theme.light()`, then
        /// `replace(accent=..., ...)` to adjust; pass it to `fg.set_theme` or `Window.set_theme`.
        #[pyclass(get_all, frozen, from_py_object)]
        #[derive(Clone, Copy, Debug)]
        pub(crate) struct Theme {
            $($(#[doc = $doc])* pub(crate) $name: Rgba,)*
        }

        impl Theme {
            fn set(&mut self, name: &str, value: Rgba) -> PyResult<()> {
                match name {
                    $(stringify!($name) => self.$name = value,)*
                    other => return Err(PyTypeError::new_err(format!("Theme has no color {other:?}"))),
                }
                Ok(())
            }
        }
    };
}

theme! {
    /// Window background, behind everything.
    background,
    /// Panels, lists.
    surface,
    /// Raised areas: title bars, text fields, inactive tabs, popups.
    surface_alt,
    /// The active tab.
    surface_active,
    /// Popup outlines.
    border,
    /// Splitter bars.
    divider,
    /// Slider tracks.
    track,
    text,
    /// Placeholders and secondary text.
    text_muted,
    /// Focus ring, caret, slider thumbs.
    accent,
    button,
    button_text,
    /// Selected text and list rows.
    selection,
    scrollbar,
    /// Dims the window behind a modal popup.
    scrim,
    /// Dock drag-and-drop preview.
    drop_indicator,
}

const DARK: Theme = Theme {
    background: (0.10, 0.11, 0.13, 1.0),
    surface: (0.12, 0.13, 0.15, 1.0),
    surface_alt: (0.16, 0.17, 0.20, 1.0),
    surface_active: (0.20, 0.22, 0.26, 1.0),
    border: (0.32, 0.35, 0.42, 1.0),
    divider: (0.20, 0.21, 0.24, 1.0),
    track: (0.30, 0.30, 0.35, 1.0),
    text: (0.92, 0.93, 0.95, 1.0),
    text_muted: (0.50, 0.52, 0.56, 1.0),
    accent: (0.40, 0.70, 1.00, 1.0),
    button: (0.25, 0.35, 0.85, 1.0),
    button_text: (1.0, 1.0, 1.0, 1.0),
    selection: (0.25, 0.45, 0.80, 0.6),
    scrollbar: (1.0, 1.0, 1.0, 0.35),
    scrim: (0.0, 0.0, 0.0, 0.45),
    drop_indicator: (0.40, 0.65, 1.00, 0.35),
};

const LIGHT: Theme = Theme {
    background: (0.94, 0.95, 0.96, 1.0),
    surface: (1.0, 1.0, 1.0, 1.0),
    surface_alt: (0.90, 0.91, 0.93, 1.0),
    surface_active: (0.82, 0.85, 0.90, 1.0),
    border: (0.72, 0.74, 0.78, 1.0),
    divider: (0.80, 0.82, 0.85, 1.0),
    track: (0.76, 0.78, 0.82, 1.0),
    text: (0.10, 0.11, 0.13, 1.0),
    text_muted: (0.45, 0.47, 0.50, 1.0),
    accent: (0.15, 0.45, 0.90, 1.0),
    button: (0.20, 0.45, 0.90, 1.0),
    button_text: (1.0, 1.0, 1.0, 1.0),
    selection: (0.20, 0.45, 0.90, 0.30),
    scrollbar: (0.0, 0.0, 0.0, 0.30),
    scrim: (0.0, 0.0, 0.0, 0.30),
    drop_indicator: (0.20, 0.45, 0.90, 0.30),
};

static CURRENT: RwLock<Theme> = RwLock::new(DARK);

/// The current theme; widget `describe`s take their default colors from it.
pub(crate) fn palette() -> Theme {
    *CURRENT.read().unwrap_or_else(|p| p.into_inner())
}

/// Make `theme` current for widgets and chrome.
pub(crate) fn install(theme: Theme) {
    *CURRENT.write().unwrap_or_else(|p| p.into_inner()) = theme;
    let color = |c: Rgba| Color([c.0, c.1, c.2, c.3]);
    set_chrome_theme(ChromeTheme {
        background: color(theme.background),
        accent: color(theme.accent),
        scrim: color(theme.scrim),
        drop_indicator: color(theme.drop_indicator),
        scrollbar: color(theme.scrollbar),
    });
}

#[pymethods]
impl Theme {
    /// The default dark theme, with any colors given as keywords replaced.
    #[new]
    #[pyo3(signature = (**colors))]
    fn new(colors: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        DARK.replace(colors)
    }

    #[staticmethod]
    fn dark() -> Self {
        DARK
    }

    #[staticmethod]
    fn light() -> Self {
        LIGHT
    }

    /// A copy with the given colors changed, e.g. `theme.replace(accent=(1, 0.5, 0, 1))`.
    #[pyo3(signature = (**colors))]
    fn replace(&self, colors: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut theme = *self;
        for (name, value) in colors.into_iter().flatten() {
            theme.set(&name.extract::<String>()?, value.extract()?)?;
        }
        Ok(theme)
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// Make `theme` current: widgets described from now on take their default colors from it, and
/// chrome (window background, focus ring, scrollbars...) uses it on its next rebuild. Use
/// `Window.set_theme` to restyle a window that's already showing.
#[pyfunction]
pub(crate) fn set_theme(theme: Theme) {
    install(theme);
}

/// The current theme.
#[pyfunction]
pub(crate) fn get_theme() -> Theme {
    palette()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Theme>()?;
    m.add_function(wrap_pyfunction!(set_theme, m)?)?;
    m.add_function(wrap_pyfunction!(get_theme, m)?)?;
    Ok(())
}
