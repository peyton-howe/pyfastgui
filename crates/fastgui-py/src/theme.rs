//! `fg.Theme`: the named colors, font sizes and spacings widgets take when their own arguments
//! are left out or given by name (`font_size="large"`, `gap="medium"`), plus the colors and
//! font family chrome draws with (pushed to `fastgui_core::theme`). One current theme per
//! process, like Qt's application palette; `fg.set_theme` swaps it for widgets described from
//! then on, and `Window.set_theme` also rebuilds a window so it shows at once.

use std::sync::{Arc, RwLock};

use fastgui_core::theme::{set_chrome_theme, ChromeTheme};
use fastgui_core::widget::Color;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Rgba = (f32, f32, f32, f32);

macro_rules! theme {
    ($($(#[doc = $doc:literal])* $name:ident),* $(,)?) => {
        /// A set of named colors. Build one with `Theme.dark()` / `Theme.light()`, then
        /// `replace(accent=..., ...)` to adjust; pass it to `fg.set_theme` or `Window.set_theme`.
        #[pyclass(get_all, frozen, from_py_object)]
        #[derive(Clone, Debug)]
        pub(crate) struct Theme {
            $($(#[doc = $doc])* pub(crate) $name: Rgba,)*
            /// Font family for all text; `None` is the system's default sans-serif.
            pub(crate) font_family: Option<String>,
            /// `font_size="small"`: lists, tabs, panel titles.
            pub(crate) font_size_small: f32,
            /// `font_size="body"`: labels, buttons, text fields.
            pub(crate) font_size: f32,
            /// `font_size="large"`: headings.
            pub(crate) font_size_large: f32,
            /// `gap=` / `padding="small"`.
            pub(crate) spacing_small: f32,
            /// `gap=` / `padding="medium"`.
            pub(crate) spacing: f32,
            /// `gap=` / `padding="large"`.
            pub(crate) spacing_large: f32,
        }

        impl Theme {
            fn set(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
                match name {
                    $(stringify!($name) => self.$name = value.extract()?,)*
                    "font_family" => self.font_family = value.extract()?,
                    "font_size_small" => self.font_size_small = positive(name, value.extract()?)?,
                    "font_size" => self.font_size = positive(name, value.extract()?)?,
                    "font_size_large" => self.font_size_large = positive(name, value.extract()?)?,
                    "spacing_small" => self.spacing_small = non_negative(name, value.extract()?)?,
                    "spacing" => self.spacing = non_negative(name, value.extract()?)?,
                    "spacing_large" => self.spacing_large = non_negative(name, value.extract()?)?,
                    other => return Err(PyTypeError::new_err(format!("Theme has no token {other:?}"))),
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
    /// Dock and widget drag-and-drop preview.
    drop_indicator,
    /// Translucent state layer over a hovered control.
    hover,
    /// Translucent veil over a disabled widget (usually the background color, part opaque).
    disabled,
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
    hover: (1.0, 1.0, 1.0, 0.08),
    disabled: (0.10, 0.11, 0.13, 0.55),
    font_family: None,
    font_size_small: 14.0,
    font_size: 16.0,
    font_size_large: 22.0,
    spacing_small: 4.0,
    spacing: 8.0,
    spacing_large: 16.0,
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
    button: (0.84, 0.86, 0.90, 1.0),
    button_text: (0.10, 0.11, 0.13, 1.0),
    selection: (0.20, 0.45, 0.90, 0.30),
    scrollbar: (0.0, 0.0, 0.0, 0.30),
    scrim: (0.0, 0.0, 0.0, 0.30),
    drop_indicator: (0.20, 0.45, 0.90, 0.30),
    hover: (0.0, 0.0, 0.0, 0.06),
    disabled: (0.94, 0.95, 0.96, 0.60),
    ..DARK
};

static CURRENT: RwLock<Theme> = RwLock::new(DARK);

/// The current theme; widget `describe`s take their default colors from it.
pub(crate) fn palette() -> Theme {
    CURRENT.read().unwrap_or_else(|p| p.into_inner()).clone()
}

/// Make `theme` current for widgets and chrome.
pub(crate) fn install(theme: Theme) {
    let color = |c: Rgba| Color([c.0, c.1, c.2, c.3]);
    set_chrome_theme(ChromeTheme {
        background: color(theme.background),
        accent: color(theme.accent),
        scrim: color(theme.scrim),
        drop_indicator: color(theme.drop_indicator),
        scrollbar: color(theme.scrollbar),
        hover: color(theme.hover),
        disabled: color(theme.disabled),
        surface_alt: color(theme.surface_alt),
        text: color(theme.text),
        border: color(theme.border),
        font_family: theme.font_family.as_deref().map(Arc::from),
    });
    *CURRENT.write().unwrap_or_else(|p| p.into_inner()) = theme;
}

fn positive(name: &str, value: f32) -> PyResult<f32> {
    if value > 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} must be a positive number, got {value}")))
    }
}

fn non_negative(name: &str, value: f32) -> PyResult<f32> {
    if value >= 0.0 && value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} can't be negative, got {value}")))
    }
}

/// A `font_size=` argument: points, or a theme size by name, resolved against the theme
/// current when the widget is described (so `Window.set_theme` resizes it too).
#[derive(Clone, Copy, Debug)]
pub(crate) enum FontSize {
    Points(f32),
    Small,
    Body,
    Large,
}

impl FontSize {
    pub(crate) fn resolve(self) -> f32 {
        let theme = CURRENT.read().unwrap_or_else(|p| p.into_inner());
        match self {
            FontSize::Points(points) => points,
            FontSize::Small => theme.font_size_small,
            FontSize::Body => theme.font_size,
            FontSize::Large => theme.font_size_large,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for FontSize {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(name) = obj.extract::<String>() {
            return match name.as_str() {
                "small" => Ok(FontSize::Small),
                "body" => Ok(FontSize::Body),
                "large" => Ok(FontSize::Large),
                other => Err(PyValueError::new_err(format!(
                    "font_size must be a number or \"small\", \"body\" or \"large\", got {other:?}"
                ))),
            };
        }
        Ok(FontSize::Points(positive("font_size", obj.extract()?)?))
    }
}

/// A `gap=` / `padding=` argument: layout units, or a theme spacing by name.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Spacing {
    Units(f32),
    Small,
    Medium,
    Large,
}

impl Spacing {
    pub(crate) fn resolve(self) -> f32 {
        let theme = CURRENT.read().unwrap_or_else(|p| p.into_inner());
        match self {
            Spacing::Units(units) => units,
            Spacing::Small => theme.spacing_small,
            Spacing::Medium => theme.spacing,
            Spacing::Large => theme.spacing_large,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Spacing {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(name) = obj.extract::<String>() {
            return match name.as_str() {
                "small" => Ok(Spacing::Small),
                "medium" => Ok(Spacing::Medium),
                "large" => Ok(Spacing::Large),
                other => Err(PyValueError::new_err(format!(
                    "spacing must be a number or \"small\", \"medium\" or \"large\", got {other:?}"
                ))),
            };
        }
        Ok(Spacing::Units(non_negative("spacing", obj.extract()?)?))
    }
}

#[pymethods]
impl Theme {
    /// The default dark theme, with any tokens given as keywords replaced.
    #[new]
    #[pyo3(signature = (**tokens))]
    fn new(tokens: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        DARK.replace(tokens)
    }

    #[staticmethod]
    fn dark() -> Self {
        DARK
    }

    #[staticmethod]
    fn light() -> Self {
        LIGHT
    }

    /// A copy with the given tokens changed, e.g.
    /// `theme.replace(accent=(1, 0.5, 0, 1), font_size=18, font_family="Menlo")`.
    #[pyo3(signature = (**tokens))]
    fn replace(&self, tokens: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut theme = self.clone();
        for (name, value) in tokens.into_iter().flatten() {
            theme.set(&name.extract::<String>()?, &value)?;
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
