//! Native file pickers via `rfd`. Color picking is in-app (Python) — rfd has no ColorDialog.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

fn apply_filters(mut dialog: rfd::FileDialog, filter: Option<&Bound<'_, PyList>>) -> PyResult<rfd::FileDialog> {
    if let Some(filter) = filter {
        for item in filter.iter() {
            let (name, extensions): (String, Vec<String>) = item.extract().map_err(|_| {
                PyValueError::new_err(
                    "filter must be a sequence of (name, [ext, ...]) pairs, e.g. [(\"Text\", [\"txt\"])]",
                )
            })?;
            let ext_refs: Vec<&str> = extensions.iter().map(String::as_str).collect();
            dialog = dialog.add_filter(name, &ext_refs);
        }
    }
    Ok(dialog)
}

/// Native open-file dialog. Returns a path string, or `None` if cancelled.
/// Prefer calling from a UI callback (already on the window/event-loop thread).
#[pyfunction]
#[pyo3(signature = (title=None, filter=None))]
fn open_file_dialog(title: Option<String>, filter: Option<Bound<'_, PyList>>) -> PyResult<Option<String>> {
    let mut dialog = rfd::FileDialog::new();
    if let Some(title) = title {
        dialog = dialog.set_title(title);
    }
    dialog = apply_filters(dialog, filter.as_ref())?;
    Ok(dialog.pick_file().map(|p| p.to_string_lossy().into_owned()))
}

/// Native save-file dialog. Returns a path string, or `None` if cancelled.
#[pyfunction]
#[pyo3(signature = (title=None, filter=None))]
fn save_file_dialog(title: Option<String>, filter: Option<Bound<'_, PyList>>) -> PyResult<Option<String>> {
    let mut dialog = rfd::FileDialog::new();
    if let Some(title) = title {
        dialog = dialog.set_title(title);
    }
    dialog = apply_filters(dialog, filter.as_ref())?;
    Ok(dialog.save_file().map(|p| p.to_string_lossy().into_owned()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_file_dialog, m)?)?;
    m.add_function(wrap_pyfunction!(save_file_dialog, m)?)?;
    Ok(())
}
