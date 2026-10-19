//! `Viewport.submit_metal` / `Image.submit_metal`. Metal validates and retains on macOS;
//! every other target refuses before a frame is queued.

use fastgui_core::ImportedLayer;
#[cfg(not(target_os = "macos"))]
use pyo3::exceptions::PyRuntimeError;
#[cfg(target_os = "macos")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn prepare(
    texture: &Bound<'_, PyAny>,
    wait_event: Option<&Bound<'_, PyAny>>,
    wait_value: u64,
    signal_event: Option<&Bound<'_, PyAny>>,
    signal_value: u64,
) -> PyResult<ImportedLayer> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (texture, wait_event, wait_value, signal_event, signal_value);
        Err(PyRuntimeError::new_err(
            "submit_metal is only available on macOS. Use submit_frame or submit_cuda on this platform.",
        ))
    }
    #[cfg(target_os = "macos")]
    {
        let texture = metal_pointer(texture, "texture")?;
        let wait_event = optional_pointer(wait_event, "wait_event")?;
        let signal_event = optional_pointer(signal_event, "signal_event")?;
        fastgui_render_mtl::prepare_metal_import(texture, wait_event, wait_value, signal_event, signal_value)
            .map_err(PyValueError::new_err)
    }
}

#[cfg(target_os = "macos")]
fn metal_pointer(value: &Bound<'_, PyAny>, label: &str) -> PyResult<usize> {
    let ptr = pointer_value(value, label)?;
    if ptr == 0 {
        return Err(PyValueError::new_err(format!("{label} pointer is null")));
    }
    Ok(ptr)
}

#[cfg(target_os = "macos")]
fn optional_pointer(value: Option<&Bound<'_, PyAny>>, label: &str) -> PyResult<usize> {
    let Some(value) = value else { return Ok(0) };
    if value.is_none() {
        return Ok(0);
    }
    pointer_value(value, label)
}

#[cfg(target_os = "macos")]
fn pointer_value(value: &Bound<'_, PyAny>, label: &str) -> PyResult<usize> {
    if let Ok(ptr) = value.extract::<usize>() {
        return Ok(ptr);
    }
    if let Ok(attr) = value.getattr("mtl_ptr") {
        if let Ok(ptr) = attr.extract::<usize>() {
            return Ok(ptr);
        }
    }
    Err(PyValueError::new_err(format!(
        "{label} must be an integer Metal pointer or an object with mtl_ptr"
    )))
}
