//! `enabled` and drag-and-drop for widgets (`set_enabled`, `set_drag_source`, `set_drop_target`,
//! `set_file_drop`). Each widget object owns one `Interaction`: its setters update it and push
//! the change to the window when the widget is shown, and `describe` turns it into an
//! `InteractionSpec` that `attach` applies, so the state survives rebuilds (`set_theme`,
//! `set_content`) like the rest of a widget's state.

use std::sync::{Arc, Mutex, MutexGuard};

use fastgui_core::dnd::{
    DragDataCallback, DragOrigin, DragSource, DropCallback, DropEvent, DropPosition, DropTarget, FileDropCallback,
};
use fastgui_core::widget::{WidgetId, WidgetTree};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::backend::{Command, CommandDispatch};
use crate::widgets::{IdCell, SenderCell};

struct State {
    enabled: bool,
    /// `(tag, data callable)`; `None` data sends a default payload (see `default_payload`).
    drag: Option<(String, Option<Py<PyAny>>)>,
    /// `(accepted tags, on_drop)`; no tags accepts any.
    drop: Option<(Vec<String>, Py<PyAny>)>,
    file_drop: Option<Py<PyAny>>,
}

pub(crate) struct Interaction {
    state: Mutex<State>,
}

/// What `attach` applies to a freshly created node: plain Rust callbacks, built under the GIL
/// while describing so attaching needs no Python.
pub(crate) struct InteractionSpec {
    disabled: bool,
    drag: Option<DragSource>,
    drop: Option<DropTarget>,
    file_drop: Option<FileDropCallback>,
}

impl InteractionSpec {
    pub(crate) fn apply(self, tree: &mut WidgetTree, id: WidgetId) {
        if self.disabled {
            tree.set_disabled(id, true);
        }
        tree.set_drag_source(id, self.drag);
        tree.set_drop_target(id, self.drop);
        tree.set_file_drop(id, self.file_drop);
    }
}

impl Interaction {
    pub(crate) fn new(enabled: bool) -> Self {
        Self { state: Mutex::new(State { enabled, drag: None, drop: None, file_drop: None }) }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn enabled(&self) -> bool {
        self.lock().enabled
    }

    pub(crate) fn spec(&self) -> InteractionSpec {
        // Attach before locking, the same order as the setters (called attached).
        Python::attach(|py| {
            let state = self.lock();
            InteractionSpec {
                disabled: !state.enabled,
                drag: state.drag.as_ref().map(|(tag, data)| drag_source(tag.clone(), data.as_ref().map(|d| d.clone_ref(py)))),
                drop: state.drop.as_ref().map(|(accepts, cb)| drop_target(accepts.clone(), cb.clone_ref(py))),
                file_drop: state.file_drop.as_ref().map(|cb| file_drop(cb.clone_ref(py))),
            }
        })
    }

    pub(crate) fn set_enabled(&self, enabled: bool, id: &IdCell, sender: &SenderCell) -> PyResult<()> {
        self.lock().enabled = enabled;
        on_window(id, sender, move |tree, id| tree.set_disabled(id, !enabled))
    }

    pub(crate) fn set_drag_source(
        &self,
        tag: Option<String>,
        data: Option<Py<PyAny>>,
        id: &IdCell,
        sender: &SenderCell,
    ) -> PyResult<()> {
        let source = Python::attach(|py| {
            tag.as_ref().map(|tag| drag_source(tag.clone(), data.as_ref().map(|d| d.clone_ref(py))))
        });
        self.lock().drag = tag.map(|tag| (tag, data));
        on_window(id, sender, move |tree, id| tree.set_drag_source(id, source))
    }

    pub(crate) fn set_drop_target(
        &self,
        py: Python<'_>,
        accept: Option<&Bound<'_, PyAny>>,
        on_drop: Option<Py<PyAny>>,
        id: &IdCell,
        sender: &SenderCell,
    ) -> PyResult<()> {
        let accepts = parse_accept(accept)?;
        let target = on_drop.as_ref().map(|cb| drop_target(accepts.clone(), cb.clone_ref(py)));
        self.lock().drop = on_drop.map(|cb| (accepts, cb));
        on_window(id, sender, move |tree, id| tree.set_drop_target(id, target))
    }

    pub(crate) fn set_file_drop(&self, on_drop: Option<Py<PyAny>>, id: &IdCell, sender: &SenderCell) -> PyResult<()> {
        let callback = Python::attach(|py| on_drop.as_ref().map(|cb| file_drop(cb.clone_ref(py))));
        self.lock().file_drop = on_drop;
        on_window(id, sender, move |tree, id| tree.set_file_drop(id, callback))
    }
}

/// Run `f` on the window the widget is shown in; a no-op before it's shown (the state is
/// applied at attach) or after the window closed.
fn on_window(
    id: &IdCell,
    sender: &SenderCell,
    f: impl FnOnce(&mut WidgetTree, WidgetId) + Send + 'static,
) -> PyResult<()> {
    let Some(id) = *id.lock().unwrap_or_else(|p| p.into_inner()) else { return Ok(()) };
    let Some(sender) = sender.lock().unwrap_or_else(|p| p.into_inner()).clone() else { return Ok(()) };
    send(&sender, move |tree| f(tree, id));
    Ok(())
}

fn send(sender: &CommandDispatch, mutation: impl FnOnce(&mut WidgetTree) + Send + 'static) {
    let command = match sender.floating_region {
        Some(region_id) => Command::MutateFloatingTree { region_id, mutation: Box::new(mutation) },
        None => Command::MutateWidgetTree(Box::new(mutation)),
    };
    // A closed window has nothing left to update.
    let _ = sender.send(command);
}

/// `accept=`: `None` (any tag), one tag, or a sequence of tags.
fn parse_accept(accept: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(accept) = accept.filter(|a| !a.is_none()) else { return Ok(Vec::new()) };
    if let Ok(tag) = accept.extract::<String>() {
        return Ok(vec![tag]);
    }
    accept
        .extract::<Vec<String>>()
        .map_err(|_| PyTypeError::new_err("accept must be None, a tag string, or a sequence of tag strings"))
}

/// The payload a drag sends when the source gave no `data` callable: the row index, the node's
/// path as `"0/2/1"`, or nothing — enough to reorder within one list or tree.
fn default_payload(origin: &DragOrigin) -> Vec<u8> {
    match origin {
        DragOrigin::Row(index) => index.to_string().into_bytes(),
        DragOrigin::Node(path) => path.iter().map(u32::to_string).collect::<Vec<_>>().join("/").into_bytes(),
        DragOrigin::Widget => Vec::new(),
    }
}

fn drag_source(tag: String, data: Option<Py<PyAny>>) -> DragSource {
    let data: DragDataCallback = match data {
        None => Arc::new(|origin| Some(default_payload(origin))),
        Some(callback) => Arc::new(move |origin| {
            Python::attach(|py| {
                let result = match origin {
                    DragOrigin::Row(index) => callback.call1(py, (*index,)),
                    DragOrigin::Node(path) => callback.call1(py, (path.clone(),)),
                    DragOrigin::Widget => callback.call0(py),
                };
                match result {
                    Ok(value) => payload_bytes(value.bind(py)),
                    Err(err) => {
                        err.print(py);
                        None
                    }
                }
            })
        }),
    };
    DragSource { tag, data }
}

/// A data callable's return value as bytes: `bytes`, `str` (UTF-8), or `None` to cancel.
fn payload_bytes(value: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    if value.is_none() {
        return None;
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Some(bytes.as_bytes().to_vec());
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Some(text.to_string().into_bytes());
    }
    let py = value.py();
    PyTypeError::new_err("a drag data callable must return bytes, str, or None").print(py);
    None
}

fn drop_target(accepts: Vec<String>, callback: Py<PyAny>) -> DropTarget {
    let on_drop: DropCallback = Arc::new(move |event: DropEvent| {
        Python::attach(|py| {
            let data = PyBytes::new(py, &event.data);
            let result = match event.position {
                DropPosition::ListGap(index) => callback.call1(py, (event.tag, data, index)),
                DropPosition::TreeNode { path, place } => callback.call1(py, (event.tag, data, path, place.as_str())),
                DropPosition::Widget { x, y } => callback.call1(py, (event.tag, data, x, y)),
            };
            if let Err(err) = result {
                err.print(py);
            }
        });
    });
    DropTarget { accepts, on_drop }
}

fn file_drop(callback: Py<PyAny>) -> FileDropCallback {
    Arc::new(move |paths, x, y| {
        let paths: Vec<String> = paths.iter().map(|p| p.to_string_lossy().into_owned()).collect();
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (paths, x, y)) {
                err.print(py);
            }
        });
    })
}
