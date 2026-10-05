//! Drag-and-drop between widgets, and files dropped from the OS.
//!
//! The payload is a string `tag` plus bytes — no mime registry. A widget offers drags with a
//! [`DragSource`] (`WidgetTree::set_drag_source`) and accepts them with a [`DropTarget`]
//! (`set_drop_target`); `fastgui-app` runs the gesture (same threshold and drop preview as dock
//! drag) and resolves targets with `WidgetTree::drop_target_at`. OS file drops go to a
//! [`FileDropCallback`] (`set_file_drop`).

use std::path::PathBuf;
use std::sync::Arc;

use crate::widget::Rect;

/// Where in the source widget the drag started, handed to the source's data callback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DragOrigin {
    /// A `ListView` row.
    Row(usize),
    /// A `TreeView` node, as its child-index path from the roots.
    Node(Vec<u32>),
    /// Anywhere else on the widget.
    Widget,
}

/// The bytes a drag carries, built when the drag actually starts (`None` cancels it).
pub type DragDataCallback = Arc<dyn Fn(&DragOrigin) -> Option<Vec<u8>> + Send + Sync>;

pub struct DragSource {
    pub tag: String,
    pub data: DragDataCallback,
}

/// Where a `TreeView` drop lands relative to the node under the cursor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreePlace {
    Before,
    Inside,
    After,
}

impl TreePlace {
    pub fn as_str(self) -> &'static str {
        match self {
            TreePlace::Before => "before",
            TreePlace::Inside => "inside",
            TreePlace::After => "after",
        }
    }
}

/// Where a drop lands in the target widget.
#[derive(Clone, Debug, PartialEq)]
pub enum DropPosition {
    /// `ListView`: insert before row `index` (`index == len` appends).
    ListGap(usize),
    /// `TreeView`: relative to the node at `path`.
    TreeNode { path: Vec<u32>, place: TreePlace },
    /// Any other widget: the point in the widget's own coordinates (layout units).
    Widget { x: f32, y: f32 },
}

pub struct DropEvent {
    pub tag: String,
    pub data: Vec<u8>,
    pub position: DropPosition,
}

pub type DropCallback = Arc<dyn Fn(DropEvent) + Send + Sync>;

pub struct DropTarget {
    /// Tags this target takes; empty takes any.
    pub accepts: Vec<String>,
    pub on_drop: DropCallback,
}

impl DropTarget {
    pub fn accepts(&self, tag: &str) -> bool {
        self.accepts.is_empty() || self.accepts.iter().any(|t| t == tag)
    }
}

/// A resolved drop target under the cursor: the widget, where in it, and the rect the drop
/// preview highlights (window coordinates, layout units).
#[derive(Clone, Debug, PartialEq)]
pub struct DropHit {
    pub target: crate::widget::WidgetId,
    pub position: DropPosition,
    pub preview: Rect,
}

/// Files dropped from the OS onto a widget: the paths, and where (widget coordinates).
pub type FileDropCallback = Arc<dyn Fn(Vec<PathBuf>, f32, f32) + Send + Sync>;

/// Thickness (layout units) of the insertion line a list/tree drop preview draws between rows.
pub const INSERT_LINE: f32 = 2.0;
