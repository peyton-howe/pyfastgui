use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use fastgui_core::taffy::prelude::*;
use fastgui_core::taffy::style::LengthPercentage;
use fastgui_core::widget::{
    ChangeCallback, ClickCallback, Color, PanelDropCallback, SplitDirection, TabSelectCallback, WidgetId, WidgetKind,
    WidgetTree,
};
use fastgui_render_vk::{Command, CommandDispatch};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

type IdCell = Arc<Mutex<Option<WidgetId>>>;
type SenderCell = Arc<Mutex<Option<CommandDispatch>>>;
/// Interior-mutable cell a `Panel`/`Tabs` region's rearrange handler lives in — `None` until a
/// `DockArea` claims it (`Panel::set_rearrange_handler`), so a standalone `Panel` not inside a
/// `DockArea` just has a no-op drag (see `WidgetKind::PanelTitleBar`'s doc comment).
type RearrangeHandlerCell = Arc<Mutex<Option<Py<PyAny>>>>;

/// Every `Panel`/`Tabs` gets a globally unique region id at construction time — the identity
/// `WidgetKind::Container { region_id, .. }` and `WidgetKind::PanelTitleBar { panel_id, .. }`
/// use for drag-and-drop hit-testing (`WidgetTree::find_region_at`) — regardless of how many
/// times its `describe()` gets called across rebuilds, so a `DockArea` rearrange (which rebuilds
/// the *entire* tree from scratch via `Window.set_content`) still recognizes the same panel.
static NEXT_REGION_ID: AtomicU64 = AtomicU64::new(1);
fn next_region_id() -> u64 {
    NEXT_REGION_ID.fetch_add(1, Ordering::Relaxed)
}

fn rgba(color: (f32, f32, f32, f32)) -> Color {
    Color([color.0, color.1, color.2, color.3])
}

fn transparent() -> Color {
    Color::TRANSPARENT
}

fn wrap_callback0(callback: Py<PyAny>) -> ClickCallback {
    Arc::new(move || {
        Python::attach(|py| {
            if let Err(err) = callback.call0(py) {
                err.print(py);
            }
        });
    })
}

fn wrap_callback1(callback: Py<PyAny>) -> ChangeCallback {
    Arc::new(move |value: f32| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (value,)) {
                err.print(py);
            }
        });
    })
}

fn wrap_callback_usize(callback: Py<PyAny>) -> TabSelectCallback {
    Arc::new(move |index: usize| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (index,)) {
                err.print(py);
            }
        });
    })
}

fn drop_zone_str(zone: fastgui_core::widget::DropZone) -> &'static str {
    use fastgui_core::widget::DropZone;
    match zone {
        DropZone::Center => "center",
        DropZone::Left => "left",
        DropZone::Right => "right",
        DropZone::Top => "top",
        DropZone::Bottom => "bottom",
    }
}

fn wrap_panel_drop_callback(callback: Py<PyAny>) -> PanelDropCallback {
    Arc::new(move |dragged_region_id: u64, target_region_id: u64, zone| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (dragged_region_id, target_region_id, drop_zone_str(zone))) {
                err.print(py);
            }
        });
    })
}

/// Plain (`Send`-safe) layout parameters, turned into a real `taffy::Style` only once we're
/// running on the render thread (in `attach`). `taffy::Style` itself contains a raw pointer
/// (for `calc()` expressions, which nothing here uses) and so isn't `Send` — it can't be
/// carried inside the `Command::MutateWidgetTree` closure directly.
#[derive(Clone, Copy)]
struct StyleParams {
    direction: FlexDirection,
    gap: f32,
    padding: f32,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    fill: bool,
    /// `None` leaves taffy's default (`Stretch`) — fine for most containers, but a fixed-height
    /// bar (e.g. `Panel`'s title bar) needs `Center` instead, or a cross-axis-stretched label
    /// gets squeezed to the bar's padded interior height and cosmic-text has no room to lay out
    /// even a single line.
    align_items: Option<AlignItems>,
    /// `Some((x, y))` makes this node `Position::Absolute`, positioned at `(x, y)` relative to
    /// its containing block (in practice always the tree root — see `Window::add_floating_panel`
    /// — since `Position::Relative` is taffy's default for every other node here, the root
    /// already qualifies as "closest positioned ancestor" without needing to say so explicitly)
    /// instead of taking part in normal flex flow. `None` (everything else) is ordinary flow.
    absolute: Option<(f32, f32)>,
}

impl StyleParams {
    fn leaf(flex_grow: f32, width: Option<f32>, height: Option<f32>) -> Self {
        Self {
            direction: FlexDirection::Column,
            gap: 0.0,
            padding: 0.0,
            flex_grow,
            width,
            height,
            fill: false,
            align_items: None,
            absolute: None,
        }
    }

    fn to_style(self) -> Style {
        let size = if self.fill {
            Size { width: Dimension::percent(1.0), height: Dimension::percent(1.0) }
        } else {
            Size {
                width: self.width.map(Dimension::length).unwrap_or(Dimension::auto()),
                height: self.height.map(Dimension::length).unwrap_or(Dimension::auto()),
            }
        };
        let lp_padding = LengthPercentage::length(self.padding);
        let lp_gap = LengthPercentage::length(self.gap);
        let (position, inset) = match self.absolute {
            Some((x, y)) => (
                Position::Absolute,
                Rect {
                    left: LengthPercentageAuto::length(x),
                    top: LengthPercentageAuto::length(y),
                    right: LengthPercentageAuto::auto(),
                    bottom: LengthPercentageAuto::auto(),
                },
            ),
            None => (
                Position::Relative,
                Rect {
                    left: LengthPercentageAuto::auto(),
                    top: LengthPercentageAuto::auto(),
                    right: LengthPercentageAuto::auto(),
                    bottom: LengthPercentageAuto::auto(),
                },
            ),
        };
        Style {
            flex_direction: self.direction,
            flex_grow: self.flex_grow,
            gap: Size { width: lp_gap, height: lp_gap },
            padding: Rect { left: lp_padding, right: lp_padding, top: lp_padding, bottom: lp_padding },
            size,
            align_items: self.align_items,
            position,
            inset,
            ..Default::default()
        }
    }
}

/// A Python-side widget, walked into this once by `Window.set_content` and sent to the render
/// thread as a single `Command::MutateWidgetTree` closure that rebuilds the whole
/// `WidgetTree` from it. `id_cell`/`sender_cell` are populated by that closure as it runs, so
/// by the time `set_content` returns (it waits — see `fastgui-py`'s `Window.set_content`)
/// every widget in the tree has a working route back to the render thread for later mutator
/// calls like `label.set_text(...)`.
/// Set only by `Splitter::describe` — tells `attach` to insert a `WidgetKind::Splitter` bar
/// node between `children[0]` and `children[1]` (which must be exactly a 2-element `first`,
/// `second` pair) instead of attaching them as plain siblings. Kept out of `StyleParams`
/// because a splitter bar's `first`/`second` widget IDs don't exist until its two panes have
/// themselves been attached — see `attach`'s special case below.
struct SplitterBarSpec {
    direction: SplitDirection,
    ratio: f32,
    bar_color: Color,
    thickness: f32,
}

/// Set only by `Tabs::describe` — tells `attach` to insert a `WidgetKind::TabBar` node before
/// `children` (one content wrapper per tab) instead of attaching them as plain siblings. Kept
/// out of `StyleParams` for the same reason as `SplitterBarSpec`: the bar's `content_ids` don't
/// exist until its children have themselves been attached.
struct TabBarSpec {
    titles: Vec<String>,
    active: usize,
    font_size: f32,
    text_color: Color,
    active_color: Color,
    inactive_color: Color,
    height: f32,
    on_select: Option<TabSelectCallback>,
    panel_ids: Vec<u64>,
    on_drop: Vec<Option<PanelDropCallback>>,
}

pub(crate) fn described_viewport(viewport_id: u64, frames: fastgui_core::FrameSlot<fastgui_core::CpuFrame>) -> DescribedWidget {
    DescribedWidget {
        style: StyleParams::leaf(1.0, None, None),
        kind: WidgetKind::Viewport { viewport_id, frames },
        id_cell: Arc::new(Mutex::new(None)),
        sender_cell: Arc::new(Mutex::new(None)),
        children: Vec::new(),
        splitter_bar: None,
        tab_bar: None,
    }
}

pub(crate) struct DescribedWidget {
    style: StyleParams,
    kind: WidgetKind,
    id_cell: IdCell,
    sender_cell: SenderCell,
    children: Vec<DescribedWidget>,
    splitter_bar: Option<SplitterBarSpec>,
    tab_bar: Option<TabBarSpec>,
}

impl DescribedWidget {
    /// `Window.set_content(widget)` calls this on the outermost widget so it always fills the
    /// window, regardless of whatever size that widget's own constructor was given — matches
    /// treating "the content root" as the window's content area, not just another box.
    pub(crate) fn force_fill(&mut self) {
        self.style.fill = true;
    }
}

pub(crate) fn describe(obj: &Bound<'_, PyAny>) -> PyResult<DescribedWidget> {
    if let Ok(w) = obj.cast::<Label>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Button>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Slider>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<BoxWidget>() {
        return w.borrow().describe();
    }
    if let Ok(w) = obj.cast::<Splitter>() {
        return w.borrow().describe();
    }
    if let Ok(w) = obj.cast::<Panel>() {
        return w.borrow().describe();
    }
    if let Ok(w) = obj.cast::<Tabs>() {
        return w.borrow().describe();
    }
    // Pure-Python composite widgets (e.g. `DockArea`) expose the real widget they build up to
    // through this attribute instead of being a pyclass themselves — keeps composition logic
    // (splitter-tree bookkeeping, region math) in Python without needing a matching Rust type
    // per composite.
    if let Ok(inner) = obj.getattr("_fastgui_widget") {
        return describe(&inner);
    }
    if let Ok(w) = obj.cast::<crate::Viewport>() {
        return Ok(w.borrow().describe_widget());
    }
    Err(PyTypeError::new_err(
        "expected a fastgui widget (Box, Label, Button, Slider, Splitter, Panel, Tabs, Viewport, DockArea, ...)",
    ))
}

/// Wire every `Viewport` in this widget tree to `dispatch` so `submit_frame` can wake the
/// idle event loop without waiting for the render thread to `attach()`.
pub(crate) fn bind_dispatch(obj: &Bound<'_, PyAny>, dispatch: &CommandDispatch) {
    if let Ok(viewport) = obj.cast::<crate::Viewport>() {
        viewport.borrow().bind_dispatch(dispatch.clone());
        return;
    }
    if let Ok(box_widget) = obj.cast::<BoxWidget>() {
        Python::attach(|py| {
            for child in box_widget.borrow().children.bind(py).iter() {
                bind_dispatch(&child, dispatch);
            }
        });
        return;
    }
    if let Ok(splitter) = obj.cast::<Splitter>() {
        Python::attach(|py| {
            bind_dispatch(splitter.borrow().first.bind(py), dispatch);
            bind_dispatch(splitter.borrow().second.bind(py), dispatch);
        });
        return;
    }
    if let Ok(panel) = obj.cast::<Panel>() {
        Python::attach(|py| bind_dispatch(panel.borrow().content.bind(py), dispatch));
        return;
    }
    if let Ok(tabs) = obj.cast::<Tabs>() {
        Python::attach(|py| {
            for item in tabs.borrow().panels.bind(py).iter() {
                bind_dispatch(&item, dispatch);
            }
        });
        return;
    }
    if let Ok(inner) = obj.getattr("_fastgui_widget") {
        bind_dispatch(&inner, dispatch);
    }
}

/// Recursively create `described` (and its children) in `tree`, populating each widget's
/// `id_cell`/`sender_cell` as it goes. Called from inside the `Command::MutateWidgetTree`
/// closure `Window.set_content` sends — i.e. always on the render thread. Returns the created
/// node's own `WidgetId` so `Splitter` handling (below) can wire up `first`/`second` — and so
/// the generic child-attaching branch can backfill any `PanelTitleBar` child's `container_id` to
/// its own freshly-created parent id (needed for floating-panel drag, which repositions the
/// *container*, not the title bar leaf — see `WidgetKind::PanelTitleBar`'s doc comment; every
/// `PanelTitleBar` is always its parent `Panel` container's first child, so this is safe to do
/// unconditionally for any such child, not just ones inside a `Panel` specifically).
pub(crate) fn attach(
    tree: &mut WidgetTree,
    parent: WidgetId,
    described: DescribedWidget,
    sender: &CommandDispatch,
) -> WidgetId {
    let id = tree.new_node(described.style.to_style(), described.kind);
    tree.add_child(parent, id);
    *described.id_cell.lock().unwrap_or_else(|p| p.into_inner()) = Some(id);
    *described.sender_cell.lock().unwrap_or_else(|p| p.into_inner()) = Some(sender.clone());

    if let Some(bar_spec) = described.splitter_bar {
        let mut children = described.children.into_iter();
        let (Some(first_described), Some(second_described)) = (children.next(), children.next()) else {
            // `Splitter::describe` always produces exactly two children; this can't happen
            // outside a bug in this crate, and there's no sane partial-splitter to build.
            return id;
        };
        let first_id = attach(tree, id, first_described, sender);
        let bar_style = StyleParams {
            direction: taffy_direction(bar_spec.direction),
            gap: 0.0,
            padding: 0.0,
            flex_grow: 0.0,
            width: (bar_spec.direction == SplitDirection::Row).then_some(bar_spec.thickness),
            height: (bar_spec.direction == SplitDirection::Column).then_some(bar_spec.thickness),
            fill: false,
            align_items: None, absolute: None,
        };
        let bar_kind = WidgetKind::Splitter {
            direction: bar_spec.direction,
            ratio: bar_spec.ratio,
            bar_color: bar_spec.bar_color,
            first: first_id,
            // Placeholder until `second_id` exists just below; never observed in between since
            // both happen inside the same `MutateWidgetTree` closure invocation.
            second: first_id,
        };
        let bar_id = tree.new_node(bar_style.to_style(), bar_kind);
        tree.add_child(id, bar_id);
        let second_id = attach(tree, id, second_described, sender);
        tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::Splitter { second, .. } = kind {
                *second = second_id;
            }
        });
    } else if let Some(bar_spec) = described.tab_bar {
        // Bar goes first (visually above its content — see `WidgetKind::TabBar`'s doc comment),
        // so it's created and `add_child`ed before any content wrapper, unlike `Splitter`'s bar
        // (which sits *between* its two children and so needs the first one's id already).
        let bar_style = StyleParams {
            direction: FlexDirection::Row,
            gap: 0.0,
            padding: 0.0,
            flex_grow: 0.0,
            width: None,
            height: Some(bar_spec.height),
            fill: false,
            align_items: None, absolute: None,
        };
        let active = bar_spec.active;
        let bar_kind = WidgetKind::TabBar {
            titles: bar_spec.titles,
            active,
            font_size: bar_spec.font_size,
            text_color: bar_spec.text_color,
            active_color: bar_spec.active_color,
            inactive_color: bar_spec.inactive_color,
            content_ids: Vec::new(),
            on_select: bar_spec.on_select,
            panel_ids: bar_spec.panel_ids,
            on_drop: bar_spec.on_drop,
        };
        let bar_id = tree.new_node(bar_style.to_style(), bar_kind);
        tree.add_child(id, bar_id);
        // Only the active tab's content is visible at attach time — everything but the click
        // handler's own `set_display` toggling (`fastgui-render-vk::app::handle_tab_click`)
        // lives here, so both paths agree on what "active" means.
        let content_ids: Vec<WidgetId> = described
            .children
            .into_iter()
            .enumerate()
            .map(|(index, child)| {
                let child_id = attach(tree, id, child, sender);
                tree.set_display(child_id, index == active);
                child_id
            })
            .collect();
        tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::TabBar { content_ids: ids, .. } = kind {
                *ids = content_ids;
            }
        });
    } else {
        for child in described.children {
            let child_id = attach(tree, id, child, sender);
            if let Some(WidgetKind::PanelTitleBar { .. }) = tree.kind(child_id) {
                tree.mutate_kind(child_id, |kind| {
                    if let WidgetKind::PanelTitleBar { container_id, .. } = kind {
                        *container_id = Some(id);
                    }
                });
            }
        }
    }
    id
}

/// Send `mutation` to whichever window `id_cell`/`sender_cell` are attached to, no-op if the
/// widget was never attached (or the window has since closed).
fn mutate(id_cell: &IdCell, sender_cell: &SenderCell, mutation: impl FnOnce(&mut WidgetKind) + Send + 'static) -> PyResult<()> {
    let id = id_cell.lock().unwrap_or_else(|p| p.into_inner()).ok_or_else(|| {
        PyRuntimeError::new_err("this widget hasn't been attached to a window yet (call window.set_content first)")
    })?;
    let sender = sender_cell
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone()
        .ok_or_else(|| PyRuntimeError::new_err("this widget hasn't been attached to a window yet"))?;
    sender
        .send(Command::MutateWidgetTree(Box::new(move |tree| {
            tree.mutate_kind(id, mutation);
        })))
        .map_err(|_| PyRuntimeError::new_err("window has already closed"))
}

/// A static line of text.
#[pyclass]
pub(crate) struct Label {
    id: IdCell,
    sender: SenderCell,
    text: String,
    font_size: f32,
    color: (f32, f32, f32, f32),
}

#[pymethods]
impl Label {
    #[new]
    #[pyo3(signature = (text, font_size=16.0, color=(1.0, 1.0, 1.0, 1.0)))]
    fn new(text: String, font_size: f32, color: (f32, f32, f32, f32)) -> Self {
        Self { id: Arc::new(Mutex::new(None)), sender: Arc::new(Mutex::new(None)), text, font_size, color }
    }

    /// Change the displayed text. Safe to call from any thread, only once this label has been
    /// attached via `window.set_content(...)`.
    fn set_text(&self, text: String) -> PyResult<()> {
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Label { text: current, .. } = kind {
                *current = text;
            }
        })
    }
}

impl Label {
    fn describe(&self) -> DescribedWidget {
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Label { text: self.text.clone(), font_size: self.font_size, color: rgba(self.color) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
        }
    }
}

/// A clickable button with a label.
#[pyclass]
pub(crate) struct Button {
    id: IdCell,
    sender: SenderCell,
    text: String,
    font_size: f32,
    text_color: (f32, f32, f32, f32),
    background: (f32, f32, f32, f32),
    on_click: Option<Py<PyAny>>,
}

#[pymethods]
impl Button {
    #[new]
    #[pyo3(signature = (text, on_click=None, font_size=16.0, text_color=(1.0, 1.0, 1.0, 1.0), background=(0.25, 0.35, 0.85, 1.0)))]
    fn new(
        text: String,
        on_click: Option<Py<PyAny>>,
        font_size: f32,
        text_color: (f32, f32, f32, f32),
        background: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            text,
            font_size,
            text_color,
            background,
            on_click,
        }
    }

    fn set_text(&self, text: String) -> PyResult<()> {
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Button { text: current, .. } = kind {
                *current = text;
            }
        })
    }
}

impl Button {
    fn describe(&self) -> DescribedWidget {
        let on_click = self.on_click.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Button {
                text: self.text.clone(),
                font_size: self.font_size,
                text_color: rgba(self.text_color),
                background: rgba(self.background),
                on_click: on_click.map(wrap_callback0),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
        }
    }
}

/// A horizontally draggable value slider.
#[pyclass]
pub(crate) struct Slider {
    id: IdCell,
    sender: SenderCell,
    value: f32,
    min: f32,
    max: f32,
    track_color: (f32, f32, f32, f32),
    thumb_color: (f32, f32, f32, f32),
    on_change: Option<Py<PyAny>>,
}

#[pymethods]
impl Slider {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=1.0,
        on_change=None,
        track_color=(0.3, 0.3, 0.35, 1.0),
        thumb_color=(0.4, 0.7, 1.0, 1.0),
    ))]
    fn new(
        value: f32,
        min: f32,
        max: f32,
        on_change: Option<Py<PyAny>>,
        track_color: (f32, f32, f32, f32),
        thumb_color: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            value,
            min,
            max,
            track_color,
            thumb_color,
            on_change,
        }
    }

    /// Set the slider's value programmatically (as opposed to the user dragging it). Does
    /// *not* invoke `on_change` — that callback is for user-driven changes only.
    fn set_value(&self, value: f32) -> PyResult<()> {
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Slider { value: current, min, max, .. } = kind {
                *current = value.clamp(*min, *max);
            }
        })
    }
}

impl Slider {
    fn describe(&self) -> DescribedWidget {
        let on_change = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, Some(24.0)),
            kind: WidgetKind::Slider {
                value: self.value,
                min: self.min,
                max: self.max,
                track_color: rgba(self.track_color),
                thumb_color: rgba(self.thumb_color),
                on_change: on_change.map(wrap_callback1),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
        }
    }
}

/// A layout container: lays its children out in a row or column via flexbox (see `taffy`).
/// `Box` in the Python API, `BoxWidget` here since `Box` is a reserved word in Rust.
#[pyclass(name = "Box")]
pub(crate) struct BoxWidget {
    id: IdCell,
    sender: SenderCell,
    direction: FlexDirection,
    gap: f32,
    padding: f32,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    background: (f32, f32, f32, f32),
    children: Py<PyList>,
}

#[pymethods]
impl BoxWidget {
    #[new]
    #[pyo3(signature = (
        children,
        direction="column",
        gap=0.0,
        padding=0.0,
        flex_grow=0.0,
        width=None,
        height=None,
        background=(0.0, 0.0, 0.0, 0.0),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        children: Py<PyList>,
        direction: &str,
        gap: f32,
        padding: f32,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        background: (f32, f32, f32, f32),
    ) -> PyResult<Self> {
        let direction = match direction {
            "row" => FlexDirection::Row,
            "column" => FlexDirection::Column,
            other => {
                return Err(PyValueError::new_err(format!(
                    "direction must be \"row\" or \"column\", got {other:?}"
                )))
            }
        };
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            direction,
            gap,
            padding,
            flex_grow,
            width,
            height,
            background,
            children,
        })
    }
}

impl BoxWidget {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let style =
            StyleParams { direction: self.direction, gap: self.gap, padding: self.padding, flex_grow: self.flex_grow, width: self.width, height: self.height, fill: false, align_items: None, absolute: None };

        let children = Python::attach(|py| -> PyResult<Vec<DescribedWidget>> {
            self.children.bind(py).iter().map(|child| describe(&child)).collect()
        })?;

        Ok(DescribedWidget {
            style,
            kind: WidgetKind::Container {
                background: if self.background.3 > 0.0 { rgba(self.background) } else { transparent() },
                region_id: None,
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children,
            splitter_bar: None,
            tab_bar: None,
        })
    }
}

fn split_direction(direction: &str) -> PyResult<SplitDirection> {
    match direction {
        "row" => Ok(SplitDirection::Row),
        "column" => Ok(SplitDirection::Column),
        other => Err(PyValueError::new_err(format!("direction must be \"row\" or \"column\", got {other:?}"))),
    }
}

fn taffy_direction(direction: SplitDirection) -> FlexDirection {
    match direction {
        SplitDirection::Row => FlexDirection::Row,
        SplitDirection::Column => FlexDirection::Column,
    }
}

/// A draggable divider between `first` and `second`, splitting the space between them along
/// `direction`. `ratio` is `first`'s initial share of the space (`0.0`..`1.0`); dragging the bar
/// updates it live (see `fastgui-render-vk::app::update_dragged_splitter`). The foundation both
/// a standalone split view and `DockArea` are built on (`python/fastgui/__init__.py`).
#[pyclass]
pub(crate) struct Splitter {
    id: IdCell,
    sender: SenderCell,
    first: Py<PyAny>,
    second: Py<PyAny>,
    direction: SplitDirection,
    ratio: f32,
    bar_color: (f32, f32, f32, f32),
    thickness: f32,
}

#[pymethods]
impl Splitter {
    #[new]
    #[pyo3(signature = (first, second, direction="row", ratio=0.5, bar_color=(0.2, 0.21, 0.24, 1.0), thickness=6.0))]
    fn new(
        first: Py<PyAny>,
        second: Py<PyAny>,
        direction: &str,
        ratio: f32,
        bar_color: (f32, f32, f32, f32),
        thickness: f32,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(PyValueError::new_err("ratio must be between 0.0 and 1.0"));
        }
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            first,
            second,
            direction: split_direction(direction)?,
            ratio,
            bar_color,
            thickness,
        })
    }
}

impl Splitter {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let (mut first_described, mut second_described) = Python::attach(|py| -> PyResult<_> {
            Ok((describe(self.first.bind(py))?, describe(self.second.bind(py))?))
        })?;
        // `flex_grow` here is what actually redistributes space between the two panes — see
        // `update_dragged_splitter`'s matching `set_flex_grow` calls when the bar is dragged.
        const GROW_SCALE: f32 = 1000.0;
        first_described.style.flex_grow = self.ratio * GROW_SCALE;
        second_described.style.flex_grow = (1.0 - self.ratio) * GROW_SCALE;

        Ok(DescribedWidget {
            style: StyleParams {
                direction: taffy_direction(self.direction),
                gap: 0.0,
                padding: 0.0,
                flex_grow: 1.0,
                width: None,
                height: None,
                fill: false,
                align_items: None, absolute: None,
            },
            kind: WidgetKind::Container { background: transparent(), region_id: None },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: vec![first_described, second_described],
            splitter_bar: Some(SplitterBarSpec {
                direction: self.direction,
                ratio: self.ratio,
                bar_color: rgba(self.bar_color),
                thickness: self.thickness,
            }),
            tab_bar: None,
        })
    }
}

/// A titled container: a fixed-height title bar above `content`, which fills the remaining
/// space. The building block `DockArea` composes into a split-tree of regions (see
/// `python/fastgui/__init__.py`); also useful standalone.
#[pyclass]
pub(crate) struct Panel {
    id: IdCell,
    sender: SenderCell,
    /// This `Panel`'s drag-and-drop identity — see `NEXT_REGION_ID`'s doc comment. Distinct from
    /// `id` above (that's the *widget tree node's* id, assigned fresh on every `describe`/
    /// `attach`; `region_id` is assigned once, at construction, and stays stable across rebuilds).
    region_id: u64,
    rearrange_handler: RearrangeHandlerCell,
    /// Set by `Window.add_floating_panel` — makes `describe()` build this panel's title bar
    /// with `floating: true` (moves-on-drag instead of the rearrange/drop-zone machinery). A
    /// plain `AtomicBool`, not GIL-gated, since it's a single flag no Python callback ever
    /// touches.
    floating: Arc<std::sync::atomic::AtomicBool>,
    title: String,
    content: Py<PyAny>,
    title_font_size: f32,
    title_color: (f32, f32, f32, f32),
    title_background: (f32, f32, f32, f32),
    background: (f32, f32, f32, f32),
    title_height: f32,
}

#[pymethods]
impl Panel {
    #[new]
    #[pyo3(signature = (
        title,
        content,
        title_font_size=14.0,
        title_color=(0.92, 0.93, 0.95, 1.0),
        title_background=(0.16, 0.17, 0.20, 1.0),
        background=(0.12, 0.13, 0.15, 1.0),
        title_height=28.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        title: String,
        content: Py<PyAny>,
        title_font_size: f32,
        title_color: (f32, f32, f32, f32),
        title_background: (f32, f32, f32, f32),
        background: (f32, f32, f32, f32),
        title_height: f32,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            region_id: next_region_id(),
            rearrange_handler: Arc::new(Mutex::new(None)),
            floating: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            title,
            content,
            title_font_size,
            title_color,
            title_background,
            background,
            title_height,
        }
    }

    /// Stable drag-and-drop identity for this panel — see `NEXT_REGION_ID`'s doc comment.
    #[getter]
    fn id(&self) -> u64 {
        self.region_id
    }

    /// Claims this panel's title-bar drag: `handler(dragged_region_id, target_region_id, zone)`
    /// fires when the user drags this panel's title bar and releases it over another region.
    /// Called by `DockArea.add_panel` (`python/fastgui/__init__.py`) — not meant to be called
    /// directly, hence no default making a bare `Panel()` draggable-but-inert by itself.
    fn set_rearrange_handler(&self, handler: Py<PyAny>) {
        *self.rearrange_handler.lock().unwrap_or_else(|p| p.into_inner()) = Some(handler);
    }
}

impl Panel {
    /// Used by `Tabs::describe` to pull just the title text and content widget out of a `Panel`
    /// used as a tab — a `Tabs` draws one combined header strip already serving as "the title",
    /// so it doesn't attach each member `Panel`'s own title bar (that would show it twice).
    pub(crate) fn title_and_content(&self, py: Python<'_>) -> (String, Py<PyAny>) {
        (self.title.clone(), self.content.clone_ref(py))
    }

    /// Used by `Tabs::describe` to give each tab segment its own drag-out identity/callback (see
    /// `WidgetKind::TabBar`'s doc comment) — the same `region_id`/`rearrange_handler` a standalone
    /// `Panel`'s own title bar drag uses, just read off a `Panel` that's currently tabbed instead
    /// of built as its own draggable title bar.
    pub(crate) fn drag_identity(&self, py: Python<'_>) -> (u64, Option<Py<PyAny>>) {
        let handler = self.rearrange_handler.lock().unwrap_or_else(|p| p.into_inner()).as_ref().map(|cb| cb.clone_ref(py));
        (self.region_id, handler)
    }

    /// Called once by `Window.add_floating_panel` (`fastgui-py::lib`), before its first
    /// `describe_floating` — makes every future `describe()`/`describe_floating()` build this
    /// panel's title bar with `floating: true`. There's no way back to plain-docked in this
    /// pass (matches `DockArea` not supporting drag-*out*-of-floating either — see its
    /// docstring's known gaps).
    pub(crate) fn set_floating(&self) {
        self.floating.store(true, Ordering::Relaxed);
    }

    /// Like `describe()`, but absolutely positioned at `(x, y)` with explicit `(width, height)`
    /// instead of filling a flex slot — what `Window.add_floating_panel` actually attaches.
    /// Reuses `describe()` for the title-bar/content structure entirely, just overrides the
    /// outer container's own layout — see `StyleParams::absolute`'s doc comment for why that's
    /// enough to make taffy treat this node as free-floating.
    pub(crate) fn describe_floating(&self, x: f32, y: f32, width: f32, height: f32) -> PyResult<DescribedWidget> {
        let mut described = self.describe()?;
        described.style.flex_grow = 0.0;
        described.style.width = Some(width);
        described.style.height = Some(height);
        described.style.fill = false;
        described.style.absolute = Some((x, y));
        Ok(described)
    }

    fn describe(&self) -> PyResult<DescribedWidget> {
        let mut content_described = Python::attach(|py| describe(self.content.bind(py)))?;
        // Fill whatever vertical space the title bar (fixed height, below) doesn't take.
        content_described.style.flex_grow = 1.0;

        let on_drop = self
            .rearrange_handler
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|cb| Python::attach(|py| cb.clone_ref(py)));

        let title_bar = DescribedWidget {
            style: StyleParams { direction: FlexDirection::Row, gap: 0.0, padding: 0.0, flex_grow: 0.0, width: None, height: Some(self.title_height), fill: false, align_items: None, absolute: None },
            kind: WidgetKind::PanelTitleBar {
                panel_id: self.region_id,
                title: self.title.clone(),
                font_size: self.title_font_size,
                text_color: rgba(self.title_color),
                background: rgba(self.title_background),
                on_drop: on_drop.map(wrap_panel_drop_callback),
                floating: self.floating.load(Ordering::Relaxed),
                // Backfilled to this title bar's actual parent by `attach`'s generic
                // child-attaching branch — not known yet at describe-time.
                container_id: None,
            },
            id_cell: Arc::new(Mutex::new(None)),
            sender_cell: Arc::new(Mutex::new(None)),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
        };

        Ok(DescribedWidget {
            style: StyleParams {
                direction: FlexDirection::Column,
                gap: 0.0,
                padding: 0.0,
                flex_grow: 1.0,
                width: None,
                height: None,
                fill: false,
                align_items: None, absolute: None,
            },
            kind: WidgetKind::Container { background: rgba(self.background), region_id: Some(self.region_id) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: vec![title_bar, content_described],
            splitter_bar: None,
            tab_bar: None,
        })
    }
}

/// Multiple `Panel`s sharing one region: one combined header strip (each member's `title`, not
/// its own title bar — see `Panel::title_and_content`) with only the active one's content
/// visible. Click a header segment (`fastgui-render-vk::app::handle_tab_click`) to switch.
#[pyclass]
pub(crate) struct Tabs {
    id: IdCell,
    sender: SenderCell,
    /// This `Tabs` group's drag-and-drop identity — see `NEXT_REGION_ID`'s doc comment. A `Tabs`
    /// is a valid drop *target* (drop a dragged `Panel` on it to split that region), but — unlike
    /// `Panel` — isn't itself draggable in this pass: its member panels don't have their own
    /// grabbable title bars while tabbed (see `WidgetKind::TabBar`'s doc comment).
    region_id: u64,
    panels: Py<PyList>,
    active: usize,
    font_size: f32,
    text_color: (f32, f32, f32, f32),
    active_color: (f32, f32, f32, f32),
    inactive_color: (f32, f32, f32, f32),
    height: f32,
    on_select: Option<Py<PyAny>>,
}

#[pymethods]
impl Tabs {
    #[new]
    #[pyo3(signature = (
        panels,
        active=0,
        font_size=14.0,
        text_color=(0.92, 0.93, 0.95, 1.0),
        active_color=(0.20, 0.22, 0.26, 1.0),
        inactive_color=(0.14, 0.15, 0.18, 1.0),
        height=28.0,
        on_select=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        panels: Py<PyList>,
        active: usize,
        font_size: f32,
        text_color: (f32, f32, f32, f32),
        active_color: (f32, f32, f32, f32),
        inactive_color: (f32, f32, f32, f32),
        height: f32,
        on_select: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            region_id: next_region_id(),
            panels,
            active,
            font_size,
            text_color,
            active_color,
            inactive_color,
            height,
            on_select,
        }
    }

    /// Stable drag-and-drop identity for this tab group — see `NEXT_REGION_ID`'s doc comment.
    #[getter]
    fn id(&self) -> u64 {
        self.region_id
    }

    /// This group's member panels, in tab order — read by `DockArea` (`python/fastgui/
    /// __init__.py`) to tear a `Tabs` group back apart when one of its tabs is dragged out.
    #[getter]
    fn panels(&self, py: Python<'_>) -> Py<PyList> {
        self.panels.clone_ref(py)
    }

    /// This group's currently-active tab index — read by `DockArea` so ungrouping down to one
    /// remaining member, or dragging out a non-active tab, doesn't silently reset the selection.
    #[getter]
    fn active(&self) -> usize {
        self.active
    }
}

impl Tabs {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let (titles, panel_ids, on_drop, mut contents) = Python::attach(|py| -> PyResult<(
            Vec<String>,
            Vec<u64>,
            Vec<Option<PanelDropCallback>>,
            Vec<DescribedWidget>,
        )> {
            let mut titles = Vec::new();
            let mut panel_ids = Vec::new();
            let mut on_drop = Vec::new();
            let mut contents = Vec::new();
            for item in self.panels.bind(py).iter() {
                let panel = item
                    .cast::<Panel>()
                    .map_err(|_| PyTypeError::new_err("Tabs(...) expects a list of Panel objects"))?;
                let panel = panel.borrow();
                let (title, content) = panel.title_and_content(py);
                let (region_id, handler) = panel.drag_identity(py);
                titles.push(title);
                panel_ids.push(region_id);
                on_drop.push(handler.map(wrap_panel_drop_callback));
                contents.push(describe(content.bind(py))?);
            }
            Ok((titles, panel_ids, on_drop, contents))
        })?;
        if titles.is_empty() {
            return Err(PyValueError::new_err("Tabs(...) needs at least one panel"));
        }
        let active = self.active.min(titles.len() - 1);
        for content in &mut contents {
            // Fill whatever space the tab bar (fixed height, above) doesn't take — same
            // reasoning as `Panel`'s content pane.
            content.style.flex_grow = 1.0;
        }

        let on_select = self.on_select.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));

        Ok(DescribedWidget {
            style: StyleParams {
                direction: FlexDirection::Column,
                gap: 0.0,
                padding: 0.0,
                flex_grow: 1.0,
                width: None,
                height: None,
                fill: false,
                align_items: None, absolute: None,
            },
            kind: WidgetKind::Container { background: transparent(), region_id: Some(self.region_id) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: contents,
            splitter_bar: None,
            tab_bar: Some(TabBarSpec {
                titles,
                active,
                font_size: self.font_size,
                text_color: rgba(self.text_color),
                active_color: rgba(self.active_color),
                inactive_color: rgba(self.inactive_color),
                height: self.height,
                on_select: on_select.map(wrap_callback_usize),
                panel_ids,
                on_drop,
            }),
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Label>()?;
    m.add_class::<Button>()?;
    m.add_class::<Slider>()?;
    m.add_class::<BoxWidget>()?;
    m.add_class::<Splitter>()?;
    m.add_class::<Panel>()?;
    m.add_class::<Tabs>()?;
    Ok(())
}
