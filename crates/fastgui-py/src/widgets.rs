use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use fastgui_core::taffy::prelude::*;
use fastgui_core::taffy::style::LengthPercentage;
use fastgui_core::widget::{
    clamp_range, round_to_decimals, Accel, BoolCallback, ChangeCallback, ClickCallback, Color, IndexCallback,
    PanelCloseCallback, PanelDropCallback, PathCallback, PointCallback, PopupAnchor, PopupSide, SplitDirection,
    TabSelectCallback, TableColumn, TableData, TextCallback, TreeData, TreeNodeData, WidgetId, WidgetKind,
    WidgetTree,
};
use fastgui_core::text_edit::TextEdit;
use fastgui_core::{CpuFrame, FrameSlot, PixelFormat, Readback, MAX_CPU_FRAME_EXTENT};
use crate::theme::{FontSize, Spacing};
use crate::backend::{Command, CommandDispatch};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PySequence};

type IdCell = Arc<Mutex<Option<WidgetId>>>;
type SenderCell = Arc<Mutex<Option<CommandDispatch>>>;
/// Interior-mutable cell a `Panel`/`Tabs` region's rearrange handler lives in — `None` until a
/// `DockArea` claims it (`Panel::set_rearrange_handler`), so a standalone `Panel` not inside a
/// `DockArea` just has a no-op drag (see `WidgetKind::PanelTitleBar`'s doc comment).
type RearrangeHandlerCell = Arc<Mutex<Option<Py<PyAny>>>>;
type CloseHandlerCell = Arc<Mutex<Option<Py<PyAny>>>>;

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

/// `context_menu=` value: a `Popup`, an object with `as_popup()`, or a `(x, y)` callable.
fn wrap_context_menu(callback: Py<PyAny>, sender_cell: SenderCell) -> PointCallback {
    Arc::new(move |x: f32, y: f32| {
        let Some(sender) = sender_cell
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
        else {
            return;
        };
        Python::attach(|py| {
            let obj = callback.bind(py);
            if let Ok(popup) = obj.cast::<Popup>() {
                let _ = popup.borrow().open_in(&sender, PopupAnchor::Point(x, y));
                return;
            }
            if let Ok(as_popup) = obj.call_method0("as_popup") {
                if let Ok(popup) = as_popup.cast::<Popup>() {
                    let _ = popup.borrow().open_in(&sender, PopupAnchor::Point(x, y));
                    return;
                }
            }
            if let Err(err) = obj.call1((x, y)) {
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

#[allow(dead_code)]
fn wrap_callback_bool(callback: Py<PyAny>) -> BoolCallback {
    Arc::new(move |value: bool| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (value,)) {
                err.print(py);
            }
        });
    })
}

/// `Image` layers share the renderer's layer map with `Viewport`s (fastgui-app's viewport/image walk),
/// so they must draw ids from the same counter — separate counters both started at 1 and the
/// first `Image` showed the first `Viewport`'s frames.
fn next_image_id() -> u64 {
    crate::next_viewport_id()
}

/// `value` clamped to the range between `a` and `b` in either order. `f32::clamp` panics when
/// `min > max` or a bound is NaN — on the render thread that took the whole app down for a
/// `ProgressBar(min=1, max=0).set_value(...)`.
/// Auto-assigned radio group ids set this high bit so they never collide with user-supplied ones.
const RADIO_AUTO_GROUP_BIT: u64 = 1 << 63;

static NEXT_RADIO_GROUP: AtomicU64 = AtomicU64::new(1);
fn next_radio_group() -> u64 {
    NEXT_RADIO_GROUP.fetch_add(1, Ordering::Relaxed) | RADIO_AUTO_GROUP_BIT
}

fn parse_radio_group(group: Option<u64>) -> PyResult<u64> {
    match group {
        None => Ok(next_radio_group()),
        Some(g) if g >= RADIO_AUTO_GROUP_BIT => Err(PyValueError::new_err(
            "Radio.group must be < 2**63 (automatic ids use that range)",
        )),
        Some(g) => Ok(g),
    }
}

/// Mirrors for radios that share a group — used by `set_selected` before the widgets attach.
static RADIO_GROUP_MIRRORS: LazyLock<Mutex<HashMap<u64, Vec<Readback<bool>>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn register_radio_mirror(group: u64, mirror: &Readback<bool>) {
    RADIO_GROUP_MIRRORS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .entry(group)
        .or_default()
        .push(mirror.clone());
}

fn sync_radio_mirrors(group: u64, chosen: &Readback<bool>, selected: bool) {
    let mirrors = RADIO_GROUP_MIRRORS.lock().unwrap_or_else(|p| p.into_inner());
    if selected {
        if let Some(peers) = mirrors.get(&group) {
            for mirror in peers {
                mirror.set(false);
            }
        }
    }
    drop(mirrors);
    chosen.set(selected);
}

fn require_finite(name: &str, value: f32) -> PyResult<f32> {
    if value.is_nan() {
        Err(PyValueError::new_err(format!("{name} must be a finite number (got NaN)")))
    } else {
        Ok(value)
    }
}

fn require_positive_step(name: &str, value: f32) -> PyResult<f32> {
    let value = require_finite(name, value)?;
    if value <= 0.0 {
        Err(PyValueError::new_err(format!("{name} must be > 0")))
    } else {
        Ok(value)
    }
}

fn wrap_callback_text(callback: Py<PyAny>) -> TextCallback {
    Arc::new(move |text: String| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (text,)) {
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
        DropZone::Float => "float",
    }
}

fn wrap_panel_drop_callback(callback: Py<PyAny>) -> PanelDropCallback {
    Arc::new(move |dragged_region_id, target_region_id, zone, float_rect| {
        Python::attach(|py| {
            let zone_s = drop_zone_str(zone);
            let result = if let Some((x, y, w, h)) = float_rect {
                callback.call1(py, (dragged_region_id, target_region_id, zone_s, x, y, w, h))
            } else {
                callback.call1(py, (dragged_region_id, target_region_id, zone_s))
            };
            if let Err(err) = result {
                err.print(py);
            }
        });
    })
}

fn wrap_panel_close_callback(callback: Py<PyAny>) -> PanelCloseCallback {
    Arc::new(move |panel_id: u64| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (panel_id,)) {
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
    /// Let children flow onto further rows/columns when they don't fit (`Box(wrap=True)`).
    wrap: bool,
    /// `Some(n)` → CSS Grid with `n` equal `1fr` columns (auto-flow rows). `None` → flexbox.
    grid_columns: Option<u16>,
    /// `false` → `Display::None` (stacked/collapsible pages; same path as inactive tabs).
    visible: bool,
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
            wrap: false,
            grid_columns: None,
            visible: true,
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
        let mut style = Style {
            flex_direction: self.direction,
            flex_grow: self.flex_grow,
            gap: Size { width: lp_gap, height: lp_gap },
            padding: Rect { left: lp_padding, right: lp_padding, top: lp_padding, bottom: lp_padding },
            size,
            align_items: self.align_items,
            flex_wrap: if self.wrap { FlexWrap::Wrap } else { FlexWrap::NoWrap },
            position,
            inset,
            ..Default::default()
        };
        if let Some(columns) = self.grid_columns.filter(|&n| n > 0) {
            style.display = Display::Grid;
            style.grid_template_columns = evenly_sized_tracks(columns);
            style.grid_auto_rows = vec![TrackSizingFunction::AUTO];
        }
        if !self.visible {
            style.display = Display::None;
        }
        style
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
    on_close: Vec<Option<PanelCloseCallback>>,
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
        tooltip: None,
        context_menu: None,
        accelerators: Vec::new(),
        hover_action: None,
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
    tooltip: Option<String>,
    context_menu: Option<PointCallback>,
    accelerators: Vec<(String, ClickCallback)>,
    /// Run after the cursor rests on the widget (`WidgetTree::set_hover_action`).
    hover_action: Option<ClickCallback>,
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
    if let Ok(w) = obj.cast::<TextInput>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<TextArea>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<ScrollArea>() {
        return w.borrow().describe();
    }
    if let Ok(w) = obj.cast::<ListView>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Table>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<TreeView>() {
        return Ok(w.borrow().describe());
    }
    if obj.cast::<Popup>().is_ok() {
        return Err(PyTypeError::new_err("a Popup isn't placed in the layout; open it with popup.show(anchor)"));
    }
    if let Ok(w) = obj.cast::<Checkbox>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Radio>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Toggle>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<SpinBox>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<NumericScrub>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<ProgressBar>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<ComboBox>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Image>() {
        return Ok(w.borrow().describe());
    }
    if let Ok(w) = obj.cast::<Grid>() {
        return w.borrow().describe();
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
        "expected a fastgui widget (Box, Grid, Label, Button, Slider, TextInput, TextArea, ScrollArea, ListView, Table, TreeView, Checkbox, Radio, Toggle, SpinBox, NumericScrub, ProgressBar, ComboBox, Image, Splitter, Panel, Tabs, Viewport, DockArea, ...)",
    ))
}

/// Wire every `Viewport` in this widget tree to `dispatch` so `submit_frame` can wake the
/// idle event loop without waiting for the render thread to `attach()`.
pub(crate) fn bind_dispatch(obj: &Bound<'_, PyAny>, dispatch: &CommandDispatch) {
    if let Ok(viewport) = obj.cast::<crate::Viewport>() {
        viewport.borrow().bind_dispatch(dispatch.clone());
        return;
    }
    if let Ok(image) = obj.cast::<Image>() {
        image.borrow().bind_dispatch(dispatch.clone());
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
    if let Ok(grid) = obj.cast::<Grid>() {
        Python::attach(|py| {
            for child in grid.borrow().children.bind(py).iter() {
                bind_dispatch(&child, dispatch);
            }
        });
        return;
    }
    if let Ok(scroll) = obj.cast::<ScrollArea>() {
        Python::attach(|py| bind_dispatch(scroll.borrow().content.bind(py), dispatch));
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
    let DescribedWidget {
        style,
        kind,
        id_cell,
        sender_cell,
        children,
        splitter_bar,
        tab_bar,
        tooltip,
        context_menu,
        accelerators,
        hover_action,
    } = described;
    let id = tree.new_node(style.to_style(), kind);
    tree.add_child(parent, id);
    *id_cell.lock().unwrap_or_else(|p| p.into_inner()) = Some(id);
    *sender_cell.lock().unwrap_or_else(|p| p.into_inner()) = Some(sender.clone());
    if let Some(text) = tooltip {
        tree.set_tooltip(id, Some(text));
    }
    if let Some(callback) = context_menu {
        tree.set_context_menu(id, Some(callback));
    }
    if let Some(callback) = hover_action {
        tree.set_hover_action(id, Some(callback));
    }
    for (shortcut, callback) in accelerators {
        // Unparseable shortcuts are rejected when the Box is constructed (`parse_accelerators`).
        if let Some(accel) = Accel::parse(&shortcut) {
            tree.register_accelerator_for(id, accel, callback);
        }
    }

    if let Some(bar_spec) = splitter_bar {
        let mut children = children.into_iter();
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
            align_items: None, absolute: None, wrap: false, grid_columns: None, visible: true,
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
        // Both panes split the whole span by `ratio` (see `WidgetTree::set_split_pane`), not
        // just whatever their content leaves over.
        tree.set_split_pane(first_id);
        tree.set_split_pane(second_id);
        tree.mutate_kind(bar_id, |kind| {
            if let WidgetKind::Splitter { second, .. } = kind {
                *second = second_id;
            }
        });
    } else if let Some(bar_spec) = tab_bar {
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
            align_items: None, absolute: None, wrap: false, grid_columns: None, visible: true,
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
            on_close: bar_spec.on_close,
        };
        let bar_id = tree.new_node(bar_style.to_style(), bar_kind);
        tree.add_child(id, bar_id);
        // Only the active tab's content is visible at attach time — everything but the click
        // handler's own `set_display` toggling (`fastgui-app::app::handle_tab_click`)
        // lives here, so both paths agree on what "active" means.
        let content_ids: Vec<WidgetId> = children
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
        for child in children {
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
    if let Some(WidgetKind::ScrollArea { .. }) = tree.kind(id) {
        tree.set_scroll_container(id);
    }
    id
}

/// Send a whole-tree `mutation` through `sender` to the window it belongs to (the main window,
/// or the floating panel's own window).
fn send_tree_mutation(
    sender: &CommandDispatch,
    mutation: impl FnOnce(&mut WidgetTree) + Send + 'static,
) -> PyResult<()> {
    let command = match sender.floating_region {
        Some(region_id) => Command::MutateFloatingTree { region_id, mutation: Box::new(mutation) },
        None => Command::MutateWidgetTree(Box::new(mutation)),
    };
    sender.send(command).map_err(|_| PyRuntimeError::new_err("window has already closed"))
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
    let command = match sender.floating_region {
        Some(region_id) => Command::MutateFloatingTree {
            region_id,
            mutation: Box::new(move |tree| {
                tree.mutate_kind(id, mutation);
            }),
        },
        None => Command::MutateWidgetTree(Box::new(move |tree| {
            tree.mutate_kind(id, mutation);
        })),
    };
    sender
        .send(command)
        .map_err(|_| PyRuntimeError::new_err("window has already closed"))
}

/// A static line of text.
#[pyclass]
pub(crate) struct Label {
    id: IdCell,
    sender: SenderCell,
    /// Latest text (constructor or `set_text`), so a rebuild (`set_theme`, `set_content`,
    /// dock rearrange) shows what was last set instead of the constructor's text.
    text: Mutex<String>,
    font_size: Option<FontSize>,
    color: Option<(f32, f32, f32, f32)>,
    tooltip: Option<String>,
    context_menu: Option<Py<PyAny>>,
}

#[pymethods]
impl Label {
    #[new]
    #[pyo3(signature = (text, font_size=None, color=None, tooltip=None, context_menu=None))]
    fn new(
        text: String,
        font_size: Option<FontSize>,
        color: Option<(f32, f32, f32, f32)>,
        tooltip: Option<String>,
        context_menu: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            text: Mutex::new(text),
            font_size,
            color,
            tooltip,
            context_menu,
        }
    }

    /// Change the displayed text. Safe to call from any thread, before or after attaching;
    /// the new text survives rebuilds.
    fn set_text(&self, text: String) -> PyResult<()> {
        *self.text.lock().unwrap_or_else(|p| p.into_inner()) = text.clone();
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Label { text: current, .. } = kind {
                *current = text;
            }
        })
    }
}

impl Label {
    fn describe(&self) -> DescribedWidget {
        let context_menu = self.context_menu.as_ref().map(|cb| {
            let cb = Python::attach(|py| cb.clone_ref(py));
            wrap_context_menu(cb, self.sender.clone())
        });
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Label { text: self.text.lock().unwrap_or_else(|p| p.into_inner()).clone(), font_size: self.font_size.unwrap_or(FontSize::Body).resolve(), color: rgba(self.color.unwrap_or(crate::theme::palette().text)) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: self.tooltip.clone(),
            context_menu,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// A clickable button with a label.
#[pyclass]
pub(crate) struct Button {
    id: IdCell,
    sender: SenderCell,
    /// Latest text (constructor or `set_text`); survives rebuilds.
    text: Mutex<String>,
    font_size: Option<FontSize>,
    text_color: Option<(f32, f32, f32, f32)>,
    /// Behind a mutex so `set_background` survives rebuilds.
    background: Mutex<Option<(f32, f32, f32, f32)>>,
    /// Menu-bar title style: compact, no fill until hover / `set_background`.
    flat: bool,
    on_click: Option<Py<PyAny>>,
    tooltip: Option<String>,
    context_menu: Option<Py<PyAny>>,
    on_hover: Option<Py<PyAny>>,
}

#[pymethods]
impl Button {
    #[new]
    #[pyo3(signature = (
        text,
        on_click=None,
        font_size=None,
        text_color=None,
        background=None,
        flat=false,
        tooltip=None,
        context_menu=None,
        on_hover=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        text: String,
        on_click: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        text_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        flat: bool,
        tooltip: Option<String>,
        context_menu: Option<Py<PyAny>>,
        on_hover: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            text: Mutex::new(text),
            font_size,
            text_color,
            background: Mutex::new(background),
            flat,
            on_click,
            tooltip,
            context_menu,
            on_hover,
        }
    }

    fn set_text(&self, text: String) -> PyResult<()> {
        *self.text.lock().unwrap_or_else(|p| p.into_inner()) = text.clone();
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Button { text: current, .. } = kind {
                *current = text;
            }
        })
    }

    /// Update the fill (e.g. menu-title open highlight). `None` clears to transparent.
    #[pyo3(signature = (color=None))]
    fn set_background(&self, color: Option<(f32, f32, f32, f32)>) -> PyResult<()> {
        *self.background.lock().unwrap_or_else(|p| p.into_inner()) = color;
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        let fill = rgba(color.unwrap_or((0.0, 0.0, 0.0, 0.0)));
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Button { background, .. } = kind {
                *background = fill;
            }
        })
    }
}

impl Button {
    fn describe(&self) -> DescribedWidget {
        let on_click = self.on_click.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        let context_menu = self.context_menu.as_ref().map(|cb| {
            let cb = Python::attach(|py| cb.clone_ref(py));
            wrap_context_menu(cb, self.sender.clone())
        });
        let default_bg = if self.flat {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            crate::theme::palette().button
        };
        let default_fg = if self.flat {
            crate::theme::palette().text
        } else {
            crate::theme::palette().button_text
        };
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Button {
                text: self.text.lock().unwrap_or_else(|p| p.into_inner()).clone(),
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(default_fg)),
                background: rgba(self.background.lock().unwrap_or_else(|p| p.into_inner()).unwrap_or(default_bg)),
                flat: self.flat,
                on_click: on_click.map(wrap_callback0),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: self.tooltip.clone(),
            context_menu,
            accelerators: Vec::new(),
            hover_action: self.on_hover.as_ref().map(|cb| wrap_callback0(Python::attach(|py| cb.clone_ref(py)))),
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
    track_color: Option<(f32, f32, f32, f32)>,
    thumb_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
    tooltip: Option<String>,
}

#[pymethods]
impl Slider {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=1.0,
        on_change=None,
        track_color=None,
        thumb_color=None,
        tooltip=None,
    ))]
    fn new(
        value: f32,
        min: f32,
        max: f32,
        on_change: Option<Py<PyAny>>,
        track_color: Option<(f32, f32, f32, f32)>,
        thumb_color: Option<(f32, f32, f32, f32)>,
        tooltip: Option<String>,
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
            tooltip,
        }
    }

    /// Set the slider's value programmatically (as opposed to the user dragging it). Does
    /// *not* invoke `on_change` — that callback is for user-driven changes only.
    fn set_value(&self, value: f32) -> PyResult<()> {
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Slider { value: current, min, max, .. } = kind {
                *current = clamp_range(value, *min, *max);
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
                track_color: rgba(self.track_color.unwrap_or(crate::theme::palette().track)),
                thumb_color: rgba(self.thumb_color.unwrap_or(crate::theme::palette().accent)),
                on_change: on_change.map(wrap_callback1),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: self.tooltip.clone(),
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// A single-line editable text field.
#[pyclass]
pub(crate) struct TextInput {
    id: IdCell,
    sender: SenderCell,
    /// The current text, published by the render thread on every edit (and by `set_text`), so
    /// `.text` never waits on it. Also what `describe` builds from, so the text survives a
    /// rebuild (e.g. a `DockArea` rearrange).
    text: Readback<String>,
    placeholder: String,
    font_size: Option<FontSize>,
    width: Option<f32>,
    flex_grow: f32,
    text_color: Option<(f32, f32, f32, f32)>,
    placeholder_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    selection_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
    on_submit: Option<Py<PyAny>>,
}

#[pymethods]
impl TextInput {
    #[new]
    #[pyo3(signature = (
        text="",
        placeholder="",
        on_change=None,
        on_submit=None,
        font_size=None,
        width=None,
        flex_grow=0.0,
        text_color=None,
        placeholder_color=None,
        background=None,
        selection_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        text: &str,
        placeholder: &str,
        on_change: Option<Py<PyAny>>,
        on_submit: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        width: Option<f32>,
        flex_grow: f32,
        text_color: Option<(f32, f32, f32, f32)>,
        placeholder_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        selection_color: Option<(f32, f32, f32, f32)>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            text: Readback::new(TextEdit::new(text).text().to_owned()),
            placeholder: placeholder.to_owned(),
            font_size,
            width,
            flex_grow,
            text_color,
            placeholder_color,
            background,
            selection_color,
            on_change,
            on_submit,
        }
    }

    /// The field's current text, including edits the user just made. Safe from any thread.
    #[getter]
    fn text(&self) -> String {
        self.text.get()
    }

    /// Replace the text (caret to the end, undo history cleared). Doesn't call `on_change`,
    /// which is for user edits. Works before the field is attached, too.
    fn set_text(&self, text: &str) -> PyResult<()> {
        let text = TextEdit::new(text).text().to_owned();
        self.text.set(text.clone());
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::TextInput { edit, scroll, preedit, .. } = kind {
                edit.set_text(&text);
                *scroll = 0.0;
                *preedit = None;
            }
        })
    }
}

impl TextInput {
    fn describe(&self) -> DescribedWidget {
        let (on_change, on_submit) = Python::attach(|py| {
            (
                self.on_change.as_ref().map(|cb| cb.clone_ref(py)),
                self.on_submit.as_ref().map(|cb| cb.clone_ref(py)),
            )
        });
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, None),
            kind: WidgetKind::TextInput {
                edit: TextEdit::new(&self.text.get()),
                placeholder: self.placeholder.clone(),
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                placeholder_color: rgba(self.placeholder_color.unwrap_or(crate::theme::palette().text_muted)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                selection_color: rgba(self.selection_color.unwrap_or(crate::theme::palette().selection)),
                scroll: 0.0,
                preedit: None,
                on_change: on_change.map(wrap_callback_text),
                on_submit: on_submit.map(wrap_callback_text),
                mirror: Some(self.text.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// A multi-line editable text field (hard newlines; soft wrap not yet).
#[pyclass]
pub(crate) struct TextArea {
    id: IdCell,
    sender: SenderCell,
    text: Readback<String>,
    placeholder: String,
    font_size: Option<FontSize>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
    text_color: Option<(f32, f32, f32, f32)>,
    placeholder_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    selection_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
    on_submit: Option<Py<PyAny>>,
}

#[pymethods]
impl TextArea {
    #[new]
    #[pyo3(signature = (
        text="",
        placeholder="",
        on_change=None,
        on_submit=None,
        font_size=None,
        width=None,
        height=None,
        flex_grow=0.0,
        text_color=None,
        placeholder_color=None,
        background=None,
        selection_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        text: &str,
        placeholder: &str,
        on_change: Option<Py<PyAny>>,
        on_submit: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        text_color: Option<(f32, f32, f32, f32)>,
        placeholder_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        selection_color: Option<(f32, f32, f32, f32)>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            text: Readback::new(TextEdit::new_multiline(text).text().to_owned()),
            placeholder: placeholder.to_owned(),
            font_size,
            width,
            height,
            flex_grow,
            text_color,
            placeholder_color,
            background,
            selection_color,
            on_change,
            on_submit,
        }
    }

    /// The field's current text, including edits the user just made. Safe from any thread.
    #[getter]
    fn text(&self) -> String {
        self.text.get()
    }

    /// Replace the text (caret to the end, undo history cleared). Doesn't call `on_change`.
    fn set_text(&self, text: &str) -> PyResult<()> {
        let text = TextEdit::new_multiline(text).text().to_owned();
        self.text.set(text.clone());
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::TextArea { edit, scroll_x, scroll_y, preedit, .. } = kind {
                edit.set_text(&text);
                *scroll_x = 0.0;
                *scroll_y = 0.0;
                *preedit = None;
            }
        })
    }
}

impl TextArea {
    fn describe(&self) -> DescribedWidget {
        let (on_change, on_submit) = Python::attach(|py| {
            (
                self.on_change.as_ref().map(|cb| cb.clone_ref(py)),
                self.on_submit.as_ref().map(|cb| cb.clone_ref(py)),
            )
        });
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::TextArea {
                edit: TextEdit::new_multiline(&self.text.get()),
                placeholder: self.placeholder.clone(),
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                placeholder_color: rgba(self.placeholder_color.unwrap_or(crate::theme::palette().text_muted)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                selection_color: rgba(self.selection_color.unwrap_or(crate::theme::palette().selection)),
                scroll_x: 0.0,
                scroll_y: 0.0,
                preedit: None,
                on_change: on_change.map(wrap_callback_text),
                on_submit: on_submit.map(wrap_callback_text),
                mirror: Some(self.text.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// A virtualized list of text rows: only the rows in view are drawn, so it handles millions.
#[pyclass]
pub(crate) struct ListView {
    id: IdCell,
    sender: SenderCell,
    /// The current items — what `describe` builds from, kept in sync by `set_items`.
    items: Arc<Mutex<Arc<Vec<String>>>>,
    selected: Readback<Option<usize>>,
    row_height: f32,
    font_size: Option<FontSize>,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    text_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    selection_color: Option<(f32, f32, f32, f32)>,
    on_select: Option<Py<PyAny>>,
    on_activate: Option<Py<PyAny>>,
}

fn wrap_index_callback(callback: Py<PyAny>) -> IndexCallback {
    Arc::new(move |index: usize| {
        Python::attach(|py| {
            if let Err(err) = callback.call1(py, (index,)) {
                err.print(py);
            }
        });
    })
}

#[pymethods]
impl ListView {
    #[new]
    #[pyo3(signature = (
        items,
        on_select=None,
        on_activate=None,
        row_height=24.0,
        font_size=None,
        flex_grow=1.0,
        width=None,
        height=None,
        text_color=None,
        background=None,
        selection_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        items: Vec<String>,
        on_select: Option<Py<PyAny>>,
        on_activate: Option<Py<PyAny>>,
        row_height: f32,
        font_size: Option<FontSize>,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        text_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        selection_color: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        if row_height <= 0.0 {
            return Err(PyValueError::new_err("row_height must be positive"));
        }
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            items: Arc::new(Mutex::new(Arc::new(items))),
            selected: Readback::new(None),
            row_height,
            font_size,
            flex_grow,
            width,
            height,
            text_color,
            background,
            selection_color,
            on_select,
            on_activate,
        })
    }

    /// Replace the rows (clears the selection and scrolls to the top). Works before attaching.
    fn set_items(&self, items: Vec<String>) -> PyResult<()> {
        let items = Arc::new(items);
        *self.items.lock().unwrap_or_else(|p| p.into_inner()) = items.clone();
        self.selected.set(None);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::ListView { items: current, selected, scroll, .. } = kind {
                *current = Arc::unwrap_or_clone(items);
                *selected = None;
                *scroll = 0.0;
            }
        })
    }

    /// Select row `index` (clamped) and scroll it into view, calling `on_select`; `None`
    /// clears the selection.
    #[pyo3(signature = (index))]
    /// Works before the list is shown, too (the selection applies when it is).
    fn select(&self, index: Option<usize>) -> PyResult<()> {
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else {
            let len = self.items.lock().unwrap_or_else(|p| p.into_inner()).len();
            self.selected.set(index.filter(|_| len > 0).map(|i| i.min(len - 1)));
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| tree.list_select(id, index))
    }

    /// The selected row, or `None`. Safe from any thread.
    #[getter]
    fn selected(&self) -> Option<usize> {
        self.selected.get()
    }

    fn __len__(&self) -> usize {
        self.items.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

impl ListView {
    fn describe(&self) -> DescribedWidget {
        let (on_select, on_activate) = Python::attach(|py| {
            (
                self.on_select.as_ref().map(|cb| cb.clone_ref(py)),
                self.on_activate.as_ref().map(|cb| cb.clone_ref(py)),
            )
        });
        let items = self.items.lock().unwrap_or_else(|p| p.into_inner()).as_ref().clone();
        let selected = self.selected.get().filter(|&i| i < items.len());
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::ListView {
                items,
                row_height: self.row_height,
                font_size: self.font_size.unwrap_or(FontSize::Small).resolve(),
                scroll: 0.0,
                selected,
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                selection_color: rgba(self.selection_color.unwrap_or(crate::theme::palette().selection)),
                on_select: on_select.map(wrap_index_callback),
                on_activate: on_activate.map(wrap_index_callback),
                mirror: Some(self.selected.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Compact cell formatting for table ingest (floats without noisy trailing zeros).
fn format_table_cell(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(v) = value.extract::<bool>() {
        // Check bool before int — bool is a subclass of int in Python.
        return Ok(if v { "True".into() } else { "False".into() });
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(v.to_string());
    }
    if let Ok(v) = value.extract::<f64>() {
        return Ok(format_table_float(v));
    }
    if value.is_none() {
        return Ok(String::new());
    }
    value.str()?.extract()
}

/// Six decimals with trailing zeros trimmed; scientific notation outside `[1e-4, 1e15)` so tiny
/// values don't collapse to `0` and huge ones don't print as long digit runs.
fn format_table_float(v: f64) -> String {
    if !v.is_finite() {
        return v.to_string();
    }
    if v == 0.0 {
        return "0".into();
    }
    fn trim(s: &str) -> &str {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.')
        } else {
            s
        }
    }
    if (1e-4..1e15).contains(&v.abs()) {
        trim(&format!("{v:.6}")).to_string()
    } else {
        let s = format!("{v:.5e}");
        let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
        format!("{}e{exp}", trim(mantissa))
    }
}

/// Turn one column sequence (list, numpy 1-D, pyarrow-ish iterable) into string cells.
fn column_to_strings(col: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(to_pylist) = col.call_method0("to_pylist") {
        return column_to_strings(&to_pylist);
    }
    if let Ok(tolist) = col.call_method0("tolist") {
        return column_to_strings(&tolist);
    }
    let seq = col.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("each table column must be a sequence (list, numpy array, …)")
    })?;
    let mut out = Vec::with_capacity(seq.len().unwrap_or(0));
    for i in 0..seq.len()? {
        out.push(format_table_cell(&seq.get_item(i)?)?);
    }
    Ok(out)
}

/// Normalize `columns=` (dict or sequence of `(name, values)`) plus optional widths into `TableData`.
fn coerce_table_data(
    columns: &Bound<'_, PyAny>,
    column_widths: Option<Vec<f32>>,
) -> PyResult<Arc<TableData>> {
    let mut headers: Vec<String> = Vec::new();
    let mut cells: Vec<Vec<String>> = Vec::new();
    if let Ok(dict) = columns.cast::<PyDict>() {
        for (key, value) in dict.iter() {
            headers.push(key.str()?.to_string_lossy().into_owned());
            cells.push(column_to_strings(&value)?);
        }
    } else {
        let seq = columns.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("columns must be a dict or a sequence of (name, values) pairs")
        })?;
        for i in 0..seq.len()? {
            let pair = seq.get_item(i)?;
            let pair_seq = pair.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err("each columns entry must be a (name, values) pair")
            })?;
            if pair_seq.len()? != 2 {
                return Err(PyValueError::new_err("each columns entry must be a (name, values) pair"));
            }
            headers.push(pair_seq.get_item(0)?.str()?.to_string_lossy().into_owned());
            cells.push(column_to_strings(&pair_seq.get_item(1)?)?);
        }
    }
    if let Some(widths) = column_widths.as_ref() {
        if !widths.is_empty() && widths.len() != headers.len() {
            return Err(PyValueError::new_err(
                "column_widths length must match the number of columns",
            ));
        }
    }
    let cols: Vec<TableColumn> = headers
        .into_iter()
        .enumerate()
        .map(|(i, header)| TableColumn {
            header,
            width: column_widths
                .as_ref()
                .and_then(|w| w.get(i).copied())
                .unwrap_or(0.0),
        })
        .collect();
    TableData::new(cols, cells)
        .map(Arc::new)
        .ok_or_else(|| PyValueError::new_err("all table columns must have the same length"))
}

/// A virtualized multi-column table: only rows in view are drawn, so it handles millions.
#[pyclass]
pub(crate) struct Table {
    id: IdCell,
    sender: SenderCell,
    data: Arc<Mutex<Arc<TableData>>>,
    selected: Readback<Option<usize>>,
    row_height: f32,
    header_height: f32,
    font_size: Option<FontSize>,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    text_color: Option<(f32, f32, f32, f32)>,
    header_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    selection_color: Option<(f32, f32, f32, f32)>,
    grid_color: Option<(f32, f32, f32, f32)>,
    on_select: Option<Py<PyAny>>,
    on_activate: Option<Py<PyAny>>,
}

#[pymethods]
impl Table {
    #[new]
    #[pyo3(signature = (
        columns,
        on_select=None,
        on_activate=None,
        row_height=24.0,
        header_height=28.0,
        column_widths=None,
        font_size=None,
        flex_grow=1.0,
        width=None,
        height=None,
        text_color=None,
        header_color=None,
        background=None,
        selection_color=None,
        grid_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        columns: &Bound<'_, PyAny>,
        on_select: Option<Py<PyAny>>,
        on_activate: Option<Py<PyAny>>,
        row_height: f32,
        header_height: f32,
        column_widths: Option<Vec<f32>>,
        font_size: Option<FontSize>,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        text_color: Option<(f32, f32, f32, f32)>,
        header_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        selection_color: Option<(f32, f32, f32, f32)>,
        grid_color: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        if row_height <= 0.0 {
            return Err(PyValueError::new_err("row_height must be positive"));
        }
        if header_height < 0.0 {
            return Err(PyValueError::new_err("header_height must be non-negative"));
        }
        let data = coerce_table_data(columns, column_widths)?;
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            data: Arc::new(Mutex::new(data)),
            selected: Readback::new(None),
            row_height,
            header_height,
            font_size,
            flex_grow,
            width,
            height,
            text_color,
            header_color,
            background,
            selection_color,
            grid_color,
            on_select,
            on_activate,
        })
    }

    /// Replace columns (clears selection and scrolls to the origin). Works before attaching.
    #[pyo3(signature = (columns, column_widths=None))]
    fn set_columns(&self, columns: &Bound<'_, PyAny>, column_widths: Option<Vec<f32>>) -> PyResult<()> {
        let data = coerce_table_data(columns, column_widths)?;
        *self.data.lock().unwrap_or_else(|p| p.into_inner()) = data.clone();
        self.selected.set(None);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Table { data: current, selected, scroll, .. } = kind {
                *current = data;
                *selected = None;
                *scroll = (0.0, 0.0);
            }
        })
    }

    /// Select row `index` (clamped) and scroll it into view; `None` clears the selection.
    #[pyo3(signature = (index))]
    fn select(&self, index: Option<usize>) -> PyResult<()> {
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else {
            let nrows = self.data.lock().unwrap_or_else(|p| p.into_inner()).nrows;
            self.selected.set(index.filter(|_| nrows > 0).map(|i| i.min(nrows - 1)));
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| tree.table_select(id, index))
    }

    #[getter]
    fn selected(&self) -> Option<usize> {
        self.selected.get()
    }

    fn __len__(&self) -> usize {
        self.data.lock().unwrap_or_else(|p| p.into_inner()).nrows
    }
}

impl Table {
    fn describe(&self) -> DescribedWidget {
        let (on_select, on_activate) = Python::attach(|py| {
            (
                self.on_select.as_ref().map(|cb| cb.clone_ref(py)),
                self.on_activate.as_ref().map(|cb| cb.clone_ref(py)),
            )
        });
        let data = self.data.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let selected = self.selected.get().filter(|&i| i < data.nrows);
        let palette = crate::theme::palette();
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::Table {
                data,
                row_height: self.row_height,
                header_height: self.header_height,
                font_size: self.font_size.unwrap_or(FontSize::Small).resolve(),
                scroll: (0.0, 0.0),
                selected,
                text_color: rgba(self.text_color.unwrap_or(palette.text)),
                header_color: rgba(self.header_color.unwrap_or(palette.text_muted)),
                background: rgba(self.background.unwrap_or(palette.surface_alt)),
                selection_color: rgba(self.selection_color.unwrap_or(palette.selection)),
                grid_color: rgba(self.grid_color.unwrap_or(palette.border)),
                on_select: on_select.map(wrap_index_callback),
                on_activate: on_activate.map(wrap_index_callback),
                mirror: Some(self.selected.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// One node in a [`TreeView`] — label plus optional nested children.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct TreeNode {
    label: String,
    children: Vec<TreeNode>,
}

#[pymethods]
impl TreeNode {
    #[new]
    #[pyo3(signature = (label, children=None))]
    fn new(label: String, children: Option<Vec<PyRef<'_, TreeNode>>>) -> Self {
        Self {
            label,
            children: children
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.clone())
                .collect(),
        }
    }

    #[getter]
    fn label(&self) -> String {
        self.label.clone()
    }
}

impl TreeNode {
    fn to_data(&self) -> TreeNodeData {
        TreeNodeData {
            label: self.label.clone(),
            children: self.children.iter().map(TreeNode::to_data).collect(),
        }
    }
}

fn wrap_path_callback(callback: Py<PyAny>) -> PathCallback {
    Arc::new(move |path: &[u32]| {
        Python::attach(|py| {
            let tuple = path.iter().map(|&i| i as usize).collect::<Vec<_>>();
            if let Err(err) = callback.call1(py, (tuple,)) {
                err.print(py);
            }
        });
    })
}

fn coerce_tree_roots(nodes: &Bound<'_, PyAny>) -> PyResult<Arc<TreeData>> {
    let seq = nodes.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("TreeView roots must be a sequence of TreeNode")
    })?;
    let mut roots = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let node = item.cast::<TreeNode>().map_err(|_| {
            PyTypeError::new_err("TreeView roots must be TreeNode instances")
        })?;
        roots.push(node.borrow().to_data());
    }
    Ok(Arc::new(TreeData::from_nested(&roots)))
}

fn path_from_py(path: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u32>>> {
    let Some(path) = path else { return Ok(None) };
    if path.is_none() {
        return Ok(None);
    }
    let seq = path.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("path must be a sequence of child indices")
    })?;
    let mut out = Vec::with_capacity(seq.len()?);
    for i in 0..seq.len()? {
        let idx: usize = seq.get_item(i)?.extract()?;
        out.push(u32::try_from(idx).map_err(|_| PyValueError::new_err("path index too large"))?);
    }
    Ok(Some(out))
}

/// A virtualized tree: only expanded rows in view are drawn.
#[pyclass]
pub(crate) struct TreeView {
    id: IdCell,
    sender: SenderCell,
    data: Arc<Mutex<Arc<TreeData>>>,
    /// Expanded node ids — kept so describe / set_nodes can restore expand state.
    expanded: Arc<Mutex<HashSet<u32>>>,
    selected: Readback<Option<Vec<u32>>>,
    row_height: f32,
    font_size: Option<FontSize>,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    text_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    selection_color: Option<(f32, f32, f32, f32)>,
    on_select: Option<Py<PyAny>>,
    on_activate: Option<Py<PyAny>>,
}

#[pymethods]
impl TreeView {
    #[new]
    #[pyo3(signature = (
        nodes,
        on_select=None,
        on_activate=None,
        row_height=24.0,
        font_size=None,
        flex_grow=1.0,
        width=None,
        height=None,
        text_color=None,
        background=None,
        selection_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nodes: &Bound<'_, PyAny>,
        on_select: Option<Py<PyAny>>,
        on_activate: Option<Py<PyAny>>,
        row_height: f32,
        font_size: Option<FontSize>,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        text_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        selection_color: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        if row_height <= 0.0 {
            return Err(PyValueError::new_err("row_height must be positive"));
        }
        let data = coerce_tree_roots(nodes)?;
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            data: Arc::new(Mutex::new(data)),
            expanded: Arc::new(Mutex::new(HashSet::new())),
            selected: Readback::new(None),
            row_height,
            font_size,
            flex_grow,
            width,
            height,
            text_color,
            background,
            selection_color,
            on_select,
            on_activate,
        })
    }

    /// Replace the tree (clears selection, collapse, and scroll). Works before attaching.
    fn set_nodes(&self, nodes: &Bound<'_, PyAny>) -> PyResult<()> {
        let data = coerce_tree_roots(nodes)?;
        *self.data.lock().unwrap_or_else(|p| p.into_inner()) = data.clone();
        self.expanded.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.selected.set(None);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::TreeView {
                data: current,
                expanded,
                selected,
                scroll,
                ..
            } = kind
            {
                *current = data;
                expanded.lock().unwrap_or_else(|p| p.into_inner()).clear();
                *selected = None;
                *scroll = 0.0;
            }
        })
    }

    /// Select the node at `path` (child-index tuple) or clear with `None`.
    #[pyo3(signature = (path))]
    fn select(&self, path: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let path = path_from_py(path)?;
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else {
            let data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            let node = path.as_ref().and_then(|p| data.id_at_path(p));
            self.selected.set(node.map(|n| data.path_of(n)));
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| {
            let node = match (&path, tree.kind(id)) {
                (Some(p), Some(WidgetKind::TreeView { data, .. })) => data.id_at_path(p),
                (None, _) => None,
                _ => return,
            };
            tree.tree_select(id, node);
        })
    }

    /// Expand or collapse the node at `path`.
    fn set_expanded(&self, path: &Bound<'_, PyAny>, expanded: bool) -> PyResult<()> {
        let path = path_from_py(Some(path))?.ok_or_else(|| PyValueError::new_err("path required"))?;
        let data = self.data.lock().unwrap_or_else(|p| p.into_inner());
        let Some(node) = data.id_at_path(&path) else {
            return Err(PyValueError::new_err("path does not resolve to a node"));
        };
        drop(data);
        if expanded {
            self.expanded.lock().unwrap_or_else(|p| p.into_inner()).insert(node);
        } else {
            self.expanded.lock().unwrap_or_else(|p| p.into_inner()).remove(&node);
        }
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else {
            // Not shown yet: mirror `tree_set_expanded`'s rule that collapsing an ancestor of the
            // selection selects the collapsed node.
            let hides_selection = self.selected.get().is_some_and(|s| s.len() > path.len() && s.starts_with(&path));
            if !expanded && hides_selection {
                self.selected.set(Some(path));
            }
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| tree.tree_set_expanded(id, node, expanded))
    }

    /// The label of the node at `path`.
    fn label(&self, path: &Bound<'_, PyAny>) -> PyResult<String> {
        let path = path_from_py(Some(path))?.ok_or_else(|| PyValueError::new_err("path required"))?;
        let data = self.data.lock().unwrap_or_else(|p| p.into_inner());
        let node = data.id_at_path(&path).ok_or_else(|| PyValueError::new_err("path does not resolve to a node"))?;
        Ok(data.nodes[node as usize].label.clone())
    }

    /// Rename the node at `path`, keeping expand state, selection and scroll. Works before
    /// attaching. Copies the tree once per call while the window shares it, so it's meant for
    /// edits, not bulk updates (use `set_nodes`).
    fn set_label(&self, path: &Bound<'_, PyAny>, label: String) -> PyResult<()> {
        let path = path_from_py(Some(path))?.ok_or_else(|| PyValueError::new_err("path required"))?;
        let data = {
            let mut data = self.data.lock().unwrap_or_else(|p| p.into_inner());
            let node =
                data.id_at_path(&path).ok_or_else(|| PyValueError::new_err("path does not resolve to a node"))?;
            Arc::make_mut(&mut data).nodes[node as usize].label = label;
            data.clone()
        };
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::TreeView { data: current, .. } = kind {
                *current = data;
            }
        })
    }

    /// Selected node path as a list of child indices, or `None`.
    #[getter]
    fn selected(&self) -> Option<Vec<usize>> {
        self.selected.get().map(|p| p.into_iter().map(|i| i as usize).collect())
    }
}

impl TreeView {
    fn describe(&self) -> DescribedWidget {
        let (on_select, on_activate) = Python::attach(|py| {
            (
                self.on_select.as_ref().map(|cb| cb.clone_ref(py)),
                self.on_activate.as_ref().map(|cb| cb.clone_ref(py)),
            )
        });
        let data = self.data.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let selected_path = self.selected.get();
        let selected = selected_path.as_ref().and_then(|p| data.id_at_path(p));
        let palette = crate::theme::palette();
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::TreeView {
                data,
                expanded: self.expanded.clone(),
                row_height: self.row_height,
                font_size: self.font_size.unwrap_or(FontSize::Small).resolve(),
                scroll: 0.0,
                selected,
                text_color: rgba(self.text_color.unwrap_or(palette.text)),
                background: rgba(self.background.unwrap_or(palette.surface_alt)),
                selection_color: rgba(self.selection_color.unwrap_or(palette.selection)),
                on_select: on_select.map(wrap_path_callback),
                on_activate: on_activate.map(wrap_path_callback),
                mirror: Some(self.selected.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// An overlay shown on demand — a menu, dropdown list, tooltip or dialog. `show(anchor)` opens
/// it next to an attached widget (in that widget's window); `Window.show_popup` opens it at a
/// point or centered. A click outside a non-modal popup, or Escape, dismisses it (`on_dismiss`
/// fires); a modal one dims the window and only closes from code (`close()`) or Escape.
#[pyclass]
pub(crate) struct Popup {
    content: Py<PyAny>,
    modal: bool,
    padding: Spacing,
    background: Option<(f32, f32, f32, f32)>,
    border: Option<(f32, f32, f32, f32)>,
    on_dismiss: Option<Py<PyAny>>,
    /// A click outside that dismisses it also reaches what's under it (menu bar menus).
    click_through: bool,
    /// A click on its anchor widget closes it (`false`: the click reaches the anchor — submenus).
    closes_on_anchor_click: bool,
    /// The open popup node, and the window it's in.
    id: IdCell,
    sender: SenderCell,
    open: Readback<bool>,
}

#[pymethods]
impl Popup {
    #[new]
    #[pyo3(signature = (
        content,
        modal=false,
        on_dismiss=None,
        padding=Spacing::Units(6.0),
        background=None,
        border=None,
        click_through=false,
        closes_on_anchor_click=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        content: Py<PyAny>,
        modal: bool,
        on_dismiss: Option<Py<PyAny>>,
        padding: Spacing,
        background: Option<(f32, f32, f32, f32)>,
        border: Option<(f32, f32, f32, f32)>,
        click_through: bool,
        closes_on_anchor_click: bool,
    ) -> Self {
        Self {
            content,
            modal,
            padding,
            background,
            border,
            on_dismiss,
            click_through,
            closes_on_anchor_click,
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            open: Readback::new(false),
        }
    }

    /// Open next to `anchor` (a widget already shown in a window) on `side`: "below" (the
    /// default), "above", "right" or "left" — flipped when there's no room. Reopening moves it.
    #[pyo3(signature = (anchor, side="below"))]
    fn show(&self, anchor: &Bound<'_, PyAny>, side: &str) -> PyResult<()> {
        let side = match side {
            "below" => PopupSide::Below,
            "above" => PopupSide::Above,
            "right" => PopupSide::Right,
            "left" => PopupSide::Left,
            other => {
                return Err(PyValueError::new_err(format!(
                    "side must be \"below\", \"above\", \"right\" or \"left\", got {other:?}"
                )))
            }
        };
        let described = describe(anchor)?;
        let not_attached = || PyRuntimeError::new_err("the anchor widget isn't shown in a window yet");
        let anchor_id = described.id_cell.lock().unwrap_or_else(|p| p.into_inner()).ok_or_else(not_attached)?;
        let sender = described.sender_cell.lock().unwrap_or_else(|p| p.into_inner()).clone().ok_or_else(not_attached)?;
        self.open_in(&sender, PopupAnchor::Widget(anchor_id, side))
    }

    /// Open with top-left at `(x, y)` in the same window as `near` (an attached widget).
    fn show_at(&self, near: &Bound<'_, PyAny>, x: f32, y: f32) -> PyResult<()> {
        let described = describe(near)?;
        let not_attached = || PyRuntimeError::new_err("the near widget isn't shown in a window yet");
        let sender = described
            .sender_cell
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or_else(not_attached)?;
        self.open_in(&sender, PopupAnchor::Point(x, y))
    }

    /// Close it (without calling `on_dismiss`). No-op when it isn't open.
    fn close(&self) -> PyResult<()> {
        let Some(sender) = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone() else { return Ok(()) };
        let id_cell = self.id.clone();
        send_tree_mutation(&sender, move |tree| {
            if let Some(id) = id_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                tree.close_popup(id);
            }
        })
    }

    /// Whether it's showing. Safe from any thread.
    #[getter]
    fn is_open(&self) -> bool {
        self.open.get()
    }
}

impl Popup {
    /// Open (or move) this popup in `sender`'s window at `anchor`.
    pub(crate) fn open_in(&self, sender: &CommandDispatch, anchor: PopupAnchor) -> PyResult<()> {
        let (content, on_dismiss) = Python::attach(|py| -> PyResult<_> {
            let content = self.content.bind(py);
            bind_dispatch(content, sender);
            Ok((describe(content)?, self.on_dismiss.as_ref().map(|cb| cb.clone_ref(py))))
        })?;
        // The content sits in a padded column inside the popup node.
        let wrapper = DescribedWidget {
            style: StyleParams { padding: self.padding.resolve(), ..StyleParams::leaf(0.0, None, None) },
            kind: WidgetKind::Container { background: transparent(), region_id: None },
            id_cell: Arc::new(Mutex::new(None)),
            sender_cell: Arc::new(Mutex::new(None)),
            children: vec![content],
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        };
        let kind = WidgetKind::Popup {
            anchor,
            modal: self.modal,
            background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
            border: rgba(self.border.unwrap_or(crate::theme::palette().border)),
            on_dismiss: on_dismiss.map(wrap_callback0),
            restore_focus: None,
            click_through: self.click_through,
            closes_on_anchor_click: self.closes_on_anchor_click,
            open: Some(self.open.clone()),
        };
        // Reopening in another window closes it in the old one first.
        let previous = self.sender.lock().unwrap_or_else(|p| p.into_inner()).replace(sender.clone());
        if let Some(previous) = previous.filter(|p| p.floating_region != sender.floating_region) {
            let id_cell = self.id.clone();
            send_tree_mutation(&previous, move |tree| {
                if let Some(id) = id_cell.lock().unwrap_or_else(|p| p.into_inner()).take() {
                    tree.close_popup(id);
                }
            })?;
        }
        let (id_cell, attach_sender) = (self.id.clone(), sender.clone());
        send_tree_mutation(sender, move |tree| {
            let mut id = id_cell.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(old) = id.take() {
                tree.close_popup(old);
            }
            // An anchor that's no longer shown (its window's content was replaced) has nowhere
            // to put the popup: leave it closed rather than open it in the corner.
            let dead_anchor = match &kind {
                WidgetKind::Popup {
                    anchor: PopupAnchor::Widget(anchor, _) | PopupAnchor::WidgetCentered(anchor, _),
                    ..
                } => tree.kind(*anchor).is_none(),
                _ => false,
            };
            if dead_anchor {
                return;
            }
            *id = Some(tree.open_popup(kind, |tree, popup| {
                attach(tree, popup, wrapper, &attach_sender);
            }));
        })
    }
}

/// A scrollable viewport onto `content`: the wheel/trackpad scrolls it, overlay scrollbars
/// appear on overflowing axes and can be dragged, and content outside it is clipped.
#[pyclass]
pub(crate) struct ScrollArea {
    id: IdCell,
    sender: SenderCell,
    content: Py<PyAny>,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    background: (f32, f32, f32, f32),
    bar_color: Option<(f32, f32, f32, f32)>,
    /// The last `scroll_to`, applied when (re)built too — so it works before the area is shown.
    initial_offset: Arc<Mutex<(f32, f32)>>,
}

#[pymethods]
impl ScrollArea {
    #[new]
    #[pyo3(signature = (
        content,
        flex_grow=1.0,
        width=None,
        height=None,
        background=(0.0, 0.0, 0.0, 0.0),
        bar_color=None,
    ))]
    fn new(
        content: Py<PyAny>,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        background: (f32, f32, f32, f32),
        bar_color: Option<(f32, f32, f32, f32)>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            content,
            flex_grow,
            width,
            height,
            background,
            bar_color,
            initial_offset: Arc::new(Mutex::new((0.0, 0.0))),
        }
    }

    /// Scroll so the content's point `(x, y)` is at the top-left (clamped to what can scroll).
    /// Works before the area is shown, too.
    fn scroll_to(&self, x: f32, y: f32) -> PyResult<()> {
        *self.initial_offset.lock().unwrap_or_else(|p| p.into_inner()) = (x.max(0.0), y.max(0.0));
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::ScrollArea { offset, .. } = kind {
                // Clamped to the content at the next layout.
                *offset = (x.max(0.0), y.max(0.0));
            }
        })
    }
}

impl ScrollArea {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let content = Python::attach(|py| describe(self.content.bind(py)))?;
        Ok(DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::ScrollArea {
                offset: *self.initial_offset.lock().unwrap_or_else(|p| p.into_inner()),
                background: rgba(self.background),
                bar_color: rgba(self.bar_color.unwrap_or(crate::theme::palette().scrollbar)),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: vec![content],
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        })
    }
}

/// A labeled on/off checkbox (`QCheckBox`). Click or Space/Enter toggles; `checked` / `set_checked`
/// mirror the value (setters never fire `on_change`).
#[pyclass]
pub(crate) struct Checkbox {
    id: IdCell,
    sender: SenderCell,
    checked: Readback<bool>,
    label: String,
    font_size: Option<FontSize>,
    text_color: Option<(f32, f32, f32, f32)>,
    box_color: Option<(f32, f32, f32, f32)>,
    check_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
    tooltip: Option<String>,
}

#[pymethods]
impl Checkbox {
    #[new]
    #[pyo3(signature = (
        label,
        checked=false,
        on_change=None,
        font_size=None,
        text_color=None,
        box_color=None,
        check_color=None,
        tooltip=None,
    ))]
    fn new(
        label: String,
        checked: bool,
        on_change: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        text_color: Option<(f32, f32, f32, f32)>,
        box_color: Option<(f32, f32, f32, f32)>,
        check_color: Option<(f32, f32, f32, f32)>,
        tooltip: Option<String>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            checked: Readback::new(checked),
            label,
            font_size,
            text_color,
            box_color,
            check_color,
            on_change,
            tooltip,
        }
    }

    #[getter]
    fn checked(&self) -> bool {
        self.checked.get()
    }

    fn set_checked(&self, checked: bool) -> PyResult<()> {
        self.checked.set(checked);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Checkbox { checked: current, .. } = kind {
                *current = checked;
            }
        })
    }
}

impl Checkbox {
    fn describe(&self) -> DescribedWidget {
        let mirror = self.checked.clone();
        let user = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        let on_change: BoolCallback = Arc::new(move |value| {
            mirror.set(value);
            if let Some(callback) = user.as_ref() {
                Python::attach(|py| {
                    if let Err(err) = callback.call1(py, (value,)) {
                        err.print(py);
                    }
                });
            }
        });
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Checkbox {
                checked: self.checked.get(),
                label: self.label.clone(),
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                box_color: rgba(self.box_color.unwrap_or(crate::theme::palette().surface_alt)),
                check_color: rgba(self.check_color.unwrap_or(crate::theme::palette().accent)),
                on_change: Some(on_change),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: self.tooltip.clone(),
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// One option in a radio group. Radios that share the same `group` id are exclusive.
#[pyclass]
pub(crate) struct Radio {
    id: IdCell,
    sender: SenderCell,
    selected: Readback<bool>,
    label: String,
    group_id: u64,
    font_size: Option<FontSize>,
    text_color: Option<(f32, f32, f32, f32)>,
    box_color: Option<(f32, f32, f32, f32)>,
    dot_color: Option<(f32, f32, f32, f32)>,
    on_select: Option<Py<PyAny>>,
}

#[pymethods]
impl Radio {
    #[new]
    #[pyo3(signature = (
        label,
        group=None,
        selected=false,
        on_select=None,
        font_size=None,
        text_color=None,
        box_color=None,
        dot_color=None,
    ))]
    fn new(
        label: String,
        group: Option<u64>,
        selected: bool,
        on_select: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        text_color: Option<(f32, f32, f32, f32)>,
        box_color: Option<(f32, f32, f32, f32)>,
        dot_color: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        let group_id = parse_radio_group(group)?;
        let mirror = Readback::new(selected);
        register_radio_mirror(group_id, &mirror);
        if selected {
            sync_radio_mirrors(group_id, &mirror, true);
        }
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            selected: mirror,
            label,
            group_id,
            font_size,
            text_color,
            box_color,
            dot_color,
            on_select,
        })
    }

    /// Stable group id — pass the same value to peer `Radio`s so only one can be selected.
    #[getter]
    fn group(&self) -> u64 {
        self.group_id
    }

    #[getter]
    fn selected(&self) -> bool {
        self.selected.get()
    }

    /// Select or clear this radio without firing `on_select`. Selecting clears peers in the
    /// same group (their `selected` getters update too).
    fn set_selected(&self, selected: bool) -> PyResult<()> {
        sync_radio_mirrors(self.group_id, &self.selected, selected);
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else {
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| {
            // Same exclusivity as a click, without firing `on_select` (setters never do).
            let Some(WidgetKind::Radio { selected: was, group_id, .. }) = tree.kind(id) else {
                return;
            };
            if *was == selected {
                return;
            }
            if !selected {
                tree.mutate_kind(id, |kind| {
                    if let WidgetKind::Radio { selected, mirror, .. } = kind {
                        *selected = false;
                        if let Some(mirror) = mirror {
                            mirror.set(false);
                        }
                    }
                });
                return;
            }
            let group_id = *group_id;
            let peers: Vec<_> = tree
                .walk()
                .filter(|&peer| {
                    matches!(tree.kind(peer), Some(WidgetKind::Radio { group_id: g, .. }) if *g == group_id)
                })
                .collect();
            for peer in peers {
                tree.mutate_kind(peer, |kind| {
                    if let WidgetKind::Radio { selected, mirror, .. } = kind {
                        *selected = peer == id;
                        if let Some(mirror) = mirror {
                            mirror.set(*selected);
                        }
                    }
                });
            }
        })
    }
}

impl Radio {
    fn describe(&self) -> DescribedWidget {
        let mirror = self.selected.clone();
        let user = self.on_select.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        // Peers cleared by `forms::select_radio` don't get a callback; their `mirror` (below)
        // is cleared there instead.
        let on_select: ClickCallback = Arc::new(move || {
            mirror.set(true);
            if let Some(callback) = user.as_ref() {
                Python::attach(|py| {
                    if let Err(err) = callback.call0(py) {
                        err.print(py);
                    }
                });
            }
        });
        DescribedWidget {
            style: StyleParams::leaf(0.0, None, None),
            kind: WidgetKind::Radio {
                selected: self.selected.get(),
                label: self.label.clone(),
                group_id: self.group_id,
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                box_color: rgba(self.box_color.unwrap_or(crate::theme::palette().surface_alt)),
                dot_color: rgba(self.dot_color.unwrap_or(crate::theme::palette().accent)),
                on_select: Some(on_select),
                mirror: Some(self.selected.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Compact on/off switch (no built-in label — pair with a `Label`).
#[pyclass]
pub(crate) struct Toggle {
    id: IdCell,
    sender: SenderCell,
    checked: Readback<bool>,
    track_off: Option<(f32, f32, f32, f32)>,
    track_on: Option<(f32, f32, f32, f32)>,
    thumb_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
    tooltip: Option<String>,
}

#[pymethods]
impl Toggle {
    #[new]
    #[pyo3(signature = (
        checked=false,
        on_change=None,
        track_off=None,
        track_on=None,
        thumb_color=None,
        tooltip=None,
    ))]
    fn new(
        checked: bool,
        on_change: Option<Py<PyAny>>,
        track_off: Option<(f32, f32, f32, f32)>,
        track_on: Option<(f32, f32, f32, f32)>,
        thumb_color: Option<(f32, f32, f32, f32)>,
        tooltip: Option<String>,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            checked: Readback::new(checked),
            track_off,
            track_on,
            thumb_color,
            on_change,
            tooltip,
        }
    }

    #[getter]
    fn checked(&self) -> bool {
        self.checked.get()
    }

    fn set_checked(&self, checked: bool) -> PyResult<()> {
        self.checked.set(checked);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Toggle { checked: current, .. } = kind {
                *current = checked;
            }
        })
    }
}

impl Toggle {
    fn describe(&self) -> DescribedWidget {
        let mirror = self.checked.clone();
        let user = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        let on_change: BoolCallback = Arc::new(move |value| {
            mirror.set(value);
            if let Some(callback) = user.as_ref() {
                Python::attach(|py| {
                    if let Err(err) = callback.call1(py, (value,)) {
                        err.print(py);
                    }
                });
            }
        });
        DescribedWidget {
            // Its natural size: a switch shouldn't stretch across a column.
            style: StyleParams::leaf(0.0, Some(fastgui_core::widget::TOGGLE_WIDTH), Some(fastgui_core::widget::TOGGLE_HEIGHT)),
            kind: WidgetKind::Toggle {
                checked: self.checked.get(),
                track_off: rgba(self.track_off.unwrap_or(crate::theme::palette().track)),
                track_on: rgba(self.track_on.unwrap_or(crate::theme::palette().accent)),
                thumb_color: rgba(self.thumb_color.unwrap_or(crate::theme::palette().text)),
                on_change: Some(on_change),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: self.tooltip.clone(),
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Numeric stepper with −/+ buttons. `decimals=0` shows an integer.
#[pyclass]
pub(crate) struct SpinBox {
    id: IdCell,
    sender: SenderCell,
    value: Readback<f32>,
    min: f32,
    max: f32,
    step: f32,
    decimals: u32,
    font_size: Option<FontSize>,
    width: Option<f32>,
    text_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    button_color: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
}

#[pymethods]
impl SpinBox {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=100.0,
        step=1.0,
        decimals=0,
        on_change=None,
        font_size=None,
        width=None,
        text_color=None,
        background=None,
        button_color=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        decimals: u32,
        on_change: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        width: Option<f32>,
        text_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        button_color: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        let value = require_finite("value", value)?;
        let min = require_finite("min", min)?;
        let max = require_finite("max", max)?;
        let step = require_positive_step("step", step)?;
        let value = round_to_decimals(clamp_range(value, min, max), decimals);
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            value: Readback::new(value),
            min,
            max,
            step,
            decimals,
            font_size,
            width,
            text_color,
            background,
            button_color,
            on_change,
        })
    }

    #[getter]
    fn value(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, value: f32) -> PyResult<()> {
        let value = require_finite("value", value)?;
        let value = round_to_decimals(clamp_range(value, self.min, self.max), self.decimals);
        self.value.set(value);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::SpinBox { value: current, .. } = kind {
                *current = value;
            }
        })
    }
}

impl SpinBox {
    fn describe(&self) -> DescribedWidget {
        let on_change = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        DescribedWidget {
            style: StyleParams::leaf(0.0, self.width, None),
            kind: WidgetKind::SpinBox {
                value: self.value.get(),
                min: self.min,
                max: self.max,
                step: self.step,
                decimals: self.decimals,
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                button_color: rgba(self.button_color.unwrap_or(crate::theme::palette().button)),
                on_change: on_change.map(wrap_callback1),
                mirror: Some(self.value.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Drag horizontally to scrub a numeric value.
#[pyclass]
pub(crate) struct NumericScrub {
    id: IdCell,
    sender: SenderCell,
    value: Readback<f32>,
    min: f32,
    max: f32,
    speed: f32,
    decimals: u32,
    font_size: Option<FontSize>,
    width: Option<f32>,
    text_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
}

#[pymethods]
impl NumericScrub {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=100.0,
        speed=0.25,
        decimals=1,
        on_change=None,
        font_size=None,
        width=None,
        text_color=None,
        background=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        value: f32,
        min: f32,
        max: f32,
        speed: f32,
        decimals: u32,
        on_change: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        width: Option<f32>,
        text_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Self> {
        let value = require_finite("value", value)?;
        let min = require_finite("min", min)?;
        let max = require_finite("max", max)?;
        let speed = require_positive_step("speed", speed)?;
        let value = round_to_decimals(clamp_range(value, min, max), decimals);
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            value: Readback::new(value),
            min,
            max,
            speed,
            decimals,
            font_size,
            width,
            text_color,
            background,
            on_change,
        })
    }

    #[getter]
    fn value(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, value: f32) -> PyResult<()> {
        let value = require_finite("value", value)?;
        let value = round_to_decimals(clamp_range(value, self.min, self.max), self.decimals);
        self.value.set(value);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::NumericScrub { value: current, .. } = kind {
                *current = value;
            }
        })
    }
}

impl NumericScrub {
    fn describe(&self) -> DescribedWidget {
        let on_change = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        DescribedWidget {
            style: StyleParams::leaf(0.0, self.width, None),
            kind: WidgetKind::NumericScrub {
                value: self.value.get(),
                min: self.min,
                max: self.max,
                speed: self.speed,
                decimals: self.decimals,
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                on_change: on_change.map(wrap_callback1),
                mirror: Some(self.value.clone()),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Determinate progress track (display only).
#[pyclass]
pub(crate) struct ProgressBar {
    id: IdCell,
    sender: SenderCell,
    value: Readback<f32>,
    min: f32,
    max: f32,
    track_color: Option<(f32, f32, f32, f32)>,
    fill_color: Option<(f32, f32, f32, f32)>,
    height: f32,
    flex_grow: f32,
}

#[pymethods]
impl ProgressBar {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=1.0,
        track_color=None,
        fill_color=None,
        height=8.0,
        flex_grow=1.0,
    ))]
    fn new(
        value: f32,
        min: f32,
        max: f32,
        track_color: Option<(f32, f32, f32, f32)>,
        fill_color: Option<(f32, f32, f32, f32)>,
        height: f32,
        flex_grow: f32,
    ) -> Self {
        let value = value.clamp(min.min(max), max.max(min));
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            value: Readback::new(value),
            min,
            max,
            track_color,
            fill_color,
            height,
            flex_grow,
        }
    }

    #[getter]
    fn value(&self) -> f32 {
        self.value.get()
    }

    fn set_value(&self, value: f32) -> PyResult<()> {
        let value = value.clamp(self.min.min(self.max), self.max.max(self.min));
        self.value.set(value);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::ProgressBar { value: current, min, max, .. } = kind {
                *current = clamp_range(value, *min, *max);
            }
        })
    }
}

impl ProgressBar {
    fn describe(&self) -> DescribedWidget {
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, None, Some(self.height)),
            kind: WidgetKind::ProgressBar {
                value: self.value.get(),
                min: self.min,
                max: self.max,
                track_color: rgba(self.track_color.unwrap_or(crate::theme::palette().track)),
                fill_color: rgba(self.fill_color.unwrap_or(crate::theme::palette().accent)),
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// Closed dropdown field: shows the selected item (or placeholder) and opens a popup list on
/// click / Space / ArrowDown.
#[pyclass]
pub(crate) struct ComboBox {
    id: IdCell,
    sender: SenderCell,
    items: Arc<Mutex<Arc<Vec<String>>>>,
    selected: Readback<Option<usize>>,
    placeholder: String,
    font_size: Option<FontSize>,
    width: Option<f32>,
    flex_grow: f32,
    text_color: Option<(f32, f32, f32, f32)>,
    placeholder_color: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    border: Option<(f32, f32, f32, f32)>,
    on_change: Option<Py<PyAny>>,
}

#[pymethods]
impl ComboBox {
    #[new]
    #[pyo3(signature = (
        items=None,
        selected=None,
        placeholder="",
        on_change=None,
        font_size=None,
        width=None,
        flex_grow=0.0,
        text_color=None,
        placeholder_color=None,
        background=None,
        border=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        items: Option<Vec<String>>,
        selected: Option<usize>,
        placeholder: &str,
        on_change: Option<Py<PyAny>>,
        font_size: Option<FontSize>,
        width: Option<f32>,
        flex_grow: f32,
        text_color: Option<(f32, f32, f32, f32)>,
        placeholder_color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        border: Option<(f32, f32, f32, f32)>,
    ) -> Self {
        let items = items.unwrap_or_default();
        let selected = selected.filter(|&i| i < items.len());
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            items: Arc::new(Mutex::new(Arc::new(items))),
            selected: Readback::new(selected),
            placeholder: placeholder.to_owned(),
            font_size,
            width,
            flex_grow,
            text_color,
            placeholder_color,
            background,
            border,
            on_change,
        }
    }

    /// Replace the dropdown rows (clears the selection). Works before attaching.
    fn set_items(&self, items: Vec<String>) -> PyResult<()> {
        let items = Arc::new(items);
        *self.items.lock().unwrap_or_else(|p| p.into_inner()) = items.clone();
        self.selected.set(None);
        let id = *self.id.lock().unwrap_or_else(|p| p.into_inner());
        let sender = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (Some(id), Some(sender)) = (id, sender) else { return Ok(()) };
        // Close an open dropdown too: it holds a copy of the old rows, and a pick in it would
        // select that index in the new rows (and leave the stale list open).
        send_tree_mutation(&sender, move |tree| {
            let mut open = None;
            tree.mutate_kind(id, |kind| {
                if let WidgetKind::ComboBox { items: current, selected, popup_id, .. } = kind {
                    *current = Arc::unwrap_or_clone(items);
                    *selected = None;
                    open = popup_id.take();
                }
            });
            if let Some(popup) = open.filter(|&p| matches!(tree.kind(p), Some(WidgetKind::Popup { .. }))) {
                tree.close_popup(popup);
            }
        })
    }

    /// Select row `index` (clamped) or clear with `None`. Does not open the popup; fires
    /// `on_change` only through the live tree path when attached.
    #[pyo3(signature = (index))]
    fn select(&self, index: Option<usize>) -> PyResult<()> {
        let items = self.items.lock().unwrap_or_else(|p| p.into_inner());
        let index = index.filter(|&i| i < items.len());
        drop(items);
        self.selected.set(index);
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::ComboBox { selected, mirror, items, on_change, .. } = kind {
                let index = index.filter(|&i| i < items.len());
                *selected = index;
                if let Some(mirror) = mirror {
                    mirror.set(index);
                }
                if let (Some(callback), Some(i)) = (on_change.clone(), index) {
                    callback(i);
                }
            }
        })
    }

    /// The selected row, or `None`. Safe from any thread.
    #[getter]
    fn selected(&self) -> Option<usize> {
        self.selected.get()
    }
}

impl ComboBox {
    fn describe(&self) -> DescribedWidget {
        let on_change = self.on_change.as_ref().map(|cb| Python::attach(|py| cb.clone_ref(py)));
        let mirror = self.selected.clone();
        let on_change: Option<IndexCallback> = on_change.map(|callback| {
            Arc::new(move |index: usize| {
                mirror.set(Some(index));
                Python::attach(|py| {
                    if let Err(err) = callback.call1(py, (index,)) {
                        err.print(py);
                    }
                });
            }) as IndexCallback
        });
        // When there is no user callback, still publish through the mirror for `.selected`.
        let on_change = on_change.or_else(|| {
            let mirror = self.selected.clone();
            Some(Arc::new(move |index: usize| {
                mirror.set(Some(index));
            }) as IndexCallback)
        });
        let items = self.items.lock().unwrap_or_else(|p| p.into_inner()).as_ref().clone();
        let selected = self.selected.get().filter(|&i| i < items.len());
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, None),
            kind: WidgetKind::ComboBox {
                items,
                selected,
                placeholder: self.placeholder.clone(),
                font_size: self.font_size.unwrap_or(FontSize::Body).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                placeholder_color: rgba(self.placeholder_color.unwrap_or(crate::theme::palette().text_muted)),
                background: rgba(self.background.unwrap_or(crate::theme::palette().surface_alt)),
                border: rgba(self.border.unwrap_or(crate::theme::palette().border)),
                selection_color: rgba(crate::theme::palette().selection),
                on_change,
                mirror: Some(self.selected.clone()),
                popup_id: None,
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// A static CPU image composited like a `Viewport`. Feed pixels with `set_image`.
#[pyclass]
pub(crate) struct Image {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl Image {
    #[new]
    #[pyo3(signature = (width=None, height=None, flex_grow=1.0))]
    fn new(width: Option<f32>, height: Option<f32>, flex_grow: f32) -> Self {
        Self {
            image_id: next_image_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            width,
            height,
            flex_grow,
        }
    }

    /// Submit a `(height, width, 3|4)` uint8 array as the image contents.
    fn set_image(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffer = pyo3::buffer::PyBuffer::<u8>::get(data)?;
        let shape = buffer.shape();
        if shape.len() != 3 || !(shape[2] == 3 || shape[2] == 4) {
            return Err(PyValueError::new_err("expected a (height, width, 3-or-4) uint8 array"));
        }
        if !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err("image buffer must be C-contiguous"));
        }
        if shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("image must be at least 1x1 pixels"));
        }
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        if width > MAX_CPU_FRAME_EXTENT || height > MAX_CPU_FRAME_EXTENT {
            return Err(PyValueError::new_err(format!(
                "image edge must be <= {MAX_CPU_FRAME_EXTENT} pixels (got {width}x{height})"
            )));
        }
        let channels = shape[2];
        let raw = buffer.to_vec(data.py())?;
        let rgba = if channels == 4 {
            raw
        } else {
            let mut out = Vec::with_capacity(raw.len() / 3 * 4);
            for pixel in raw.chunks_exact(3) {
                out.extend_from_slice(pixel);
                out.push(255);
            }
            out
        };
        self.frames.submit(CpuFrame { width, height, format: PixelFormat::Rgba8, data: rgba });
        if let Some(dispatch) = self.dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            dispatch.waker.wake();
        }
        Ok(())
    }
}

impl Image {
    fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    fn describe(&self) -> DescribedWidget {
        DescribedWidget {
            style: StyleParams::leaf(self.flex_grow, self.width, self.height),
            kind: WidgetKind::Image { image_id: self.image_id, frames: self.frames.clone() },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: Vec::new(),
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        }
    }
}

/// CSS Grid layout: `columns` equal-width tracks, children auto-flow into rows.
#[pyclass]
pub(crate) struct Grid {
    id: IdCell,
    sender: SenderCell,
    columns: u16,
    gap: Spacing,
    padding: Spacing,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    background: (f32, f32, f32, f32),
    children: Py<PyList>,
}

#[pymethods]
impl Grid {
    #[new]
    #[pyo3(signature = (
        children,
        columns=2,
        gap=Spacing::Units(0.0),
        padding=Spacing::Units(0.0),
        flex_grow=0.0,
        width=None,
        height=None,
        background=(0.0, 0.0, 0.0, 0.0),
    ))]
    fn new(
        children: Bound<'_, PyList>,
        columns: u16,
        gap: Spacing,
        padding: Spacing,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        background: (f32, f32, f32, f32),
    ) -> PyResult<Self> {
        if columns == 0 {
            return Err(PyValueError::new_err("Grid.columns must be >= 1"));
        }
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            columns,
            gap,
            padding,
            flex_grow,
            width,
            height,
            background,
            children: children.unbind(),
        })
    }
}

impl Grid {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let children = Python::attach(|py| {
            self.children
                .bind(py)
                .iter()
                .map(|child| describe(&child))
                .collect::<PyResult<Vec<_>>>()
        })?;
        Ok(DescribedWidget {
            style: StyleParams {
                direction: FlexDirection::Row,
                gap: self.gap.resolve(),
                padding: self.padding.resolve(),
                flex_grow: self.flex_grow,
                width: self.width,
                height: self.height,
                fill: false,
                align_items: None,
                absolute: None,
                wrap: false,
                grid_columns: Some(self.columns),
                visible: true,
            },
            kind: WidgetKind::Container {
                background: if self.background.3 > 0.0 { rgba(self.background) } else { transparent() },
                region_id: None,
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children,
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        })
    }
}

/// A layout container: lays its children out in a row or column via flexbox (see `taffy`).
/// `Box` in the Python API, `BoxWidget` here since `Box` is a reserved word in Rust.
#[pyclass(name = "Box")]
pub(crate) struct BoxWidget {
    id: IdCell,
    sender: SenderCell,
    direction: FlexDirection,
    gap: Spacing,
    padding: Spacing,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    /// Behind a mutex so `set_background` also applies to later rebuilds.
    background: Mutex<(f32, f32, f32, f32)>,
    wrap: bool,
    /// Mirrored into taffy `Display` at attach / via `set_display`.
    visible: AtomicBool,
    children: Py<PyList>,
    context_menu: Option<Py<PyAny>>,
    /// `(shortcut, on_click)` pairs registered as accelerators when this box attaches.
    accelerators: Vec<(String, Py<PyAny>)>,
}

#[pymethods]
impl BoxWidget {
    /// Change the fill color (`None`: transparent). Works before the box is shown, and survives
    /// rebuilds (e.g. `Window.set_theme`).
    #[pyo3(signature = (color=None))]
    fn set_background(&self, color: Option<(f32, f32, f32, f32)>) -> PyResult<()> {
        let color = color.unwrap_or((0.0, 0.0, 0.0, 0.0));
        *self.background.lock().unwrap_or_else(|p| p.into_inner()) = color;
        if self.id.lock().unwrap_or_else(|p| p.into_inner()).is_none() {
            return Ok(());
        }
        let fill = if color.3 > 0.0 { rgba(color) } else { transparent() };
        mutate(&self.id, &self.sender, move |kind| {
            if let WidgetKind::Container { background, .. } = kind {
                *background = fill;
            }
        })
    }

    #[new]
    #[pyo3(signature = (
        children,
        direction="column",
        gap=Spacing::Units(0.0),
        padding=Spacing::Units(0.0),
        flex_grow=0.0,
        width=None,
        height=None,
        background=(0.0, 0.0, 0.0, 0.0),
        wrap=false,
        visible=true,
        context_menu=None,
        accelerators=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        children: Py<PyList>,
        direction: &str,
        gap: Spacing,
        padding: Spacing,
        flex_grow: f32,
        width: Option<f32>,
        height: Option<f32>,
        background: (f32, f32, f32, f32),
        wrap: bool,
        visible: bool,
        context_menu: Option<Py<PyAny>>,
        accelerators: Option<Bound<'_, PyList>>,
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
        let mut accel = Vec::new();
        if let Some(list) = accelerators {
            for item in list.iter() {
                let (shortcut, callback): (String, Py<PyAny>) = item.extract()?;
                // A shortcut that can never fire would still be drawn in its menu row.
                if Accel::parse(&shortcut).is_none() {
                    return Err(PyValueError::new_err(format!(
                        "unsupported shortcut {shortcut:?}: use modifiers (Ctrl/Cmd, Shift, Alt) plus \
                         one letter/digit, F1-F12, or Del/Backspace/Enter/Tab/Space/Insert/Home/End/\
                         PageUp/PageDown/Up/Down/Left/Right (Alt+F4 is reserved)"
                    )));
                }
                accel.push((shortcut, callback));
            }
        }
        Ok(Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            direction,
            gap,
            padding,
            flex_grow,
            width,
            height,
            background: Mutex::new(background),
            wrap,
            visible: AtomicBool::new(visible),
            children,
            context_menu,
            accelerators: accel,
        })
    }

    /// Show or hide this box (`Display::None` when hidden). Works before and after attach.
    fn set_display(&self, visible: bool) -> PyResult<()> {
        self.visible.store(visible, Ordering::Relaxed);
        let Some(id) = *self.id.lock().unwrap_or_else(|p| p.into_inner()) else {
            return Ok(());
        };
        let Some(sender) = self.sender.lock().unwrap_or_else(|p| p.into_inner()).clone() else {
            return Ok(());
        };
        send_tree_mutation(&sender, move |tree| {
            tree.set_display(id, visible);
        })
    }
}

impl BoxWidget {
    fn describe(&self) -> PyResult<DescribedWidget> {
        let style = StyleParams {
            direction: self.direction,
            gap: self.gap.resolve(),
            padding: self.padding.resolve(),
            flex_grow: self.flex_grow,
            width: self.width,
            height: self.height,
            fill: false,
            align_items: None,
            absolute: None,
            wrap: self.wrap,
            grid_columns: None,
            visible: self.visible.load(Ordering::Relaxed),
        };

        let children = Python::attach(|py| -> PyResult<Vec<DescribedWidget>> {
            self.children.bind(py).iter().map(|child| describe(&child)).collect()
        })?;
        let context_menu = self.context_menu.as_ref().map(|cb| {
            let cb = Python::attach(|py| cb.clone_ref(py));
            wrap_context_menu(cb, self.sender.clone())
        });
        let accelerators = self
            .accelerators
            .iter()
            .map(|(shortcut, cb)| {
                let cb = Python::attach(|py| cb.clone_ref(py));
                (shortcut.clone(), wrap_callback0(cb))
            })
            .collect();

        Ok(DescribedWidget {
            style,
            kind: WidgetKind::Container {
                background: {
                    let background = *self.background.lock().unwrap_or_else(|p| p.into_inner());
                    if background.3 > 0.0 { rgba(background) } else { transparent() }
                },
                region_id: None,
            },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children,
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu,
            accelerators,
            hover_action: None,
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
/// updates it live (see `fastgui-app::app::update_dragged_splitter`). The foundation both
/// a standalone split view and `DockArea` are built on (`python/fastgui/__init__.py`).
#[pyclass]
pub(crate) struct Splitter {
    id: IdCell,
    sender: SenderCell,
    first: Py<PyAny>,
    second: Py<PyAny>,
    direction: SplitDirection,
    ratio: f32,
    bar_color: Option<(f32, f32, f32, f32)>,
    thickness: f32,
}

#[pymethods]
impl Splitter {
    #[new]
    #[pyo3(signature = (first, second, direction="row", ratio=0.5, bar_color=None, thickness=6.0))]
    fn new(
        first: Py<PyAny>,
        second: Py<PyAny>,
        direction: &str,
        ratio: f32,
        bar_color: Option<(f32, f32, f32, f32)>,
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
                align_items: None, absolute: None, wrap: false, grid_columns: None, visible: true,
            },
            kind: WidgetKind::Container { background: transparent(), region_id: None },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: vec![first_described, second_described],
            splitter_bar: Some(SplitterBarSpec {
                direction: self.direction,
                ratio: self.ratio,
                bar_color: rgba(self.bar_color.unwrap_or(crate::theme::palette().divider)),
                thickness: self.thickness,
            }),
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
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
    close_handler: CloseHandlerCell,
    /// Set by `Window.add_floating_panel` — makes `describe()` build this panel's title bar
    /// with `floating: true` (moves-on-drag instead of the rearrange/drop-zone machinery). A
    /// plain `AtomicBool`, not GIL-gated, since it's a single flag no Python callback ever
    /// touches.
    floating: Arc<std::sync::atomic::AtomicBool>,
    title: String,
    content: Py<PyAny>,
    title_font_size: Option<FontSize>,
    title_color: Option<(f32, f32, f32, f32)>,
    title_background: Option<(f32, f32, f32, f32)>,
    background: Option<(f32, f32, f32, f32)>,
    title_height: f32,
}

#[pymethods]
impl Panel {
    #[new]
    #[pyo3(signature = (
        title,
        content,
        title_font_size=None,
        title_color=None,
        title_background=None,
        background=None,
        title_height=28.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        title: String,
        content: Py<PyAny>,
        title_font_size: Option<FontSize>,
        title_color: Option<(f32, f32, f32, f32)>,
        title_background: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        title_height: f32,
    ) -> Self {
        Self {
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            region_id: next_region_id(),
            rearrange_handler: Arc::new(Mutex::new(None)),
            close_handler: Arc::new(Mutex::new(None)),
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
    pub(crate) fn set_rearrange_handler(&self, handler: Py<PyAny>) {
        *self.rearrange_handler.lock().unwrap_or_else(|p| p.into_inner()) = Some(handler);
    }

    pub(crate) fn set_close_handler(&self, handler: Py<PyAny>) {
        *self.close_handler.lock().unwrap_or_else(|p| p.into_inner()) = Some(handler);
    }
}

impl Panel {
    pub(crate) fn region_id(&self) -> u64 {
        self.region_id
    }

    /// Replace both handlers (`None` clears one) — how a floater follows whatever the window's
    /// content currently is. Returns whether either changed, compared with Python `==` since
    /// each `dock._on_rearrange` read is a fresh (but equal) bound-method object.
    pub(crate) fn replace_dock_handlers(
        &self,
        py: Python<'_>,
        rearrange: Option<Py<PyAny>>,
        close: Option<Py<PyAny>>,
    ) -> bool {
        let replace = |cell: &Mutex<Option<Py<PyAny>>>, new: Option<Py<PyAny>>| {
            let mut slot = cell.lock().unwrap_or_else(|p| p.into_inner());
            let same = match (slot.as_ref(), new.as_ref()) {
                (None, None) => true,
                (Some(old), Some(new)) => old.bind(py).eq(new.bind(py)).unwrap_or(false),
                _ => false,
            };
            *slot = new;
            !same
        };
        let rearrange_changed = replace(&self.rearrange_handler, rearrange);
        let close_changed = replace(&self.close_handler, close);
        rearrange_changed || close_changed
    }

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

    pub(crate) fn close_handler(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.close_handler
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|cb| cb.clone_ref(py))
    }

    /// Called by `Window.add_floating_panel` (`fastgui-py::lib`), before its
    /// `describe_window_content` — makes every future `describe()` build this panel's title bar
    /// with `floating: true`. Cleared by `clear_floating` when the panel is
    /// dropped back into a `DockArea` (see `Window::take_floating_panel`).
    pub(crate) fn set_floating(&self) {
        self.floating.store(true, Ordering::Relaxed);
    }

    /// Undo `set_floating` so the next `describe()` builds a normal docked title bar. Called
    /// when a floating panel is re-docked; without this, a later ungroup/`set_content` would
    /// re-attach it as an overlay instead of a flex leaf.
    pub(crate) fn clear_floating(&self) {
        self.floating.store(false, Ordering::Relaxed);
    }

    pub(crate) fn title(&self) -> String {
        self.title.clone()
    }

    /// Full-window content for a floating OS window (`Window.add_floating_panel`). Same panel
    /// chrome as `describe()`, forced to fill the dedicated window rather than absolutely
    /// positioned inside the main window.
    pub(crate) fn describe_window_content(&self) -> PyResult<DescribedWidget> {
        let mut described = self.describe()?;
        described.force_fill();
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
        let on_close = self
            .close_handler
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|cb| Python::attach(|py| cb.clone_ref(py)));

        let title_bar = DescribedWidget {
            style: StyleParams {
                direction: FlexDirection::Row,
                gap: 0.0,
                padding: 0.0,
                flex_grow: 0.0,
                width: None,
                height: Some(self.title_height),
                fill: false,
                align_items: None,
                absolute: None,
                wrap: false,
                grid_columns: None,
                visible: true,
            },
            kind: WidgetKind::PanelTitleBar {
                panel_id: self.region_id,
                title: self.title.clone(),
                font_size: self.title_font_size.unwrap_or(FontSize::Small).resolve(),
                text_color: rgba(self.title_color.unwrap_or(crate::theme::palette().text)),
                background: rgba(self.title_background.unwrap_or(crate::theme::palette().surface_alt)),
                on_drop: on_drop.map(wrap_panel_drop_callback),
                on_close: on_close.map(wrap_panel_close_callback),
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
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
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
                align_items: None, absolute: None, wrap: false, grid_columns: None, visible: true,
            },
            kind: WidgetKind::Container { background: rgba(self.background.unwrap_or(crate::theme::palette().surface)), region_id: Some(self.region_id) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: vec![title_bar, content_described],
            splitter_bar: None,
            tab_bar: None,
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        })
    }
}

/// Multiple `Panel`s sharing one region: one combined header strip (each member's `title`, not
/// its own title bar — see `Panel::title_and_content`) with only the active one's content
/// visible. Click a header segment (`fastgui-app::app::handle_tab_click`) to switch.
#[pyclass]
pub(crate) struct Tabs {
    id: IdCell,
    sender: SenderCell,
    /// This `Tabs` group's drag-and-drop identity — see `NEXT_REGION_ID`'s doc comment. A `Tabs`
    /// is a valid drop *target* (edge drop splits the region; center drop appends a tab). Member
    /// panels aren't individually titled while tabbed — drag a header segment out to ungroup
    /// (see `WidgetKind::TabBar`'s doc comment).
    region_id: u64,
    panels: Py<PyList>,
    active: usize,
    font_size: Option<FontSize>,
    text_color: Option<(f32, f32, f32, f32)>,
    active_color: Option<(f32, f32, f32, f32)>,
    inactive_color: Option<(f32, f32, f32, f32)>,
    height: f32,
    on_select: Option<Py<PyAny>>,
}

#[pymethods]
impl Tabs {
    #[new]
    #[pyo3(signature = (
        panels,
        active=0,
        font_size=None,
        text_color=None,
        active_color=None,
        inactive_color=None,
        height=28.0,
        on_select=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        panels: Py<PyList>,
        active: usize,
        font_size: Option<FontSize>,
        text_color: Option<(f32, f32, f32, f32)>,
        active_color: Option<(f32, f32, f32, f32)>,
        inactive_color: Option<(f32, f32, f32, f32)>,
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
        let (titles, panel_ids, on_drop, on_close, mut contents) = Python::attach(|py| -> PyResult<(
            Vec<String>,
            Vec<u64>,
            Vec<Option<PanelDropCallback>>,
            Vec<Option<PanelCloseCallback>>,
            Vec<DescribedWidget>,
        )> {
            let mut titles = Vec::new();
            let mut panel_ids = Vec::new();
            let mut on_drop = Vec::new();
            let mut on_close = Vec::new();
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
                on_close.push(panel.close_handler(py).map(wrap_panel_close_callback));
                contents.push(describe(content.bind(py))?);
            }
            Ok((titles, panel_ids, on_drop, on_close, contents))
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
                align_items: None, absolute: None, wrap: false, grid_columns: None, visible: true,
            },
            kind: WidgetKind::Container { background: transparent(), region_id: Some(self.region_id) },
            id_cell: self.id.clone(),
            sender_cell: self.sender.clone(),
            children: contents,
            splitter_bar: None,
            tab_bar: Some(TabBarSpec {
                titles,
                active,
                font_size: self.font_size.unwrap_or(FontSize::Small).resolve(),
                text_color: rgba(self.text_color.unwrap_or(crate::theme::palette().text)),
                active_color: rgba(self.active_color.unwrap_or(crate::theme::palette().surface_active)),
                inactive_color: rgba(self.inactive_color.unwrap_or(crate::theme::palette().surface_alt)),
                height: self.height,
                on_select: on_select.map(wrap_callback_usize),
                panel_ids,
                on_drop,
                on_close,
            }),
            tooltip: None,
            context_menu: None,
            accelerators: Vec::new(),
            hover_action: None,
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Label>()?;
    m.add_class::<Button>()?;
    m.add_class::<Slider>()?;
    m.add_class::<TextInput>()?;
    m.add_class::<TextArea>()?;
    m.add_class::<ScrollArea>()?;
    m.add_class::<Popup>()?;
    m.add_class::<ListView>()?;
    m.add_class::<Table>()?;
    m.add_class::<TreeNode>()?;
    m.add_class::<TreeView>()?;
    m.add_class::<Checkbox>()?;
    m.add_class::<Radio>()?;
    m.add_class::<Toggle>()?;
    m.add_class::<SpinBox>()?;
    m.add_class::<NumericScrub>()?;
    m.add_class::<ProgressBar>()?;
    m.add_class::<ComboBox>()?;
    m.add_class::<Image>()?;
    m.add_class::<Grid>()?;
    m.add_class::<BoxWidget>()?;
    m.add_class::<Splitter>()?;
    m.add_class::<Panel>()?;
    m.add_class::<Tabs>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::format_table_float;

    #[test]
    fn table_floats_keep_small_and_large_magnitudes() {
        assert_eq!(format_table_float(1.23456789), "1.234568");
        assert_eq!(format_table_float(2.5), "2.5");
        assert_eq!(format_table_float(-0.0), "0");
        assert_eq!(format_table_float(1e-4), "0.0001");
        assert_eq!(format_table_float(1e-9), "1e-9");
        assert_eq!(format_table_float(-2.5e-7), "-2.5e-7");
        assert_eq!(format_table_float(6.02e23), "6.02e23");
        assert_eq!(format_table_float(1e303), "1e303");
        assert_eq!(format_table_float(f64::NAN), "NaN");
    }
}
