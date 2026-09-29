use std::collections::HashMap;
use std::sync::Arc;

use taffy::prelude::*;

use crate::text_edit::{TextEdit, TextMeasure};
use crate::{FrameSlot, Readback};

/// A node in a `WidgetTree` — just a taffy `NodeId`, since taffy already owns the
/// parent/child/style graph; we only need a side-table for the widget-specific data
/// (text, colors, callbacks) taffy doesn't know about.
pub type WidgetId = NodeId;

/// Linear RGBA, 0.0..=1.0 per channel — matches `Window.set_clear_color`'s convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color(pub [f32; 4]);

impl Color {
    pub const TRANSPARENT: Color = Color([0.0, 0.0, 0.0, 0.0]);
}

/// A click/drag callback. Type-erased on purpose: `fastgui-core` has no PyO3 dependency, so
/// `fastgui-py` is the one that wraps a `Py<PyAny>` into a closure satisfying this bound
/// (re-acquiring the GIL via `Python::attach` internally) — this crate and `fastgui-render-vk`
/// never need to know Python exists.
pub type ClickCallback = Arc<dyn Fn() + Send + Sync>;
pub type ChangeCallback = Arc<dyn Fn(f32) + Send + Sync>;
pub type BoolCallback = Arc<dyn Fn(bool) + Send + Sync>;
pub type TabSelectCallback = Arc<dyn Fn(usize) + Send + Sync>;
/// `(dragged_region_id, target_region_id, zone, float_rect)`.
/// `float_rect` is `Some((x, y, width, height))` in main-window client coords when `zone` is
/// `Float` (tear a docked panel out into an OS window); `None` for ordinary dock rearrange.
pub type PanelDropCallback =
    Arc<dyn Fn(u64, u64, DropZone, Option<(f32, f32, f32, f32)>) + Send + Sync>;
/// A `ListView` row index (`on_select` / `on_activate`).
pub type IndexCallback = Arc<dyn Fn(usize) + Send + Sync>;
/// A `TextInput`'s edited text (`on_change`) or submitted text (`on_submit`, Enter).
pub type TextCallback = Arc<dyn Fn(String) + Send + Sync>;
/// Fired when the user clicks a panel/tab close control — argument is that panel's region id.
pub type PanelCloseCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Width/height of the drawn × square in a title bar or tab segment (layout units).
pub const CLOSE_BUTTON_SIZE: f32 = 22.0;

/// Extra width the × *hit* target gets to the left of its drawn square — see
/// `close_hit_rect`. Modest on purpose: it must never eat much of a tab header segment.
pub const CLOSE_HIT_SLOP: f32 = 6.0;

/// Largest fraction of a title bar / tab segment's width the × may take (drawn or hit), so a
/// narrow tab header always keeps most of its width for "switch to / drag this tab".
const CLOSE_MAX_WIDTH_FRACTION: f32 = 0.4;

/// Where an open `Popup` goes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PopupAnchor {
    /// Next to a widget, on `side` (flipped to the opposite side when there's no room there).
    Widget(WidgetId, PopupSide),
    /// With its top-left at a window point (flipped up/left near the window's far edges).
    Point(f32, f32),
    /// Centered in the window (dialogs).
    Center,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopupSide {
    Below,
    Above,
    Right,
    Left,
}

/// Gap between a popup and the widget it's anchored to, and its minimum distance from the
/// window's edges (layout units).
const POPUP_GAP: f32 = 2.0;

/// Horizontal space between a `TextInput` / `TextArea`'s edge and its text (layout units).
/// Shared by chrome's drawing and `fastgui-app`'s click-to-caret, so both agree on where text starts.
pub const TEXT_INPUT_PADDING: f32 = 8.0;
/// Vertical inset for `TextInput` / `TextArea` content (layout units).
pub const TEXT_INPUT_VERTICAL_PADDING: f32 = 6.0;

/// Drawn checkbox / radio indicator size (layout units). Shared by measure, chrome, and hit tests.
pub const CHECK_SIZE: f32 = 18.0;
/// Gap between the indicator and the label text on `Checkbox` / `Radio`.
pub const CHECK_LABEL_GAP: f32 = 8.0;
/// Intrinsic size of a `Toggle` track (layout units).
pub const TOGGLE_WIDTH: f32 = 40.0;
pub const TOGGLE_HEIGHT: f32 = 22.0;
/// Width reserved for each ± button on a `SpinBox` (layout units).
pub const SPIN_BUTTON_WIDTH: f32 = 22.0;
/// Default track thickness for a `ProgressBar` when style doesn't pin height.
pub const PROGRESS_HEIGHT: f32 = 8.0;

/// Scrollbar thickness and inset from the scroll area's edge (layout units). Overlay bars:
/// drawn over the content's edge rather than taking layout space.
pub const SCROLLBAR_WIDTH: f32 = 6.0;
const SCROLLBAR_INSET: f32 = 2.0;
/// Shortest a scrollbar thumb gets, so a very long list still has something to grab.
const SCROLLBAR_MIN_THUMB: f32 = 24.0;

/// How far (layout units, each side, across the bar) a press still grabs a `Splitter` bar —
/// see `WidgetTree::splitter_at`. The bar keeps its drawn thickness; only the hit area grows.
pub const SPLITTER_HIT_SLOP: f32 = 5.0;

/// Which axis a `Splitter` divides its two panes along — matches the parent `Box`'s own
/// `flex_direction` (a `Splitter` is itself a row/column container; see
/// `fastgui-py::widgets::Splitter`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitDirection {
    Row,
    Column,
}

/// Where a dragged panel was released relative to the region it was dropped on — `Center` means
/// "merge as a tab", the edge variants mean "split the target and put the dragged panel on that
/// edge", and `Float` means "tear out into a floating OS window" (cursor released outside the
/// main window with no dock hover).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropZone {
    Center,
    Left,
    Right,
    Top,
    Bottom,
    Float,
}

impl DropZone {
    /// Classify a point against `rect` for drag-and-drop purposes: the middle 50% of each axis
    /// (a centered box half the width and half the height) is `Center`; anywhere in the outer
    /// 25% band picks whichever edge the point is nearest to (as a fraction of that axis, so a
    /// wide, short panel still has usable top/bottom bands). `rect` is the target region's own
    /// absolute rect (title bar included) in the same logical units as the cursor — never the
    /// whole window. Shared by hover-preview and drop-commit so they always agree on the same
    /// zone for the same cursor position; `preview_rect` is the matching highlight.
    pub fn classify(rect: Rect, x: f32, y: f32) -> Self {
        if rect.width <= 0.0 || rect.height <= 0.0 {
            return DropZone::Center;
        }
        let fx = (x - rect.x) / rect.width;
        let fy = (y - rect.y) / rect.height;
        const MARGIN: f32 = 0.25;
        let dist_left = fx;
        let dist_right = 1.0 - fx;
        let dist_top = fy;
        let dist_bottom = 1.0 - fy;
        // Prefer the smallest distance with a fixed axis order so equal floats (corners,
        // rounding) don't depend on `==` matching the `min()` result.
        let mut zone = DropZone::Center;
        let mut best = MARGIN;
        if dist_left < best {
            best = dist_left;
            zone = DropZone::Left;
        }
        if dist_right < best {
            best = dist_right;
            zone = DropZone::Right;
        }
        if dist_top < best {
            best = dist_top;
            zone = DropZone::Top;
        }
        if dist_bottom < best {
            zone = DropZone::Bottom;
        }
        zone
    }

    /// The sub-area of `region` the drop preview highlights for this zone, which is where the
    /// dragged panel will end up if released now. For an edge zone that's the half of `region`
    /// the split gives it. For `Center` it's the whole region ("becomes a tab"). Every point
    /// `classify` maps to an edge zone lies inside that zone's preview rect, so the highlight
    /// always sits under the cursor.
    pub fn preview_rect(self, region: Rect) -> Rect {
        match self {
            DropZone::Center | DropZone::Float => region,
            DropZone::Left => Rect { width: region.width / 2.0, ..region },
            DropZone::Right => Rect { x: region.x + region.width / 2.0, width: region.width / 2.0, ..region },
            DropZone::Top => Rect { height: region.height / 2.0, ..region },
            DropZone::Bottom => Rect { y: region.y + region.height / 2.0, height: region.height / 2.0, ..region },
        }
    }
}

pub enum WidgetKind {
    /// A pure layout node — `Box` in the Python API (VBox/HBox is just `flex_direction`).
    /// `region_id` is `Some` only for the outermost container of a `DockArea` region (a `Panel`
    /// or `Tabs`) — see `WidgetTree::find_region_at`, used for drag-and-drop hit-testing.
    Container { background: Color, region_id: Option<u64> },
    Label { text: String, font_size: f32, color: Color },
    Button {
        text: String,
        font_size: f32,
        text_color: Color,
        background: Color,
        on_click: Option<ClickCallback>,
    },
    Slider {
        value: f32,
        min: f32,
        max: f32,
        track_color: Color,
        thumb_color: Color,
        on_change: Option<ChangeCallback>,
    },
    /// A draggable divider between two sibling panes (`first`/`second`, both children of the
    /// same parent as this node). Dragging updates `ratio` and the render thread reassigns
    /// `first`/`second`'s `flex_grow` to match — see `fastgui-app::app`'s
    /// `update_dragged_splitter`. The bar itself has no children; it's a thin leaf sized by its
    /// own fixed-`thickness` style.
    Splitter {
        direction: SplitDirection,
        ratio: f32,
        bar_color: Color,
        first: WidgetId,
        second: WidgetId,
    },
    /// A self-contained tab strip (bakes its own header-segment + text rendering, like `Slider`
    /// bakes its track+thumb, rather than composing child `Label` nodes — sidesteps the
    /// cross-axis-stretch text-clipping issue `Panel`'s title bar hit in M6, see
    /// `fastgui-py::widgets::Panel`'s history). `content_ids` are this tab strip's sibling
    /// content-wrapper nodes (same parent `Box`, one per tab, in title order); clicking a header
    /// segment sets `active` and flips the clicked wrapper's style to `Display::Flex` and every
    /// other wrapper's to `Display::None` — see `fastgui-app::app`'s tab click handling.
    /// `panel_ids`/`on_drop`/`on_close` (parallel to `titles`, one entry per tab) let a tab be
    /// dragged out or closed. Each tab's `panel_id` is its member `Panel`'s region id; `on_drop`
    /// / `on_close` come from that panel's rearrange/close handlers.
    TabBar {
        titles: Vec<String>,
        active: usize,
        font_size: f32,
        text_color: Color,
        active_color: Color,
        inactive_color: Color,
        content_ids: Vec<WidgetId>,
        on_select: Option<TabSelectCallback>,
        panel_ids: Vec<u64>,
        on_drop: Vec<Option<PanelDropCallback>>,
        on_close: Vec<Option<PanelCloseCallback>>,
    },
    /// A `Panel`'s title bar, self-contained like `TabBar` (own background+text, no child
    /// `Label`). `panel_id` matches its `Panel`'s outer `Container { region_id, .. }`, giving it
    /// an identity: dragging this bar (`fastgui-app::app`'s panel-drag handling) and
    /// releasing over another region calls `on_drop(panel_id, target_region_id, zone)` — the
    /// same callback on every `PanelTitleBar` a `DockArea` built, letting Python's `DockArea`
    /// restructure its tree and ask the `Window` to re-attach it. Standalone `Panel`s (not in a
    /// `DockArea`) just have `on_drop: None`, so dragging their title bar is a no-op.
    ///
    /// `floating`/`container_id` are for `Window.add_floating_panel`: when `floating` is true,
    /// this panel lives in its own OS window and dragging the title bar moves that window
    /// (via `set_outer_position`), including outside the main window. If `on_drop` is also set
    /// (main window content is a `DockArea`), the same drag drives drop-zone hover on the main
    /// window so releasing over a docked region re-docks the panel. `container_id` is still
    /// backfilled by `attach` for compatibility; OS-window drag no longer uses `set_position`.
    PanelTitleBar {
        panel_id: u64,
        title: String,
        font_size: f32,
        text_color: Color,
        background: Color,
        on_drop: Option<PanelDropCallback>,
        /// Click the title-bar × to close; `None` for panels that aren't closeable.
        on_close: Option<PanelCloseCallback>,
        floating: bool,
        container_id: Option<WidgetId>,
    },
    /// A single-line editable text field (`QLineEdit`). `edit` holds the text, caret, selection
    /// and undo history; `scroll` (layout units) is how far the text is shifted left so the
    /// caret stays in view — kept up to date by `fastgui-app`, which can measure text through
    /// `TextMeasure`. `preedit` is an in-progress IME composition shown at the caret (text,
    /// plus the IME's own cursor range within it). `mirror` publishes every change for Python's
    /// synchronous `TextInput.text` getter.
    TextInput {
        edit: TextEdit,
        placeholder: String,
        font_size: f32,
        text_color: Color,
        placeholder_color: Color,
        background: Color,
        selection_color: Color,
        scroll: f32,
        preedit: Option<(String, Option<(usize, usize)>)>,
        on_change: Option<TextCallback>,
        on_submit: Option<TextCallback>,
        mirror: Option<Readback<String>>,
    },
    /// A multi-line editable text field (`QTextEdit`) with hard newlines only (no soft wrap yet).
    /// Same editing/IME/`mirror` model as `TextInput`; `scroll_y` keeps the caret line in view.
    TextArea {
        edit: TextEdit,
        placeholder: String,
        font_size: f32,
        text_color: Color,
        placeholder_color: Color,
        background: Color,
        selection_color: Color,
        scroll_y: f32,
        preedit: Option<(String, Option<(usize, usize)>)>,
        on_change: Option<TextCallback>,
        on_submit: Option<TextCallback>,
        mirror: Option<Readback<String>>,
    },
    /// A viewport onto its (usually single) child, scrolled by `offset` (layout units, both
    /// axes; clamped to the content's overflow at each layout). Children lay out at their
    /// natural size and are shifted by `-offset`; everything inside is clipped to this node's
    /// rect (see `WidgetTree::clip_rect`) and can't be clicked where it's clipped out.
    /// Overflowing axes get overlay scrollbars (`scrollbar_thumbs`), drawn in `bar_color`.
    ScrollArea { offset: (f32, f32), background: Color, bar_color: Color },
    /// A virtualized list of text rows (`QListView`): `items` can be huge — only the rows in
    /// view are laid out, shaped and drawn, each `row_height` tall. It scrolls itself
    /// (`scroll`, layout units, vertical) through the same wheel/scrollbar machinery as
    /// `ScrollArea` (see `WidgetTree::scroll_offset`). Click or arrow keys select (`on_select`),
    /// double-click or Enter activates (`on_activate`); `mirror` publishes the selection for
    /// Python's `ListView.selected`.
    ListView {
        items: Vec<String>,
        row_height: f32,
        font_size: f32,
        scroll: f32,
        selected: Option<usize>,
        text_color: Color,
        background: Color,
        selection_color: Color,
        on_select: Option<IndexCallback>,
        on_activate: Option<IndexCallback>,
        mirror: Option<Readback<Option<usize>>>,
    },
    /// An overlay (menu, dropdown list, tooltip, dialog): an absolutely positioned child of the
    /// root, so it paints above everything else and wins hit tests, placed after layout by
    /// `anchor` (see `WidgetTree::open_popup`). A click outside the topmost popup dismisses it
    /// (`on_dismiss` fires) unless it's `modal`, in which case the click does nothing and
    /// everything behind it is dimmed. Chrome draws popups after `Viewport` layers, so they
    /// cover video too. `restore_focus` is where keyboard focus goes back to on close; `open`
    /// is cleared then (Python's `Popup.is_open`).
    Popup {
        anchor: PopupAnchor,
        modal: bool,
        background: Color,
        border: Color,
        on_dismiss: Option<ClickCallback>,
        restore_focus: Option<WidgetId>,
        open: Option<Readback<bool>>,
    },
    /// A GPU/CPU image rect composited on top of chrome. `frames` is the same latest-wins
    /// mailbox `Viewport.submit_frame` writes; `viewport_id` is a stable identity for the
    /// renderer's GPU texture cache across tree rebuilds (unlike `WidgetId`).
    Viewport {
        viewport_id: u64,
        frames: FrameSlot<crate::CpuFrame>,
    },
    /// A labeled on/off control. `checked` is the current value; `on_change` fires on toggle
    /// (Space/click). Indicator size is `CHECK_SIZE`; chrome draws the box and checkmark.
    Checkbox {
        checked: bool,
        label: String,
        font_size: f32,
        text_color: Color,
        box_color: Color,
        check_color: Color,
        on_change: Option<BoolCallback>,
    },
    /// One option in a radio group. `group_id` ties peers together — exclusivity is enforced
    /// later in `fastgui-app` (selecting one clears others with the same id). `on_select` fires
    /// when this option becomes selected (click/Space); it is not re-fired if already selected.
    Radio {
        selected: bool,
        label: String,
        group_id: u64,
        font_size: f32,
        text_color: Color,
        box_color: Color,
        dot_color: Color,
        on_select: Option<ClickCallback>,
    },
    /// A compact on/off switch with no built-in label (pair with a `Label` in demos). Track
    /// colors swap with `checked`; thumb is always `thumb_color`.
    Toggle {
        checked: bool,
        track_off: Color,
        track_on: Color,
        thumb_color: Color,
        on_change: Option<BoolCallback>,
    },
    /// Numeric stepper with −/+ buttons. `decimals == 0` shows an integer; otherwise that many
    /// fractional digits. `mirror` publishes every change for a synchronous Python getter, like
    /// `TextInput`.
    SpinBox {
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        decimals: u32,
        font_size: f32,
        text_color: Color,
        background: Color,
        button_color: Color,
        on_change: Option<ChangeCallback>,
        mirror: Option<Readback<f32>>,
    },
    /// Drag-to-scrub numeric field: horizontal drag changes `value` by `speed` per logical
    /// pixel. Same display/`mirror` conventions as `SpinBox`.
    NumericScrub {
        value: f32,
        min: f32,
        max: f32,
        speed: f32,
        decimals: u32,
        font_size: f32,
        text_color: Color,
        background: Color,
        on_change: Option<ChangeCallback>,
        mirror: Option<Readback<f32>>,
    },
    /// Non-interactive determinate progress track. `value` is clamped to `[min, max]` when drawn.
    ProgressBar {
        value: f32,
        min: f32,
        max: f32,
        track_color: Color,
        fill_color: Color,
    },
    /// Static/CPU image via the same latest-wins mailbox as `Viewport`. `image_id` is a stable
    /// identity for the renderer's texture cache across tree rebuilds; a later GPU path can
    /// reuse viewport draws for the same slot.
    Image {
        image_id: u64,
        frames: FrameSlot<crate::CpuFrame>,
    },
    /// Closed field showing the selected item (or `placeholder`) with a chevron; click / Space /
    /// ArrowDown opens a non-modal `Popup` anchored below with a `ListView` of `items`.
    /// `popup_id` is the open dropdown (runtime only — `describe` leaves it `None`).
    ComboBox {
        items: Vec<String>,
        selected: Option<usize>,
        placeholder: String,
        font_size: f32,
        text_color: Color,
        placeholder_color: Color,
        background: Color,
        border: Color,
        on_change: Option<IndexCallback>,
        mirror: Option<Readback<Option<usize>>>,
        /// Open dropdown popup id, if any (runtime; describe leaves None).
        popup_id: Option<WidgetId>,
    },
}

impl WidgetKind {
    /// Whether this widget can take keyboard focus (click or Tab to it).
    pub fn is_focusable(&self) -> bool {
        matches!(
            self,
            WidgetKind::Button { .. }
                | WidgetKind::Slider { .. }
                | WidgetKind::TextInput { .. }
                | WidgetKind::TextArea { .. }
                | WidgetKind::ListView { .. }
                | WidgetKind::Checkbox { .. }
                | WidgetKind::Radio { .. }
                | WidgetKind::Toggle { .. }
                | WidgetKind::SpinBox { .. }
                | WidgetKind::NumericScrub { .. }
                | WidgetKind::ComboBox { .. }
        )
    }
}

/// An axis-aligned rect in window (physical pixel) coordinates — a widget's absolute position
/// after `compute_layout`, unlike taffy's own `Layout::location`, which is parent-relative.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    /// The overlap of two rects (zero-sized, not negative, when they don't overlap).
    pub fn intersect(&self, other: &Rect) -> Rect {
        let (x0, y0) = (self.x.max(other.x), self.y.max(other.y));
        let x1 = (self.x + self.width).min(other.x + other.width);
        let y1 = (self.y + self.height).min(other.y + other.height);
        Rect { x: x0, y: y0, width: (x1 - x0).max(0.0), height: (y1 - y0).max(0.0) }
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Whether two rects share any area (touching edges don't count).
    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width
            && other.x < self.x + self.width
            && self.y < other.y + other.height
            && other.y < self.y + self.height
    }
}

/// Drawn × square on the right of a title bar or tab segment, in the same units as `bar`
/// (callers pass *logical* rects and scale the result, so drawing and hit-testing agree at any
/// DPI).
pub fn close_button_rect(bar: Rect) -> Rect {
    let size = CLOSE_BUTTON_SIZE.min(bar.height).min(bar.width.max(0.0) * CLOSE_MAX_WIDTH_FRACTION);
    Rect {
        x: bar.x + bar.width - size,
        y: bar.y + (bar.height - size) * 0.5,
        width: size,
        height: size,
    }
}

/// Click target for the ×: the drawn square widened by `CLOSE_HIT_SLOP` to the left and
/// stretched to the bar's full height, so a click that's a few px off the glyph still closes.
/// Always contains `close_button_rect(bar)`, and never takes more than
/// `CLOSE_MAX_WIDTH_FRACTION` of the bar's width, so the rest of a tab header still selects it.
pub fn close_hit_rect(bar: Rect) -> Rect {
    let drawn = close_button_rect(bar);
    let width = (drawn.width + CLOSE_HIT_SLOP).min(bar.width.max(0.0) * CLOSE_MAX_WIDTH_FRACTION).max(drawn.width);
    Rect { x: bar.x + bar.width - width, y: bar.y, width, height: bar.height }
}

/// Checkbox / radio indicator square (or the circle's bounding box), vertically centred in `row`.
pub fn check_indicator_rect(row: Rect) -> Rect {
    let size = CHECK_SIZE.min(row.height).min(row.width.max(0.0));
    Rect {
        x: row.x,
        y: row.y + (row.height - size) * 0.5,
        width: size,
        height: size,
    }
}

/// Right-hand column reserved for a `SpinBox`'s −/+ buttons.
pub fn spin_buttons_rect(row: Rect) -> Rect {
    let width = SPIN_BUTTON_WIDTH.min(row.width.max(0.0) * 0.5).min(row.height.max(SPIN_BUTTON_WIDTH));
    Rect { x: row.x + row.width - width, y: row.y, width, height: row.height }
}

/// Top half of `spin_buttons_rect` (increment).
pub fn spin_up_rect(row: Rect) -> Rect {
    let buttons = spin_buttons_rect(row);
    Rect { height: buttons.height * 0.5, ..buttons }
}

/// Bottom half of `spin_buttons_rect` (decrement).
pub fn spin_down_rect(row: Rect) -> Rect {
    let buttons = spin_buttons_rect(row);
    let half = buttons.height * 0.5;
    Rect { y: buttons.y + half, height: buttons.height - half, ..buttons }
}

/// Value area of a `SpinBox` (everything left of the button column).
pub fn spin_value_rect(row: Rect) -> Rect {
    let buttons = spin_buttons_rect(row);
    Rect { width: (row.width - buttons.width).max(0.0), ..row }
}

/// Format a spin/scrub value for display (`decimals == 0` → integer).
pub fn format_decimal(value: f32, decimals: u32) -> String {
    if decimals == 0 {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.prec$}", prec = decimals as usize)
    }
}

/// The retained-mode widget tree: taffy owns layout (parent/child structure + style), this
/// owns everything taffy doesn't (text, colors, callbacks) in a side-table keyed by the same
/// `NodeId`s. Mutated exclusively on the render thread — every Python-facing setter reaches
/// this through a `Command::MutateWidgetTree` closure (see `fastgui-render-vk::command`),
/// never directly.
pub struct WidgetTree {
    taffy: TaffyTree<()>,
    kinds: HashMap<WidgetId, WidgetKind>,
    root: WidgetId,
    /// Set on every mutation, cleared once `fastgui-chrome` has rasterized this state. Lets
    /// the render thread skip re-rasterizing (and re-uploading) a texture every frame when
    /// nothing about the UI actually changed.
    dirty: bool,
    absolute_rects: HashMap<WidgetId, Rect>,
    /// For nodes inside a `ScrollArea`: the visible part of the window they may draw in and be
    /// clicked in (the intersection of every enclosing scroll area's rect).
    clip_rects: HashMap<WidgetId, Rect>,
    /// Each `ScrollArea`'s content size (layout units), from the last layout.
    scroll_content: HashMap<WidgetId, Size<f32>>,
    /// The widget keyboard input goes to (see `fastgui-app`'s key handling). Cleared by
    /// `reset`, so a `set_content` rebuild starts unfocused.
    focused: Option<WidgetId>,
}

/// Always fills the window: whatever `Window.set_content(widget)` was given becomes this
/// node's one child, sized by its own style.
fn root_style() -> Style {
    Style {
        flex_direction: FlexDirection::Column,
        size: Size { width: Dimension::percent(1.0), height: Dimension::percent(1.0) },
        ..Default::default()
    }
}

impl WidgetTree {
    pub fn new() -> Self {
        let mut taffy = TaffyTree::new();
        let root = taffy.new_leaf(root_style()).expect("creating the root node cannot fail");
        let mut kinds = HashMap::new();
        kinds.insert(root, WidgetKind::Container { background: Color::TRANSPARENT, region_id: None });
        Self {
            taffy,
            kinds,
            root,
            dirty: true,
            absolute_rects: HashMap::new(),
            clip_rects: HashMap::new(),
            scroll_content: HashMap::new(),
            focused: None,
        }
    }

    pub fn root(&self) -> WidgetId {
        self.root
    }

    pub fn kind(&self, id: WidgetId) -> Option<&WidgetKind> {
        self.kinds.get(&id)
    }

    pub fn style(&self, id: WidgetId) -> Option<&Style> {
        self.taffy.style(id).ok()
    }

    /// Clear everything back to a fresh, empty root node. `Window.set_content(widget)`
    /// (fastgui-py) calls this and then rebuilds the tree from the given widget via
    /// `new_node`/`add_child` — simpler than exposing incremental tree-surgery for M4's scope,
    /// and a full rebuild is cheap enough (this is a UI tree, not a scene graph with thousands
    /// of nodes).
    pub fn reset(&mut self) {
        for kind in self.kinds.values() {
            if let WidgetKind::Popup { open: Some(open), .. } = kind {
                open.set(false);
            }
        }
        // Clear the existing tree rather than replacing it: its slot versions carry on, so an
        // id held from before (a closed popup, a label no longer shown) never matches a widget
        // created after. A fresh `TaffyTree` would hand those same ids out again.
        self.taffy.clear();
        self.kinds.clear();
        self.absolute_rects.clear();
        self.clip_rects.clear();
        self.scroll_content.clear();
        self.root = self.taffy.new_leaf(root_style()).expect("creating the root node cannot fail");
        self.kinds.insert(self.root, WidgetKind::Container { background: Color::TRANSPARENT, region_id: None });
        self.focused = None;
        self.mark_dirty();
    }

    pub fn new_node(&mut self, style: Style, kind: WidgetKind) -> WidgetId {
        let id = self.taffy.new_leaf(style).expect("taffy node allocation cannot fail");
        self.kinds.insert(id, kind);
        self.mark_dirty();
        id
    }

    pub fn add_child(&mut self, parent: WidgetId, child: WidgetId) {
        let _ = self.taffy.add_child(parent, child);
        self.mark_dirty();
    }

    pub fn set_style(&mut self, id: WidgetId, style: Style) {
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    /// Used by `Splitter` drag-resize (`fastgui-app::app::update_dragged_splitter`) to
    /// resize a pane without needing to reconstruct its whole `Style` at the call site.
    pub fn set_flex_grow(&mut self, id: WidgetId, flex_grow: f32) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        style.flex_grow = flex_grow;
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    /// Make `id` one of a `Splitter`'s two panes, so the panes' `flex_grow` values
    /// (`ratio * GROW_SCALE` / `(1 - ratio) * GROW_SCALE`) divide the *whole* span in exactly
    /// that ratio. Needs `flex_basis: 0` and a zero minimum size (CSS's `flex: N 1 0` +
    /// `min-width: 0`). With taffy's defaults (`flex_basis: auto`, automatic minimum size) each
    /// pane first takes its content size and only the leftover is split, and a pane never goes
    /// below its content's min-content width. So a 0.7/0.3 split with a wide label in the
    /// second pane laid out at about 0.52/0.48, and dragging the bar barely moved it. Drop-zone
    /// aiming, splitter dragging, and the documented `DockArea` sizes all assumed the ratio held.
    /// The splitter drag's 5% floor keeps a pane from vanishing. Content wider than its pane
    /// overflows (drawn over by the later sibling) rather than moving the bar.
    pub fn set_split_pane(&mut self, id: WidgetId) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        style.flex_basis = Dimension::length(0.0);
        style.min_size = Size { width: Dimension::length(0.0), height: Dimension::length(0.0) };
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    /// Make `id` (a `ScrollArea`) a scroll container: overflow scrolls on both axes (so taffy
    /// lets it be smaller than its content and reports the content's size), with no layout
    /// space reserved for scrollbars (ours overlay the content), and its children keep their
    /// natural size instead of shrinking to fit. Call once its children are added.
    pub fn set_scroll_container(&mut self, id: WidgetId) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        style.overflow = taffy::geometry::Point { x: taffy::style::Overflow::Scroll, y: taffy::style::Overflow::Scroll };
        style.scrollbar_width = 0.0;
        let _ = self.taffy.set_style(id, style);
        for child in self.taffy.children(id).unwrap_or_default() {
            let Some(mut style) = self.taffy.style(child).ok().cloned() else { continue };
            style.flex_shrink = 0.0;
            let _ = self.taffy.set_style(child, style);
        }
        self.mark_dirty();
    }

    /// Used by `TabBar` click handling (`fastgui-app::app`) to show/hide a tab's content
    /// wrapper without touching anything else about its style.
    pub fn set_display(&mut self, id: WidgetId, visible: bool) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        style.display = if visible { Display::Flex } else { Display::None };
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    /// Move an absolutely-positioned node (a floating panel's outer container — see
    /// `WidgetKind::PanelTitleBar`'s doc comment) to `(x, y)`. No-op on a node that isn't
    /// `Position::Absolute`: taffy ignores `inset` for normal-flow nodes, so this would silently
    /// do nothing useful anyway rather than actually move it.
    pub fn set_position(&mut self, id: WidgetId, x: f32, y: f32) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        if style.position != Position::Absolute {
            return;
        }
        style.inset.left = LengthPercentageAuto::length(x);
        style.inset.top = LengthPercentageAuto::length(y);
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    pub fn mutate_kind(&mut self, id: WidgetId, f: impl FnOnce(&mut WidgetKind)) {
        if let Some(kind) = self.kinds.get_mut(&id) {
            f(kind);
            self.mark_dirty();
        }
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        let _ = self.taffy.mark_dirty(self.root);
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Recompute layout for `width`x`height` (the window's physical pixel size) and refresh
    /// every node's absolute rect. Cheap to call unconditionally when `is_dirty()` is true —
    /// taffy caches unchanged subtrees internally (see `mark_dirty`'s doc).
    pub fn compute_layout(&mut self, width: f32, height: f32) {
        self.layout(width, height, None);
    }

    /// `compute_layout`, sizing text-bearing widgets (labels, buttons) by real shaping through
    /// `measure` instead of the rough per-character estimate — so a label is exactly as wide as
    /// the text chrome draws, and never loses its last word to a wrap it has no room to show.
    /// `fastgui-app` always lays out this way, with the window's `ChromeRenderer`.
    pub fn compute_layout_measured(&mut self, width: f32, height: f32, measure: &mut dyn TextMeasure) {
        self.layout(width, height, Some(measure));
    }

    fn layout(&mut self, width: f32, height: f32, mut measure: Option<&mut (dyn TextMeasure + '_)>) {
        let available = Size { width: AvailableSpace::Definite(width), height: AvailableSpace::Definite(height) };
        let kinds = &self.kinds;
        let _ = self.taffy.compute_layout_with_measure(
            self.root,
            available,
            |known_dimensions, _available_space, node_id, _node_context, _style| {
                measure_leaf(known_dimensions, node_id, kinds, measure.as_deref_mut())
            },
        );

        // Scroll offsets can't exceed the (possibly just shrunk) content's overflow.
        self.scroll_content.clear();
        for (&id, kind) in self.kinds.iter_mut() {
            let Ok(layout) = self.taffy.layout(id) else { continue };
            match kind {
                WidgetKind::ScrollArea { offset, .. } => {
                    self.scroll_content.insert(id, layout.content_size);
                    let max = max_scroll(layout.size, layout.content_size);
                    *offset = (offset.0.clamp(0.0, max.0), offset.1.clamp(0.0, max.1));
                }
                WidgetKind::ListView { items, row_height, scroll, .. } => {
                    let content = Size { width: layout.size.width, height: items.len() as f32 * *row_height };
                    self.scroll_content.insert(id, content);
                    *scroll = scroll.clamp(0.0, max_scroll(layout.size, content).1);
                }
                WidgetKind::TextArea { edit, font_size, scroll_y, .. } => {
                    // Same height math as caret-follow scroll: line stack + vertical padding.
                    let line_height = *font_size * LINE_HEIGHT_RATIO;
                    let lines = edit.text().split('\n').count().max(1) as f32;
                    let content = Size {
                        width: layout.size.width,
                        height: lines * line_height + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
                    };
                    self.scroll_content.insert(id, content);
                    *scroll_y = scroll_y.clamp(0.0, max_scroll(layout.size, content).1);
                }
                _ => {}
            }
        }

        self.absolute_rects.clear();
        self.clip_rects.clear();
        let mut stack: Vec<(WidgetId, f32, f32, Option<Rect>)> = vec![(self.root, 0.0, 0.0, None)];
        while let Some((id, parent_x, parent_y, clip)) = stack.pop() {
            let Ok(layout) = self.taffy.layout(id) else { continue };
            let x = parent_x + layout.location.x;
            let y = parent_y + layout.location.y;
            let rect = Rect { x, y, width: layout.size.width, height: layout.size.height };
            self.absolute_rects.insert(id, rect);
            if let Some(clip) = clip {
                self.clip_rects.insert(id, clip);
            }
            // A scroll area shifts its children by its offset and clips them to itself. A dock
            // region (a `Panel`'s or `Tabs`' outer container) clips too, so content wider than
            // its pane is cut off there instead of spilling over the neighbouring pane.
            let clip_to_self = || Some(clip.map_or(rect, |c| c.intersect(&rect)));
            let (child_x, child_y, child_clip) = match self.kinds.get(&id) {
                Some(WidgetKind::ScrollArea { offset, .. }) => (x - offset.0, y - offset.1, clip_to_self()),
                Some(WidgetKind::Container { region_id: Some(_), .. }) => (x, y, clip_to_self()),
                _ => (x, y, clip),
            };
            if let Ok(children) = self.taffy.children(id) {
                for child in children {
                    stack.push((child, child_x, child_y, child_clip));
                }
            }
        }
        self.place_popups(width, height);
    }

    /// Move each popup (and everything in it) to where its anchor says, now that its size and
    /// its anchor's rect are known. Popups sit at (0, 0) in layout; this shifts their rects.
    fn place_popups(&mut self, width: f32, height: f32) {
        let popups: Vec<WidgetId> = self.taffy.children(self.root).unwrap_or_default();
        for popup in popups {
            let Some(WidgetKind::Popup { anchor, .. }) = self.kinds.get(&popup) else { continue };
            let Some(size) = self.absolute_rect(popup) else { continue };
            let (x, y) = place_popup(*anchor, size, |id| self.absolute_rect(id), width, height);
            let inside: Vec<WidgetId> = self.walk_from(popup).collect();
            for node in inside {
                if let Some(rect) = self.absolute_rects.get_mut(&node) {
                    rect.x += x - size.x;
                    rect.y += y - size.y;
                }
                if let Some(clip) = self.clip_rects.get_mut(&node) {
                    clip.x += x - size.x;
                    clip.y += y - size.y;
                }
            }
        }
    }

    /// Where `id` may draw and be clicked, if something clips it (it's inside a `ScrollArea`).
    pub fn clip_rect(&self, id: WidgetId) -> Option<Rect> {
        self.clip_rects.get(&id).copied()
    }

    /// Whether `(x, y)` lands on a visible part of `id`: inside its rect and its clip.
    fn visible_at(&self, id: WidgetId, x: f32, y: f32) -> bool {
        self.absolute_rect(id).is_some_and(|r| r.contains(x, y))
            && self.clip_rects.get(&id).is_none_or(|c| c.contains(x, y))
    }

    /// How far `id` (a `ScrollArea`) can scroll on each axis, from the last layout.
    pub fn scroll_extent(&self, id: WidgetId) -> Option<(f32, f32)> {
        let content = *self.scroll_content.get(&id)?;
        Some(max_scroll(self.taffy.layout(id).ok()?.size, content))
    }

    /// The `ListView` rows at least partly in view (for drawing), from the last layout.
    pub fn list_visible_rows(&self, id: WidgetId) -> std::ops::Range<usize> {
        let (Some(WidgetKind::ListView { items, row_height, scroll, .. }), Some(rect)) = (self.kind(id), self.absolute_rect(id))
        else {
            return 0..0;
        };
        if *row_height <= 0.0 {
            return 0..0;
        }
        let first = (*scroll / row_height).floor().max(0.0) as usize;
        let last = ((*scroll + rect.height) / row_height).ceil().max(0.0) as usize;
        first.min(items.len())..last.min(items.len())
    }

    /// The `ListView` row under window y, if any.
    pub fn list_row_at(&self, id: WidgetId, y: f32) -> Option<usize> {
        let (Some(WidgetKind::ListView { items, row_height, scroll, .. }), Some(rect)) = (self.kind(id), self.absolute_rect(id))
        else {
            return None;
        };
        let row = ((y - rect.y + scroll) / row_height).floor();
        (row >= 0.0 && (row as usize) < items.len()).then_some(row as usize)
    }

    /// Select row `index` of `ListView` `id` (clamped to its rows; `None` clears) and scroll it
    /// into view. Fires `on_select` and updates the mirror when the selection changes.
    pub fn list_select(&mut self, id: WidgetId, index: Option<usize>) {
        let Some(WidgetKind::ListView { items, row_height, scroll, selected, on_select, mirror, .. }) = self.kinds.get_mut(&id)
        else {
            return;
        };
        let index = index.filter(|_| !items.is_empty()).map(|i| i.min(items.len() - 1));
        let changed = *selected != index;
        *selected = index;
        let view = self.absolute_rects.get(&id).map_or(0.0, |r| r.height);
        if let Some(i) = index {
            let (top, bottom) = (i as f32 * *row_height, (i + 1) as f32 * *row_height);
            *scroll = scroll.min(top).max(bottom - view);
        }
        let (callback, mirror) = (on_select.clone(), mirror.clone());
        self.mark_dirty();
        if changed {
            if let Some(mirror) = mirror {
                mirror.set(index);
            }
            if let (Some(callback), Some(i)) = (callback, index) {
                callback(i);
            }
        }
    }

    /// How many whole rows fit in `ListView` `id` (at least 1) — PageUp/PageDown's step.
    pub fn list_page_rows(&self, id: WidgetId) -> usize {
        let (Some(WidgetKind::ListView { row_height, .. }), Some(rect)) = (self.kind(id), self.absolute_rect(id)) else {
            return 1;
        };
        ((rect.height / row_height).floor() as usize).max(1)
    }

    /// A scrollable widget's current offset (`ScrollArea`, a `ListView`'s vertical scroll, or a
    /// `TextArea`'s `scroll_y`).
    pub fn scroll_offset(&self, id: WidgetId) -> Option<(f32, f32)> {
        match self.kind(id)? {
            WidgetKind::ScrollArea { offset, .. } => Some(*offset),
            WidgetKind::ListView { scroll, .. } => Some((0.0, *scroll)),
            WidgetKind::TextArea { scroll_y, .. } => Some((0.0, *scroll_y)),
            _ => None,
        }
    }

    /// Set a scrollable widget's offset (clamped). Returns whether it moved.
    pub fn set_scroll_offset(&mut self, id: WidgetId, x: f32, y: f32) -> bool {
        let Some(max) = self.scroll_extent(id) else { return false };
        let target = (x.clamp(0.0, max.0), y.clamp(0.0, max.1));
        if self.scroll_offset(id) == Some(target) {
            return false;
        }
        match self.kinds.get_mut(&id) {
            Some(WidgetKind::ScrollArea { offset, .. }) => *offset = target,
            Some(WidgetKind::ListView { scroll, .. }) => *scroll = target.1,
            Some(WidgetKind::TextArea { scroll_y, .. }) => *scroll_y = target.1,
            _ => return false,
        }
        self.mark_dirty();
        true
    }

    /// Scroll by `(dx, dy)` (layout units; positive reveals content further right/down) the
    /// innermost scrollable under `(x, y)` (`ScrollArea`, `ListView`, `TextArea`) that can still
    /// move that way; one at its limit passes the scroll out to the one around it. Returns
    /// whether anything scrolled.
    pub fn scroll_at(&mut self, x: f32, y: f32, dx: f32, dy: f32) -> bool {
        let areas: Vec<WidgetId> =
            self.walk().filter(|&id| self.scroll_offset(id).is_some() && self.visible_at(id, x, y)).collect();
        // `walk` visits parents first, so the innermost area comes last.
        for id in areas.into_iter().rev() {
            let Some((ox, oy)) = self.scroll_offset(id) else { continue };
            if self.set_scroll_offset(id, ox + dx, oy + dy) {
                return true;
            }
        }
        false
    }

    /// Scroll every `ScrollArea` around `id` just enough to bring `id`'s rect into view — used
    /// when keyboard focus moves to something scrolled out of sight.
    pub fn scroll_into_view(&mut self, id: WidgetId) {
        let Some(mut target) = self.absolute_rect(id) else { return };
        let mut node = id;
        while let Some(parent) = self.taffy.parent(node) {
            node = parent;
            let (Some(WidgetKind::ScrollArea { offset, .. }), Some(view)) = (self.kind(node), self.absolute_rect(node))
            else {
                continue;
            };
            let (before_x, before_y) = *offset;
            // Where the target sits in the area's content, and the part of it now in view.
            let (tx, ty) = (target.x - view.x + before_x, target.y - view.y + before_y);
            // Scroll the least that shows all of it; something bigger than the area shows its
            // start (top/left) rather than its end.
            let reveal = |before: f32, start: f32, size: f32, view: f32| {
                if size > view { start } else { before.min(start).max(start + size - view) }
            };
            let ox = reveal(before_x, tx, target.width, view.width);
            let oy = reveal(before_y, ty, target.height, view.height);
            self.set_scroll_offset(node, ox, oy);
            // Rects aren't recomputed until the next layout: move the target by what this area
            // actually scrolled (after clamping) so the areas around it aim at where it now is.
            let (after_x, after_y) = self.scroll_offset(node).unwrap_or((before_x, before_y));
            target.x -= after_x - before_x;
            target.y -= after_y - before_y;
        }
    }

    /// A `ScrollArea`'s overlay scrollbar thumbs (vertical, horizontal), in window layout
    /// units; `None` for an axis that doesn't overflow. Shared by chrome (drawing) and
    /// `fastgui-app` (dragging) so the drawn thumb is the one you grab.
    pub fn scrollbar_thumbs(&self, id: WidgetId) -> (Option<Rect>, Option<Rect>) {
        let (Some(offset), Some(rect)) = (self.scroll_offset(id), self.absolute_rect(id)) else {
            return (None, None);
        };
        let corner = SCROLLBAR_WIDTH + SCROLLBAR_INSET;
        let v = self.scrollbar_track(id, true).map(|track| {
            let y = track.start + track.travel() * (offset.1 / track.max);
            Rect { x: rect.x + rect.width - corner, y, width: SCROLLBAR_WIDTH, height: track.thumb }
        });
        let h = self.scrollbar_track(id, false).map(|track| {
            let x = track.start + track.travel() * (offset.0 / track.max);
            Rect { x, y: rect.y + rect.height - corner, width: track.thumb, height: SCROLLBAR_WIDTH }
        });
        (v, h)
    }

    /// The scroll area whose scrollbar thumb is at `(x, y)` (with `slop` extra across the bar),
    /// and whether it's the vertical one. Checked before other hit tests, like splitters.
    pub fn scrollbar_at(&self, x: f32, y: f32, slop: f32) -> Option<(WidgetId, bool)> {
        let areas: Vec<WidgetId> = self.walk().filter(|&id| self.scroll_offset(id).is_some()).collect();
        areas.into_iter().rev().find_map(|id| {
            if self.clip_rects.get(&id).is_some_and(|c| !c.contains(x, y)) {
                return None;
            }
            let (v, h) = self.scrollbar_thumbs(id);
            let grown_v = v.map(|r| Rect { x: r.x - slop, width: r.width + 2.0 * slop, ..r });
            let grown_h = h.map(|r| Rect { y: r.y - slop, height: r.height + 2.0 * slop, ..r });
            if grown_v.is_some_and(|r| r.contains(x, y)) {
                Some((id, true))
            } else if grown_h.is_some_and(|r| r.contains(x, y)) {
                Some((id, false))
            } else {
                None
            }
        })
    }

    /// Add a popup holding `build`'s content (a subtree `build` creates under the popup node it's
    /// given) and focus its first focusable widget, remembering the current focus to restore on
    /// close. Returns the popup's id.
    pub fn open_popup(&mut self, kind: WidgetKind, build: impl FnOnce(&mut Self, WidgetId)) -> WidgetId {
        let style = Style {
            position: Position::Absolute,
            flex_direction: FlexDirection::Column,
            inset: taffy::prelude::Rect {
                left: LengthPercentageAuto::length(0.0),
                top: LengthPercentageAuto::length(0.0),
                right: LengthPercentageAuto::auto(),
                bottom: LengthPercentageAuto::auto(),
            },
            ..Default::default()
        };
        let restore = self.focused;
        let id = self.new_node(style, kind);
        if let Some(WidgetKind::Popup { restore_focus, open, .. }) = self.kinds.get_mut(&id) {
            *restore_focus = restore;
            if let Some(open) = open {
                open.set(true);
            }
        }
        let root = self.root;
        self.add_child(root, id);
        build(self, id);
        let first = self.walk_from(id).find(|&n| self.kind(n).is_some_and(WidgetKind::is_focusable));
        self.focused = first;
        id
    }

    /// Close popup `id` (and its content): remove it, restore focus, clear `open`. Doesn't
    /// fire `on_dismiss` — `dismiss_popup` is the user-driven close.
    pub fn close_popup(&mut self, id: WidgetId) {
        let Some(WidgetKind::Popup { restore_focus, open, .. }) = self.kinds.get(&id) else { return };
        let restore = restore_focus.filter(|f| self.kinds.contains_key(f));
        if let Some(open) = open {
            open.set(false);
        }
        let inside: Vec<WidgetId> = self.walk_from(id).collect();
        if self.focused.is_none_or(|f| inside.contains(&f)) {
            self.focused = restore;
        }
        for node in inside {
            self.kinds.remove(&node);
            self.absolute_rects.remove(&node);
            self.clip_rects.remove(&node);
        }
        let _ = self.taffy.remove_child(self.root, id);
        remove_subtree(&mut self.taffy, id);
        self.mark_dirty();
    }

    /// Close the topmost popup as the user did (outside click, Escape): fires its `on_dismiss`.
    /// Returns whether there was one.
    pub fn dismiss_popup(&mut self) -> bool {
        let Some(id) = self.topmost_popup() else { return false };
        let callback = match self.kind(id) {
            Some(WidgetKind::Popup { on_dismiss, .. }) => on_dismiss.clone(),
            _ => None,
        };
        self.close_popup(id);
        if let Some(callback) = callback {
            callback();
        }
        true
    }

    /// The last-opened popup still open, if any.
    pub fn topmost_popup(&self) -> Option<WidgetId> {
        self.taffy
            .children(self.root)
            .ok()?
            .into_iter()
            .rev()
            .find(|id| matches!(self.kind(*id), Some(WidgetKind::Popup { .. })))
    }

    /// The popup `id` is inside (or is), if any.
    pub fn popup_of(&self, id: WidgetId) -> Option<WidgetId> {
        let mut node = id;
        loop {
            if matches!(self.kind(node), Some(WidgetKind::Popup { .. })) {
                return Some(node);
            }
            node = self.taffy.parent(node)?;
        }
    }

    /// What a press at `(x, y)` does with popups open: `Pass` lets it through to whatever's
    /// under it; `Consumed` means it dismissed a popup (outside click) or hit a modal popup's
    /// dimmed backdrop, and nothing else should see it.
    pub fn popup_press(&mut self, x: f32, y: f32) -> PopupPress {
        let Some(top) = self.topmost_popup() else { return PopupPress::Pass };
        if self.absolute_rect(top).is_some_and(|r| r.contains(x, y)) {
            return PopupPress::Pass;
        }
        match self.kind(top) {
            Some(WidgetKind::Popup { modal: true, .. }) => PopupPress::Consumed,
            _ => {
                self.dismiss_popup();
                PopupPress::Consumed
            }
        }
    }

    /// Depth-first walk of `id`'s subtree (including `id`).
    fn walk_from(&self, id: WidgetId) -> impl Iterator<Item = WidgetId> + '_ {
        let mut stack = vec![id];
        std::iter::from_fn(move || {
            let id = stack.pop()?;
            if let Ok(children) = self.taffy.children(id) {
                stack.extend(children.into_iter().rev());
            }
            Some(id)
        })
    }

    /// Scroll `id` so its vertical (or horizontal) thumb starts at `thumb_start` (window layout
    /// units along the bar) — dragging a scrollbar. Returns whether it moved.
    pub fn drag_scrollbar(&mut self, id: WidgetId, vertical: bool, thumb_start: f32) -> bool {
        let Some(track) = self.scrollbar_track(id, vertical) else { return false };
        let Some(offset) = self.scroll_offset(id) else { return false };
        let fraction = if track.travel() > 0.0 { (thumb_start - track.start) / track.travel() } else { 0.0 };
        let value = fraction.clamp(0.0, 1.0) * track.max;
        let (x, y) = if vertical { (offset.0, value) } else { (value, offset.1) };
        self.set_scroll_offset(id, x, y)
    }

    fn scrollbar_track(&self, id: WidgetId, vertical: bool) -> Option<ScrollTrack> {
        let rect = self.absolute_rect(id)?;
        let content = *self.scroll_content.get(&id)?;
        let max = self.scroll_extent(id)?;
        let (axis_max, other_max) = if vertical { (max.1, max.0) } else { (max.0, max.1) };
        if axis_max <= 0.0 {
            return None;
        }
        let (start, extent, visible, content) = if vertical {
            (rect.y, rect.height, rect.height, content.height)
        } else {
            (rect.x, rect.width, rect.width, content.width)
        };
        // Leave the corner free when both bars show.
        let corner = if other_max > 0.0 { SCROLLBAR_WIDTH + SCROLLBAR_INSET } else { 0.0 };
        let length = (extent - 2.0 * SCROLLBAR_INSET - corner).max(0.0);
        let thumb = (length * visible / content).clamp(SCROLLBAR_MIN_THUMB.min(length), length);
        Some(ScrollTrack { start: start + SCROLLBAR_INSET, length, thumb, max: axis_max })
    }

    pub fn absolute_rect(&self, id: WidgetId) -> Option<Rect> {
        self.absolute_rects.get(&id).copied()
    }

    /// Depth-first (parent before children) walk of every node currently in the tree, for
    /// `fastgui-chrome`'s rasterizer and for hit-testing.
    pub fn walk(&self) -> impl Iterator<Item = WidgetId> + '_ {
        let mut stack = vec![self.root];
        std::iter::from_fn(move || {
            let id = stack.pop()?;
            if let Ok(children) = self.taffy.children(id) {
                stack.extend(children.into_iter().rev());
            }
            Some(id)
        })
    }

    /// Topmost (last-drawn-on-top wins) widget whose rect contains `(x, y)`, for click/drag
    /// hit-testing. Depth-first-last-child-wins matches normal painter's-algorithm z-order.
    pub fn hit_test(&self, x: f32, y: f32) -> Option<WidgetId> {
        self.walk().filter(|&id| self.visible_at(id, x, y)).last()
    }

    /// The `Splitter` bar whose rect, widened by `slop` on both sides across the bar (x for a
    /// `Row` splitter's vertical bar, y for a `Column` splitter's horizontal bar), contains
    /// `(x, y)`. Press handling checks this *before* `hit_test`, so a thin bar wins over the
    /// panel title bar or content it borders. If two bars are in range the nearest wins.
    pub fn splitter_at(&self, x: f32, y: f32, slop: f32) -> Option<WidgetId> {
        let mut best: Option<(f32, WidgetId)> = None;
        for id in self.walk() {
            let Some(WidgetKind::Splitter { direction, .. }) = self.kind(id) else { continue };
            let Some(rect) = self.absolute_rect(id) else { continue };
            if rect.width <= 0.0 || rect.height <= 0.0 || self.clip_rects.get(&id).is_some_and(|c| !c.contains(x, y)) {
                continue;
            }
            let grown = match direction {
                SplitDirection::Row => Rect { x: rect.x - slop, width: rect.width + 2.0 * slop, ..rect },
                SplitDirection::Column => Rect { y: rect.y - slop, height: rect.height + 2.0 * slop, ..rect },
            };
            if !grown.contains(x, y) {
                continue;
            }
            let distance = match direction {
                SplitDirection::Row => (x - (rect.x + rect.width / 2.0)).abs(),
                SplitDirection::Column => (y - (rect.y + rect.height / 2.0)).abs(),
            };
            if best.is_none_or(|(d, _)| distance < d) {
                best = Some((distance, id));
            }
        }
        best.map(|(_, id)| id)
    }

    pub fn focused(&self) -> Option<WidgetId> {
        self.focused
    }

    /// Give `id` keyboard focus (`None` clears it). Ignores ids that aren't focusable, so a
    /// click on a label or background just clears focus. Marks dirty only on a change, since
    /// the focus ring is chrome.
    pub fn set_focus(&mut self, id: Option<WidgetId>) {
        let id = id.filter(|&id| self.kind(id).is_some_and(WidgetKind::is_focusable));
        if id != self.focused {
            self.focused = id;
            self.dirty = true;
        }
    }

    /// Move focus to the next (or, with `backward`, previous) focusable widget in tree order,
    /// wrapping around — Tab / Shift+Tab. Skips widgets laid out at zero size, which is how
    /// hidden tab content and other `Display::None` subtrees end up. Returns the new focus.
    pub fn focus_next(&mut self, backward: bool) -> Option<WidgetId> {
        // An open popup keeps Tab inside it (menus and dialogs trap focus).
        let scope = self.topmost_popup().unwrap_or(self.root);
        let order: Vec<WidgetId> = self
            .walk_from(scope)
            .filter(|&id| self.kind(id).is_some_and(WidgetKind::is_focusable))
            .filter(|&id| self.absolute_rect(id).is_some_and(|r| r.width > 0.0 && r.height > 0.0))
            .collect();
        if order.is_empty() {
            self.set_focus(None);
            return None;
        }
        let current = self.focused.and_then(|f| order.iter().position(|&id| id == f));
        let next = match (current, backward) {
            (Some(i), false) => (i + 1) % order.len(),
            (Some(i), true) => (i + order.len() - 1) % order.len(),
            (None, false) => 0,
            (None, true) => order.len() - 1,
        };
        self.set_focus(Some(order[next]));
        self.focused
    }

    /// Find the `DockArea` region (a `Container { region_id: Some(_), .. }` — a `Panel`'s or
    /// `Tabs`' own outer container) containing `(x, y)`, for drag-and-drop drop-target
    /// hit-testing. Unlike `hit_test`, this deliberately isn't "topmost wins": docked regions
    /// never overlap (each occupies a distinct rect via the `Splitter` tree they're arranged in),
    /// so any match is *the* match, and a region's rect always contains whatever's drawn on top
    /// of it (its title bar, content, etc.) — walking to find one that both carries a `region_id`
    /// and contains the point is enough, no z-order tie-breaking needed.
    ///
    /// Absolutely-positioned containers (floating panels) also carry a `region_id` but are
    /// overlays, not dock drop targets — skip them so a drop lands on the docked region
    /// underneath rather than on the floating panel the cursor is visually over.
    pub fn find_region_at(&self, x: f32, y: f32) -> Option<(u64, Rect)> {
        self.walk().find_map(|id| {
            let Some(WidgetKind::Container { region_id: Some(region_id), .. }) = self.kind(id) else {
                return None;
            };
            if self.taffy.style(id).ok().is_some_and(|s| s.position == Position::Absolute) {
                return None;
            }
            let rect = self.absolute_rect(id)?;
            rect.contains(x, y).then_some((*region_id, rect))
        })
    }

    /// Absolute rect of the `Container` carrying `region_id`, if any — used when tearing a
    /// docked panel out to size the new floating window.
    pub fn find_region_rect(&self, region_id: u64) -> Option<Rect> {
        self.walk().find_map(|id| {
            let Some(WidgetKind::Container { region_id: Some(rid), .. }) = self.kind(id) else {
                return None;
            };
            (*rid == region_id).then(|| self.absolute_rect(id)).flatten()
        })
    }
}

/// One scrollbar's geometry along its axis (window layout units).
struct ScrollTrack {
    start: f32,
    length: f32,
    thumb: f32,
    /// The scroll area's maximum offset on this axis.
    max: f32,
}

impl ScrollTrack {
    /// How far the thumb's start can move along the track.
    fn travel(&self) -> f32 {
        self.length - self.thumb
    }
}

/// See `WidgetTree::popup_press`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PopupPress {
    Pass,
    Consumed,
}

/// Top-left for a popup of `size` (its laid-out rect) given its anchor, keeping it inside a
/// `width`×`height` window: flipped to the anchor's other side when it doesn't fit, then
/// clamped. `rect_of` looks up the anchor widget's rect.
fn place_popup(
    anchor: PopupAnchor,
    size: Rect,
    rect_of: impl Fn(WidgetId) -> Option<Rect>,
    width: f32,
    height: f32,
) -> (f32, f32) {
    let (w, h) = (size.width, size.height);
    let fits_x = |x: f32| x >= POPUP_GAP && x + w <= width - POPUP_GAP;
    let fits_y = |y: f32| y >= POPUP_GAP && y + h <= height - POPUP_GAP;
    let (x, y) = match anchor {
        PopupAnchor::Center => ((width - w) / 2.0, (height - h) / 2.0),
        PopupAnchor::Point(px, py) => {
            let x = if px + w > width - POPUP_GAP && px - w >= POPUP_GAP { px - w } else { px };
            let y = if py + h > height - POPUP_GAP && py - h >= POPUP_GAP { py - h } else { py };
            (x, y)
        }
        PopupAnchor::Widget(id, side) => {
            let a = rect_of(id).unwrap_or_default();
            let below = a.y + a.height + POPUP_GAP;
            let above = a.y - POPUP_GAP - h;
            let right = a.x + a.width + POPUP_GAP;
            let left = a.x - POPUP_GAP - w;
            match side {
                PopupSide::Below => (a.x, if fits_y(below) || !fits_y(above) { below } else { above }),
                PopupSide::Above => (a.x, if fits_y(above) || !fits_y(below) { above } else { below }),
                PopupSide::Right => (if fits_x(right) || !fits_x(left) { right } else { left }, a.y),
                PopupSide::Left => (if fits_x(left) || !fits_x(right) { left } else { right }, a.y),
            }
        }
    };
    let clamp = |v: f32, extent: f32, limit: f32| v.min(limit - POPUP_GAP - extent).max(POPUP_GAP);
    (clamp(x, w, width), clamp(y, h, height))
}

/// Free `id` and its descendants from `taffy` (it's already detached from its parent).
fn remove_subtree(taffy: &mut TaffyTree<()>, id: WidgetId) {
    for child in taffy.children(id).unwrap_or_default() {
        remove_subtree(taffy, child);
    }
    let _ = taffy.remove(id);
}

/// How far content of `content` size can scroll inside a `size` viewport, per axis.
fn max_scroll(size: Size<f32>, content: Size<f32>) -> (f32, f32) {
    ((content.width - size.width).max(0.0), (content.height - size.height).max(0.0))
}

impl Default for WidgetTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Very rough text-extent estimate (average proportional-font advance width) — the fallback
/// for `compute_layout` without a `TextMeasure`; the app always uses real shaping
/// (`compute_layout_measured`), where this underestimated wide text and labels lost their last
/// words. Not a substitute for
/// real shaping — `fastgui-chrome` does that with `cosmic-text` when it actually rasterizes —
/// this only has to be good enough that layout doesn't look broken before that happens. Good
/// enough for M4; swap for measuring through `fastgui-chrome`'s font system if/when this
/// approximation visibly matters (e.g. non-Latin scripts, tight-fitting layouts).
pub const LINE_HEIGHT_RATIO: f32 = 1.3;

fn measure_text(text: &str, font_size: f32) -> Size<f32> {
    const AVG_ADVANCE_RATIO: f32 = 0.55;
    let width = text.chars().count() as f32 * font_size * AVG_ADVANCE_RATIO;
    Size { width, height: font_size * LINE_HEIGHT_RATIO }
}

/// Real text extent via `measure`: the widest line's shaped width, rounded up to whole units,
/// and one line height per line.
fn measure_shaped(measure: &mut dyn TextMeasure, text: &str, font_size: f32) -> Size<f32> {
    let width = text.split('\n').map(|line| measure.caret_x(line, font_size, line.len())).fold(0.0, f32::max);
    let lines = text.split('\n').count().max(1) as f32;
    Size { width: width.ceil(), height: font_size * LINE_HEIGHT_RATIO * lines }
}

/// taffy calls this once per leaf node during layout. `known_dimensions` is already `Some` for
/// any axis the node's own style pins (explicit size, or a parent that stretched it) — we only
/// need to supply the *intrinsic content* size for whichever axes are still `None`.
fn measure_leaf<'m>(
    known_dimensions: Size<Option<f32>>,
    node_id: NodeId,
    kinds: &HashMap<WidgetId, WidgetKind>,
    mut measure: Option<&mut (dyn TextMeasure + 'm)>,
) -> Size<f32> {
    let mut text_size = |text: &str, font_size: f32| match measure.as_deref_mut() {
        Some(measure) => measure_shaped(measure, text, font_size),
        None => measure_text(text, font_size),
    };
    let content_size = match kinds.get(&node_id) {
        Some(WidgetKind::Label { text, font_size, .. }) => text_size(text, *font_size),
        Some(WidgetKind::Button { text, font_size, .. }) => {
            let text_size = text_size(text, *font_size);
            // Room for the button's own padding beyond the text itself; real padding is
            // applied via the node's taffy `Style`, this is just the intrinsic minimum.
            const BUTTON_PADDING: f32 = 16.0;
            Size { width: text_size.width + BUTTON_PADDING, height: text_size.height + BUTTON_PADDING }
        }
        Some(WidgetKind::TextInput { font_size, .. }) => Size {
            // Wide enough to type into when nothing stretches it; the height fits one line.
            width: 160.0,
            height: font_size * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
        },
        Some(WidgetKind::TextArea { font_size, .. }) => Size {
            // Default ~4 lines tall; soft wrap is not modeled — height is for hard newlines.
            width: 160.0,
            height: font_size * LINE_HEIGHT_RATIO * 4.0 + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
        },
        Some(WidgetKind::Checkbox { label, font_size, .. } | WidgetKind::Radio { label, font_size, .. }) => {
            let text_size = measure_text(label, *font_size);
            Size {
                width: CHECK_SIZE + CHECK_LABEL_GAP + text_size.width,
                height: CHECK_SIZE.max(text_size.height),
            }
        }
        Some(WidgetKind::Toggle { .. }) => Size { width: TOGGLE_WIDTH, height: TOGGLE_HEIGHT },
        Some(WidgetKind::SpinBox { font_size, .. }) => Size {
            width: 100.0,
            height: font_size * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
        },
        Some(WidgetKind::NumericScrub { font_size, .. }) => Size {
            width: 80.0,
            height: font_size * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
        },
        Some(WidgetKind::ProgressBar { .. }) => Size { width: 120.0, height: PROGRESS_HEIGHT },
        Some(WidgetKind::ComboBox { font_size, .. }) => Size {
            width: 160.0,
            height: font_size * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING,
        },
        Some(
            WidgetKind::Container { .. }
            | WidgetKind::Slider { .. }
            | WidgetKind::Splitter { .. }
            | WidgetKind::TabBar { .. }
            | WidgetKind::PanelTitleBar { .. }
            | WidgetKind::ScrollArea { .. }
            | WidgetKind::Popup { .. }
            | WidgetKind::Viewport { .. }
            | WidgetKind::Image { .. },
        )
        | None => Size::ZERO,
        Some(WidgetKind::ListView { row_height, items, .. }) => {
            // Up to eight rows tall unless stretched or sized; wide enough to read.
            Size { width: 160.0, height: row_height * items.len().clamp(1, 8) as f32 }
        }
    };
    Size {
        width: known_dimensions.width.unwrap_or(content_size.width),
        height: known_dimensions.height.unwrap_or(content_size.height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameSlot;

    #[test]
    fn drop_zone_center_and_edges() {
        let rect = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
        assert_eq!(DropZone::classify(rect, 50.0, 50.0), DropZone::Center);
        assert_eq!(DropZone::classify(rect, 10.0, 50.0), DropZone::Left);
        assert_eq!(DropZone::classify(rect, 90.0, 50.0), DropZone::Right);
        assert_eq!(DropZone::classify(rect, 50.0, 10.0), DropZone::Top);
        assert_eq!(DropZone::classify(rect, 50.0, 90.0), DropZone::Bottom);
    }

    /// A region deliberately away from the origin (like a right-hand panel below a title row),
    /// so any mix-up between region-local and window coordinates shows up.
    const OFFSET: Rect = Rect { x: 612.0, y: 30.0, width: 388.0, height: 421.0 };

    #[test]
    fn drop_zone_offset_rect_each_zone() {
        let r = OFFSET;
        let at = |fx: f32, fy: f32| DropZone::classify(r, r.x + fx * r.width, r.y + fy * r.height);
        assert_eq!(at(0.5, 0.5), DropZone::Center);
        assert_eq!(at(0.05, 0.5), DropZone::Left);
        assert_eq!(at(0.20, 0.5), DropZone::Left);
        assert_eq!(at(0.95, 0.5), DropZone::Right);
        assert_eq!(at(0.80, 0.5), DropZone::Right);
        assert_eq!(at(0.5, 0.05), DropZone::Top);
        assert_eq!(at(0.5, 0.95), DropZone::Bottom);
        // Window-space coordinates that would be an edge of a rect at the origin are just the
        // middle of this one's band structure — never classified against the window.
        assert_eq!(DropZone::classify(r, 806.0, 240.0), DropZone::Center);
    }

    #[test]
    fn drop_zone_center_covers_middle_half_of_each_axis() {
        let r = OFFSET;
        for fx in [0.26, 0.4, 0.5, 0.6, 0.74] {
            for fy in [0.26, 0.4, 0.5, 0.6, 0.74] {
                let zone = DropZone::classify(r, r.x + fx * r.width, r.y + fy * r.height);
                assert_eq!(zone, DropZone::Center, "fx={fx} fy={fy}");
            }
        }
    }

    #[test]
    fn drop_zone_band_picks_nearest_edge() {
        let r = OFFSET;
        let at = |fx: f32, fy: f32| DropZone::classify(r, r.x + fx * r.width, r.y + fy * r.height);
        // In the top-right corner band: nearer the right edge than the top edge.
        assert_eq!(at(0.95, 0.15), DropZone::Right);
        // Nearer the top.
        assert_eq!(at(0.85, 0.05), DropZone::Top);
        assert_eq!(at(0.10, 0.90), DropZone::Left);
        assert_eq!(at(0.20, 0.97), DropZone::Bottom);
    }

    #[test]
    fn drop_zone_preview_rect_contains_cursor_and_is_the_committed_half() {
        let r = OFFSET;
        let mut fx = 0.0;
        while fx < 1.0 {
            let mut fy = 0.0;
            while fy < 1.0 {
                let (x, y) = (r.x + fx * r.width, r.y + fy * r.height);
                let zone = DropZone::classify(r, x, y);
                let preview = zone.preview_rect(r);
                assert!(preview.contains(x, y), "{zone:?} preview misses cursor at fx={fx} fy={fy}");
                fy += 0.03;
            }
            fx += 0.03;
        }
        let right = DropZone::Right.preview_rect(r);
        assert_eq!((right.x, right.y, right.width, right.height), (806.0, 30.0, 194.0, 421.0));
        let bottom = DropZone::Bottom.preview_rect(r);
        assert_eq!((bottom.x, bottom.y, bottom.width, bottom.height), (612.0, 240.5, 388.0, 210.5));
        let center = DropZone::Center.preview_rect(r);
        assert_eq!((center.x, center.width), (r.x, r.width));
    }

    #[test]
    fn close_hit_rect_covers_glyph_but_not_header() {
        let bar = Rect { x: 178.0, y: 427.0, width: 822.0, height: 28.0 };
        let drawn = close_button_rect(bar);
        let hit = close_hit_rect(bar);
        assert_eq!((drawn.x, drawn.width), (1000.0 - 22.0, 22.0));
        assert!(hit.x <= drawn.x && hit.x + hit.width >= drawn.x + drawn.width);
        assert!(hit.y <= drawn.y && hit.y + hit.height >= drawn.y + drawn.height);
        assert!(hit.contains(1000.0 - 26.0, 428.0), "a few px left of the glyph, top row");
        // A narrow tab segment: the × never takes more than 40% of it.
        let segment = Rect { x: 0.0, y: 0.0, width: 60.0, height: 28.0 };
        let hit = close_hit_rect(segment);
        assert!(hit.width <= 24.0 + f32::EPSILON);
        assert!(!hit.contains(30.0, 14.0), "segment middle must still select the tab");
        assert!(close_hit_rect(segment).contains(close_button_rect(segment).x, 14.0));
    }

    fn split_row_tree(ratio: f32, split_panes: bool) -> (WidgetTree, WidgetId, WidgetId, WidgetId) {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let row = tree.new_node(
            Style { flex_direction: FlexDirection::Row, flex_grow: 1.0, ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
        );
        let pane = |tree: &mut WidgetTree, grow: f32, text: &str| {
            let id = tree.new_node(
                Style { flex_direction: FlexDirection::Column, flex_grow: grow, ..Default::default() },
                WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
            );
            let label = tree.new_node(
                Style::default(),
                WidgetKind::Label { text: text.into(), font_size: 14.0, color: Color::TRANSPARENT },
            );
            tree.add_child(id, label);
            id
        };
        let first = pane(&mut tree, ratio * 1000.0, "");
        let bar = tree.new_node(
            Style { size: Size { width: Dimension::length(6.0), height: Dimension::auto() }, ..Default::default() },
            WidgetKind::Splitter {
                direction: SplitDirection::Row,
                ratio,
                bar_color: Color::TRANSPARENT,
                first,
                second: first,
            },
        );
        let second = pane(&mut tree, (1.0 - ratio) * 1000.0, "A fairly long label in the second pane");
        tree.mutate_kind(bar, |k| {
            if let WidgetKind::Splitter { second: s, .. } = k {
                *s = second;
            }
        });
        tree.add_child(root, row);
        tree.add_child(row, first);
        tree.add_child(row, bar);
        tree.add_child(row, second);
        if split_panes {
            tree.set_split_pane(first);
            tree.set_split_pane(second);
        }
        tree.compute_layout(1006.0, 400.0);
        (tree, first, bar, second)
    }

    #[test]
    fn split_pane_honours_ratio_despite_content() {
        let (tree, first, _, second) = split_row_tree(0.7, true);
        let a = tree.absolute_rect(first).unwrap().width;
        let b = tree.absolute_rect(second).unwrap().width;
        assert!((a - 700.0).abs() < 0.5 && (b - 300.0).abs() < 0.5, "got {a} / {b}");
        // Second pane (100) narrower than its label's min-content width (~290): ratio still wins.
        let (tree, first, _, second) = split_row_tree(0.9, true);
        let a = tree.absolute_rect(first).unwrap().width;
        let b = tree.absolute_rect(second).unwrap().width;
        assert!((a - 900.0).abs() < 0.5 && (b - 100.0).abs() < 0.5, "got {a} / {b}");
        // Regression guard for the bug itself: without `set_split_pane` content skews it.
        let (tree, first, _, _) = split_row_tree(0.7, false);
        assert!(tree.absolute_rect(first).unwrap().width < 690.0);
    }

    #[test]
    fn splitter_at_uses_slop_across_bar_only() {
        let (tree, _, bar, _) = split_row_tree(0.5, true);
        let r = tree.absolute_rect(bar).unwrap();
        let mid_y = r.y + r.height / 2.0;
        assert_eq!(tree.splitter_at(r.x + 3.0, mid_y, SPLITTER_HIT_SLOP), Some(bar));
        assert_eq!(tree.splitter_at(r.x - 4.0, mid_y, SPLITTER_HIT_SLOP), Some(bar));
        assert_eq!(tree.splitter_at(r.x + r.width + 4.0, mid_y, SPLITTER_HIT_SLOP), Some(bar));
        assert_eq!(tree.splitter_at(r.x - 6.0, mid_y, SPLITTER_HIT_SLOP), None);
        assert_eq!(tree.splitter_at(r.x + 3.0, r.y - 1.0, SPLITTER_HIT_SLOP), None);
    }

    #[test]
    fn drop_zone_corner_prefers_horizontal() {
        let rect = Rect { x: 0.0, y: 0.0, width: 100.0, height: 100.0 };
        // Equal distance to left and top — left is checked first.
        assert_eq!(DropZone::classify(rect, 5.0, 5.0), DropZone::Left);
    }

    #[test]
    fn hit_test_prefers_later_child() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let style = |w, h, x, y| Style {
            position: Position::Absolute,
            inset: taffy::prelude::Rect {
                left: LengthPercentageAuto::length(x),
                top: LengthPercentageAuto::length(y),
                right: LengthPercentageAuto::auto(),
                bottom: LengthPercentageAuto::auto(),
            },
            size: Size { width: Dimension::length(w), height: Dimension::length(h) },
            ..Default::default()
        };
        let back = tree.new_node(
            style(100.0, 100.0, 0.0, 0.0),
            WidgetKind::Label { text: "back".into(), font_size: 12.0, color: Color::TRANSPARENT },
        );
        let front = tree.new_node(
            style(50.0, 50.0, 10.0, 10.0),
            WidgetKind::Label { text: "front".into(), font_size: 12.0, color: Color::TRANSPARENT },
        );
        tree.add_child(root, back);
        tree.add_child(root, front);
        tree.compute_layout(200.0, 200.0);

        assert_eq!(tree.hit_test(20.0, 20.0), Some(front));
        assert_eq!(tree.hit_test(80.0, 80.0), Some(back));
        assert_eq!(tree.hit_test(150.0, 150.0), Some(root));
    }

    #[test]
    fn layout_fills_explicit_size() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let child = tree.new_node(
            Style {
                size: Size { width: Dimension::length(80.0), height: Dimension::length(40.0) },
                ..Default::default()
            },
            WidgetKind::Button {
                text: "Go".into(),
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                on_click: None,
            },
        );
        tree.add_child(root, child);
        tree.compute_layout(200.0, 100.0);
        let rect = tree.absolute_rect(child).expect("laid out");
        assert_eq!(rect.width, 80.0);
        assert_eq!(rect.height, 40.0);
    }

    #[test]
    fn find_region_at_matches_region_id() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let region = tree.new_node(
            Style {
                size: Size { width: Dimension::percent(1.0), height: Dimension::percent(1.0) },
                ..Default::default()
            },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: Some(7) },
        );
        tree.add_child(root, region);
        tree.compute_layout(100.0, 80.0);
        let found = tree.find_region_at(10.0, 10.0).expect("region");
        assert_eq!(found.0, 7);
        assert_eq!(found.1.width, 100.0);
        assert_eq!(found.1.height, 80.0);
    }

    #[test]
    fn find_region_at_skips_floating_overlay() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let docked = tree.new_node(
            Style {
                size: Size { width: Dimension::percent(1.0), height: Dimension::percent(1.0) },
                ..Default::default()
            },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: Some(1) },
        );
        let floating = tree.new_node(
            Style {
                position: Position::Absolute,
                inset: taffy::prelude::Rect {
                    left: LengthPercentageAuto::length(10.0),
                    top: LengthPercentageAuto::length(10.0),
                    right: LengthPercentageAuto::auto(),
                    bottom: LengthPercentageAuto::auto(),
                },
                size: Size { width: Dimension::length(50.0), height: Dimension::length(50.0) },
                ..Default::default()
            },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: Some(2) },
        );
        tree.add_child(root, docked);
        tree.add_child(root, floating);
        tree.compute_layout(100.0, 80.0);
        let found = tree.find_region_at(20.0, 20.0).expect("docked region under overlay");
        assert_eq!(found.0, 1);
    }

    fn focus_tree() -> (WidgetTree, WidgetId, WidgetId, WidgetId, WidgetId) {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let button = |tree: &mut WidgetTree| {
            tree.new_node(
                Style { size: Size { width: Dimension::length(40.0), height: Dimension::length(20.0) }, ..Default::default() },
                WidgetKind::Button {
                    text: "b".into(),
                    font_size: 12.0,
                    text_color: Color::TRANSPARENT,
                    background: Color::TRANSPARENT,
                    on_click: None,
                },
            )
        };
        let a = button(&mut tree);
        let label = tree.new_node(
            Style::default(),
            WidgetKind::Label { text: "x".into(), font_size: 12.0, color: Color::TRANSPARENT },
        );
        let hidden = tree.new_node(
            Style { display: Display::None, ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
        );
        let hidden_button = button(&mut tree);
        tree.add_child(hidden, hidden_button);
        let b = button(&mut tree);
        for child in [a, label, hidden, b] {
            tree.add_child(root, child);
        }
        tree.compute_layout(200.0, 200.0);
        (tree, a, label, hidden_button, b)
    }

    #[test]
    fn focus_next_cycles_visible_focusables_both_ways() {
        let (mut tree, a, _, _, b) = focus_tree();
        assert_eq!(tree.focus_next(false), Some(a));
        assert_eq!(tree.focus_next(false), Some(b), "skips the label and the hidden button");
        assert_eq!(tree.focus_next(false), Some(a), "wraps");
        assert_eq!(tree.focus_next(true), Some(b), "backward wraps");
        tree.set_focus(None);
        assert_eq!(tree.focus_next(true), Some(b), "backward from nothing starts at the end");
    }

    #[test]
    fn set_focus_ignores_unfocusable_and_reset_clears() {
        let (mut tree, a, label, _, _) = focus_tree();
        tree.set_focus(Some(a));
        tree.clear_dirty();
        tree.set_focus(Some(a));
        assert!(!tree.is_dirty(), "refocusing the same widget changes nothing");
        tree.set_focus(Some(label));
        assert_eq!(tree.focused(), None);
        assert!(tree.is_dirty());
        tree.set_focus(Some(a));
        tree.reset();
        assert_eq!(tree.focused(), None);
    }

    /// A 100×100 scroll area at (0,0) holding a column of ten 40-tall buttons (400 tall), and
    /// inside that a nested 100×60 scroll area as the first child holding 120 of content.
    fn scroll_tree() -> (WidgetTree, WidgetId, WidgetId, Vec<WidgetId>) {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let area_style = |h: f32| Style {
            overflow: taffy::geometry::Point { x: taffy::style::Overflow::Scroll, y: taffy::style::Overflow::Scroll },
            scrollbar_width: 0.0,
            size: Size { width: Dimension::length(100.0), height: Dimension::length(h) },
            flex_shrink: 0.0,
            ..Default::default()
        };
        let area = |tree: &mut WidgetTree, h: f32| {
            tree.new_node(
                area_style(h),
                WidgetKind::ScrollArea { offset: (0.0, 0.0), background: Color::TRANSPARENT, bar_color: Color::TRANSPARENT },
            )
        };
        let column = |tree: &mut WidgetTree| {
            tree.new_node(
                Style { flex_direction: FlexDirection::Column, flex_shrink: 0.0, ..Default::default() },
                WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
            )
        };
        let button = |tree: &mut WidgetTree, h: f32| {
            tree.new_node(
                Style { size: Size { width: Dimension::length(80.0), height: Dimension::length(h) }, flex_shrink: 0.0, ..Default::default() },
                WidgetKind::Button {
                    text: String::new(),
                    font_size: 12.0,
                    text_color: Color::TRANSPARENT,
                    background: Color::TRANSPARENT,
                    on_click: None,
                },
            )
        };
        let outer = area(&mut tree, 100.0);
        let outer_column = column(&mut tree);
        let inner = area(&mut tree, 60.0);
        let inner_column = column(&mut tree);
        let inner_button = button(&mut tree, 120.0);
        tree.add_child(root, outer);
        tree.add_child(outer, outer_column);
        tree.add_child(outer_column, inner);
        tree.add_child(inner, inner_column);
        tree.add_child(inner_column, inner_button);
        let buttons: Vec<_> = (0..10).map(|_| button(&mut tree, 40.0)).collect();
        for &b in &buttons {
            tree.add_child(outer_column, b);
        }
        tree.compute_layout(300.0, 300.0);
        (tree, outer, inner, buttons)
    }

    #[test]
    fn scroll_area_offsets_children_and_clamps() {
        let (mut tree, outer, _, buttons) = scroll_tree();
        assert_eq!(tree.scroll_extent(outer), Some((0.0, 360.0)), "60 + 10×40 content in 100");
        assert!(tree.set_scroll_offset(outer, 0.0, 100.0));
        tree.compute_layout(300.0, 300.0);
        assert_eq!(tree.absolute_rect(buttons[0]).unwrap().y, 60.0 - 100.0);
        assert!(tree.set_scroll_offset(outer, 50.0, 1e6), "clamps rather than refusing");
        let Some(WidgetKind::ScrollArea { offset, .. }) = tree.kind(outer) else { panic!() };
        assert_eq!(*offset, (0.0, 360.0));
        assert!(!tree.set_scroll_offset(outer, 0.0, 999.0), "already at the end");
    }

    #[test]
    fn clipped_content_is_not_hit_and_clip_rect_is_the_area() {
        let (mut tree, outer, _, buttons) = scroll_tree();
        // buttons[1] spans y 100..140 unscrolled — entirely below the 100-tall area.
        assert_eq!(tree.hit_test(40.0, 120.0), Some(tree.root()));
        assert_eq!(tree.clip_rect(buttons[1]).map(|c| (c.y, c.height)), Some((0.0, 100.0)));
        tree.set_scroll_offset(outer, 0.0, 60.0);
        tree.compute_layout(300.0, 300.0);
        assert_eq!(tree.hit_test(40.0, 50.0), Some(buttons[1]));
        assert!(tree.clip_rect(outer).is_none(), "the area itself isn't clipped");
    }

    #[test]
    fn wheel_scrolls_innermost_area_then_hands_off() {
        let (mut tree, outer, inner, _) = scroll_tree();
        let offset = |tree: &WidgetTree, id| match tree.kind(id) {
            Some(WidgetKind::ScrollArea { offset, .. }) => offset.1,
            _ => panic!(),
        };
        assert!(tree.scroll_at(10.0, 10.0, 0.0, 50.0));
        assert_eq!((offset(&tree, inner), offset(&tree, outer)), (50.0, 0.0), "inner first");
        tree.compute_layout(300.0, 300.0);
        assert!(tree.scroll_at(10.0, 10.0, 0.0, 50.0));
        assert_eq!((offset(&tree, inner), offset(&tree, outer)), (60.0, 0.0), "inner reaches its end (60)");
        assert!(tree.scroll_at(10.0, 10.0, 0.0, 50.0));
        assert_eq!(offset(&tree, outer), 50.0, "then the outer one scrolls");
        assert!(!tree.scroll_at(250.0, 250.0, 0.0, 50.0), "nothing to scroll out there");
    }

    #[test]
    fn scroll_into_view_reveals_focused_widget() {
        let (mut tree, outer, _, buttons) = scroll_tree();
        tree.scroll_into_view(buttons[5]); // content y 260..300
        tree.compute_layout(300.0, 300.0);
        let rect = tree.absolute_rect(buttons[5]).unwrap();
        assert_eq!((rect.y, rect.y + rect.height), (60.0, 100.0), "bottom-aligned, the minimum scroll");
        tree.scroll_into_view(buttons[0]);
        tree.compute_layout(300.0, 300.0);
        assert_eq!(tree.absolute_rect(buttons[0]).unwrap().y, 0.0);
        let Some(WidgetKind::ScrollArea { offset, .. }) = tree.kind(outer) else { panic!() };
        assert_eq!(offset.1, 60.0);
    }

    #[test]
    fn scrollbar_thumb_tracks_offset() {
        let (mut tree, outer, _, _) = scroll_tree();
        let (v, h) = tree.scrollbar_thumbs(outer);
        assert!(h.is_none(), "no horizontal overflow");
        let v = v.unwrap();
        assert_eq!((v.y, v.width), (SCROLLBAR_INSET, SCROLLBAR_WIDTH));
        assert_eq!(v.height, SCROLLBAR_MIN_THUMB, "96 × 100/460 ≈ 21 is below the minimum");
        tree.set_scroll_offset(outer, 0.0, 360.0);
        let end = tree.scrollbar_thumbs(outer).0.unwrap();
        assert!((end.y + end.height - (100.0 - SCROLLBAR_INSET)).abs() < 1e-3, "thumb at the bottom of the track");
    }

    #[test]
    fn dragging_thumb_maps_back_to_offset() {
        let (mut tree, outer, inner, _) = scroll_tree();
        // The nested area is as wide as the outer one, so its thumb shares the column; move it
        // down out of the way (where both overlap, the inner one wins, as for the wheel).
        let inner_thumb = tree.scrollbar_thumbs(inner).0.unwrap();
        assert_eq!(tree.scrollbar_at(inner_thumb.x + 1.0, inner_thumb.y + 5.0, 3.0), Some((inner, true)));
        tree.set_scroll_offset(inner, 0.0, 60.0);
        let thumb = tree.scrollbar_thumbs(outer).0.unwrap();
        assert_eq!(tree.scrollbar_at(thumb.x + 1.0, thumb.y + 5.0, 3.0), Some((outer, true)));
        assert_eq!(tree.scrollbar_at(thumb.x - 2.0, thumb.y + 5.0, 3.0), Some((outer, true)), "slop");
        assert_eq!(tree.scrollbar_at(thumb.x + 1.0, 90.0, 3.0), None, "the track below the thumb isn't a grab");
        // Halfway along the travel is half the extent.
        let travel = 100.0 - 2.0 * SCROLLBAR_INSET - thumb.height;
        assert!(tree.drag_scrollbar(outer, true, thumb.y + travel / 2.0));
        let Some(WidgetKind::ScrollArea { offset, .. }) = tree.kind(outer) else { panic!() };
        assert!((offset.1 - 180.0).abs() < 1e-3, "{offset:?}");
        assert!(tree.drag_scrollbar(outer, true, 1e6));
        assert_eq!(tree.scrollbar_thumbs(outer).0.map(|t| t.y + t.height), Some(100.0 - SCROLLBAR_INSET));
    }

    fn popup_kind(anchor: PopupAnchor, modal: bool, dismissed: Option<Arc<std::sync::atomic::AtomicUsize>>) -> WidgetKind {
        WidgetKind::Popup {
            anchor,
            modal,
            background: Color::TRANSPARENT,
            border: Color::TRANSPARENT,
            on_dismiss: dismissed.map(|count| {
                Arc::new(move || {
                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }) as ClickCallback
            }),
            restore_focus: None,
            open: Some(Readback::new(false)),
        }
    }

    fn sized_button(tree: &mut WidgetTree, w: f32, h: f32) -> WidgetId {
        tree.new_node(
            Style { size: Size { width: Dimension::length(w), height: Dimension::length(h) }, flex_shrink: 0.0, ..Default::default() },
            WidgetKind::Button {
                text: String::new(),
                font_size: 12.0,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                on_click: None,
            },
        )
    }

    /// A 200×200 window with an anchor button at `(x, y)` (60×20) via margins.
    fn popup_tree(x: f32, y: f32) -> (WidgetTree, WidgetId) {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let anchor = sized_button(&mut tree, 60.0, 20.0);
        let mut style = tree.style(anchor).unwrap().clone();
        style.margin = taffy::prelude::Rect {
            left: LengthPercentageAuto::length(x),
            top: LengthPercentageAuto::length(y),
            right: LengthPercentageAuto::length(0.0),
            bottom: LengthPercentageAuto::length(0.0),
        };
        tree.set_style(anchor, style);
        tree.add_child(root, anchor);
        tree.compute_layout(200.0, 200.0);
        (tree, anchor)
    }

    fn open_menu(tree: &mut WidgetTree, anchor: PopupAnchor, modal: bool, items: usize) -> (WidgetId, Vec<WidgetId>) {
        let mut ids = Vec::new();
        let popup = tree.open_popup(popup_kind(anchor, modal, None), |tree, popup| {
            for _ in 0..items {
                let b = sized_button(tree, 50.0, 30.0);
                tree.add_child(popup, b);
                ids.push(b);
            }
        });
        tree.compute_layout(200.0, 200.0);
        (popup, ids)
    }

    #[test]
    fn popup_opens_below_its_anchor_and_flips_or_clamps() {
        let (mut tree, anchor) = popup_tree(10.0, 20.0);
        let (popup, items) = open_menu(&mut tree, PopupAnchor::Widget(anchor, PopupSide::Below), false, 2);
        let r = tree.absolute_rect(popup).unwrap();
        assert_eq!((r.x, r.y, r.width, r.height), (10.0, 20.0 + 20.0 + POPUP_GAP, 50.0, 60.0));
        assert_eq!(tree.absolute_rect(items[1]).unwrap().y, r.y + 30.0, "content moves with it");

        let (mut tree, anchor) = popup_tree(170.0, 150.0);
        let (popup, _) = open_menu(&mut tree, PopupAnchor::Widget(anchor, PopupSide::Below), false, 2);
        let r = tree.absolute_rect(popup).unwrap();
        assert_eq!(r.y, 150.0 - POPUP_GAP - 60.0, "no room below: flips above");
        assert_eq!(r.x, 200.0 - POPUP_GAP - 50.0, "clamped inside the window");

        let (popup, _) = open_menu(&mut tree, PopupAnchor::Center, true, 1);
        let r = tree.absolute_rect(popup).unwrap();
        assert_eq!((r.x, r.y), (75.0, 85.0));
    }

    #[test]
    fn outside_press_dismisses_non_modal_but_not_modal() {
        let dismissed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (mut tree, anchor) = popup_tree(10.0, 20.0);
        tree.set_focus(Some(anchor));
        let popup = tree.open_popup(
            popup_kind(PopupAnchor::Widget(anchor, PopupSide::Below), false, Some(dismissed.clone())),
            |tree, popup| {
                let b = sized_button(tree, 50.0, 30.0);
                tree.add_child(popup, b);
            },
        );
        tree.compute_layout(200.0, 200.0);
        let Some(WidgetKind::Popup { open: Some(open), .. }) = tree.kind(popup) else { panic!() };
        let open = open.clone();
        assert!(open.get());
        assert_ne!(tree.focused(), Some(anchor), "focus moved into the popup");
        assert_eq!(tree.popup_press(20.0, 50.0), PopupPress::Pass, "inside the popup");
        assert_eq!(tree.popup_press(150.0, 150.0), PopupPress::Consumed);
        assert_eq!(dismissed.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(!open.get() && tree.topmost_popup().is_none() && tree.kind(popup).is_none());
        assert_eq!(tree.focused(), Some(anchor), "focus restored");
        assert_eq!(tree.popup_press(150.0, 150.0), PopupPress::Pass, "nothing open");

        let (modal, _) = open_menu(&mut tree, PopupAnchor::Center, true, 1);
        assert_eq!(tree.popup_press(1.0, 1.0), PopupPress::Consumed);
        assert_eq!(tree.topmost_popup(), Some(modal), "a modal popup stays open");
        tree.close_popup(modal);
        tree.compute_layout(200.0, 200.0);
        assert_eq!(tree.hit_test(20.0, 25.0), Some(anchor), "gone from hit testing");
    }

    #[test]
    fn tab_stays_inside_the_topmost_popup() {
        let (mut tree, anchor) = popup_tree(10.0, 20.0);
        let (_, items) = open_menu(&mut tree, PopupAnchor::Widget(anchor, PopupSide::Below), false, 2);
        assert_eq!(tree.focused(), Some(items[0]));
        assert_eq!(tree.focus_next(false), Some(items[1]));
        assert_eq!(tree.focus_next(false), Some(items[0]), "wraps within the popup, skipping the anchor");
        assert_eq!(tree.popup_of(items[1]), tree.topmost_popup());
        assert_eq!(tree.popup_of(anchor), None);
    }

    /// Every `on_select` call a test list received.
    type Selections = Arc<std::sync::Mutex<Vec<usize>>>;

    /// A 100-tall list of `n` 20-tall rows at (0, 0).
    fn list_tree(n: usize) -> (WidgetTree, WidgetId, Selections, Readback<Option<usize>>) {
        let selections = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = selections.clone();
        let mirror = Readback::new(None);
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let list = tree.new_node(
            Style { size: Size { width: Dimension::length(150.0), height: Dimension::length(100.0) }, ..Default::default() },
            WidgetKind::ListView {
                items: (0..n).map(|i| format!("item {i}")).collect(),
                row_height: 20.0,
                font_size: 14.0,
                scroll: 0.0,
                selected: None,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                on_select: Some(Arc::new(move |i| sink.lock().unwrap().push(i))),
                on_activate: None,
                mirror: Some(mirror.clone()),
            },
        );
        tree.add_child(root, list);
        tree.compute_layout(300.0, 300.0);
        (tree, list, selections, mirror)
    }

    #[test]
    fn list_view_draws_only_visible_rows_of_a_million() {
        let (mut tree, list, _, _) = list_tree(1_000_000);
        assert_eq!(tree.list_visible_rows(list), 0..5);
        assert_eq!(tree.scroll_extent(list), Some((0.0, 20_000_000.0 - 100.0)));
        assert!(tree.scroll_at(10.0, 10.0, 0.0, 30.0), "the wheel scrolls it like a ScrollArea");
        assert_eq!(tree.list_visible_rows(list), 1..7, "partly visible rows at both ends");
        assert_eq!(tree.list_row_at(list, 5.0), Some(1));
        tree.set_scroll_offset(list, 0.0, 1e12);
        tree.compute_layout(300.0, 300.0);
        assert_eq!(tree.list_visible_rows(list), 999_995..1_000_000);
        assert!(tree.scrollbar_thumbs(list).0.is_some(), "it has a draggable scrollbar");
    }

    #[test]
    fn list_view_selection_scrolls_into_view_and_notifies() {
        let (mut tree, list, selections, mirror) = list_tree(50);
        tree.list_select(list, Some(10));
        assert_eq!(tree.scroll_offset(list), Some((0.0, 11.0 * 20.0 - 100.0)), "row 10 bottom-aligned");
        tree.list_select(list, Some(10));
        tree.list_select(list, Some(2));
        assert_eq!(tree.scroll_offset(list), Some((0.0, 40.0)), "row 2 top-aligned");
        tree.list_select(list, Some(999));
        assert_eq!(*selections.lock().unwrap(), [10, 2, 49], "reselecting fires nothing; clamps to the last row");
        assert_eq!(mirror.get(), Some(49));
        assert_eq!(tree.list_page_rows(list), 5);
        let (mut empty, id, _, _) = list_tree(0);
        empty.list_select(id, Some(0));
        assert_eq!(empty.list_row_at(id, 5.0), None);
    }

    #[test]
    fn dock_region_clips_overflowing_content() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let region = tree.new_node(
            Style { size: Size { width: Dimension::length(100.0), height: Dimension::length(50.0) }, ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: Some(3) },
        );
        let wide = sized_button(&mut tree, 300.0, 20.0);
        tree.add_child(root, region);
        tree.add_child(region, wide);
        tree.compute_layout(400.0, 100.0);
        assert_eq!(tree.clip_rect(wide).map(|c| (c.x, c.width)), Some((0.0, 100.0)));
        assert_eq!(tree.hit_test(50.0, 10.0), Some(wide));
        assert_ne!(tree.hit_test(250.0, 10.0), Some(wide), "the overflowing part isn't clickable");
        assert!(tree.clip_rect(region).is_none(), "a region isn't clipped by itself");
    }

    #[test]
    fn wrapping_row_stays_inside_a_narrow_pane() {
        // Like theme_demo's left pane: a region holding a column holding a wrapping row of
        // buttons whose total width is well over the pane's.
        let (mut tree, first, _, _) = split_row_tree(0.3, true);
        let region = tree.new_node(
            Style { flex_direction: FlexDirection::Column, flex_grow: 1.0, ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: Some(9) },
        );
        let row = tree.new_node(
            Style {
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                gap: Size { width: LengthPercentage::length(4.0), height: LengthPercentage::length(4.0) },
                ..Default::default()
            },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
        );
        tree.add_child(first, region);
        tree.add_child(region, row);
        let buttons: Vec<_> = (0..5).map(|_| sized_button(&mut tree, 110.0, 30.0)).collect();
        for &b in &buttons {
            tree.add_child(row, b);
        }
        tree.compute_layout(1006.0, 400.0);
        let pane = tree.absolute_rect(first).unwrap();
        let rects: Vec<Rect> = buttons.iter().map(|&b| tree.absolute_rect(b).unwrap()).collect();
        for r in &rects {
            assert!(r.x >= pane.x && r.x + r.width <= pane.x + pane.width + 0.5, "{r:?} outside {pane:?}");
        }
        let lines = {
            let mut ys: Vec<i32> = rects.iter().map(|r| r.y as i32).collect();
            ys.dedup();
            ys.len()
        };
        assert!(lines >= 2, "5 × 110 in a ~300-wide pane wraps onto several lines");
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                assert!(!a.intersects(b), "wrapped buttons don't overlap: {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn reset_never_reuses_widget_ids() {
        let (mut tree, a, label, hidden_button, b) = focus_tree();
        let old: Vec<WidgetId> = tree.walk().collect();
        tree.reset();
        let (root, mut fresh) = (tree.root(), Vec::new());
        for _ in 0..old.len() + 4 {
            let id = tree.new_node(Style::default(), WidgetKind::Container { background: Color::TRANSPARENT, region_id: None });
            tree.add_child(root, id);
            fresh.push(id);
        }
        fresh.push(root);
        for id in [a, label, hidden_button, b].into_iter().chain(old) {
            assert!(!fresh.contains(&id), "stale id {id:?} aliases a node of the new tree");
            assert!(tree.kind(id).is_none());
            // A stale widget's `set_text` / a closed popup's `close()` reach nothing.
            tree.mutate_kind(id, |_| panic!("mutating a stale id reaches nothing"));
            tree.close_popup(id);
        }
    }

    #[test]
    fn scroll_into_view_through_nested_areas_lands_inside_both() {
        let (mut tree, outer, inner, _) = scroll_tree();
        // The inner area's only child: 120 tall inside a 60-tall area at the top of the outer one.
        let inner_child = tree.walk_from(inner).nth(2).unwrap();
        // Scroll the inner area to its end and the outer one so the inner sits partly above it:
        // revealing the target moves both, and the outer must use where the inner put it.
        tree.set_scroll_offset(inner, 0.0, 60.0);
        tree.set_scroll_offset(outer, 0.0, 40.0);
        tree.compute_layout(300.0, 300.0);
        tree.scroll_into_view(inner_child);
        tree.compute_layout(300.0, 300.0);
        let target = tree.absolute_rect(inner_child).unwrap();
        let (o, i) = (tree.absolute_rect(outer).unwrap(), tree.absolute_rect(inner).unwrap());
        assert!(target.y >= o.y - 0.01 && target.y >= i.y - 0.01, "target top {target:?} is visible in outer {o:?} and inner {i:?}");
        assert_eq!((tree.scroll_offset(inner), tree.scroll_offset(outer)), (Some((0.0, 0.0)), Some((0.0, 0.0))));
    }

    #[test]
    fn scroll_into_view_does_not_overshoot_after_the_inner_area_scrolls() {
        // Outer 100-tall area: 200 of spacer, then a 60-tall inner area (five 40-tall buttons,
        // 200 of content), then 300 more spacer. Target: the inner's 4th button (content y 120).
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let area = |tree: &mut WidgetTree, h: f32| {
            tree.new_node(
                Style {
                    flex_direction: FlexDirection::Column,
                    size: Size { width: Dimension::length(100.0), height: Dimension::length(h) },
                    flex_shrink: 0.0,
                    ..Default::default()
                },
                WidgetKind::ScrollArea { offset: (0.0, 0.0), background: Color::TRANSPARENT, bar_color: Color::TRANSPARENT },
            )
        };
        let column = |tree: &mut WidgetTree| {
            tree.new_node(
                Style { flex_direction: FlexDirection::Column, flex_shrink: 0.0, ..Default::default() },
                WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
            )
        };
        let outer = area(&mut tree, 100.0);
        let outer_column = column(&mut tree);
        let inner = area(&mut tree, 60.0);
        let inner_column = column(&mut tree);
        tree.add_child(root, outer);
        tree.add_child(outer, outer_column);
        let top = sized_button(&mut tree, 80.0, 200.0);
        tree.add_child(outer_column, top);
        tree.add_child(outer_column, inner);
        let bottom = sized_button(&mut tree, 80.0, 300.0);
        tree.add_child(outer_column, bottom);
        tree.add_child(inner, inner_column);
        let buttons: Vec<_> = (0..5).map(|_| sized_button(&mut tree, 80.0, 40.0)).collect();
        for &b in &buttons {
            tree.add_child(inner_column, b);
        }
        tree.set_scroll_container(outer);
        tree.set_scroll_container(inner);
        tree.compute_layout(300.0, 300.0);

        tree.scroll_into_view(buttons[3]);
        tree.compute_layout(300.0, 300.0);
        let target = tree.absolute_rect(buttons[3]).unwrap();
        let (o, i) = (tree.absolute_rect(outer).unwrap(), tree.absolute_rect(inner).unwrap());
        assert!(
            target.y >= o.y && target.y + target.height <= o.y + o.height,
            "target {target:?} inside the outer area {o:?}"
        );
        assert!(target.y >= i.y && target.y + target.height <= i.y + i.height, "and the inner area {i:?}");
        assert_eq!(tree.scroll_offset(inner), Some((0.0, 100.0)));
        assert_eq!(tree.scroll_offset(outer), Some((0.0, 160.0)), "not 260: the inner scroll is accounted for");
    }

    #[test]
    fn measured_layout_sizes_labels_by_real_text_width() {
        /// "Shaping" where every character is 20 wide — far wider than the 0.55 × 14 = 7.7
        /// the estimate assumes — so the result shows which one layout used.
        struct Wide;
        impl TextMeasure for Wide {
            fn caret_x(&mut self, text: &str, _: f32, index: usize) -> f32 {
                text[..index].chars().count() as f32 * 20.0
            }
            fn index_at(&mut self, _: &str, _: f32, _: f32) -> usize {
                0
            }
        }
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let row = tree.new_node(
            Style { flex_direction: FlexDirection::Row, align_items: Some(AlignItems::FLEX_START), ..Default::default() },
            WidgetKind::Container { background: Color::TRANSPARENT, region_id: None },
        );
        let label = |tree: &mut WidgetTree, text: &str| {
            tree.new_node(Style::default(), WidgetKind::Label { text: text.into(), font_size: 14.0, color: Color::TRANSPARENT })
        };
        let one = label(&mut tree, "Modal");
        let two = label(&mut tree, "ab\nlonger");
        tree.add_child(root, row);
        tree.add_child(row, one);
        tree.add_child(row, two);
        tree.compute_layout_measured(400.0, 200.0, &mut Wide);
        assert_eq!(tree.absolute_rect(one).unwrap().width, 100.0, "5 × 20, not the 5 × 7.7 estimate");
        let r = tree.absolute_rect(two).unwrap();
        assert_eq!(r.width, 120.0, "the widest line");
        assert!((r.height - 14.0 * LINE_HEIGHT_RATIO * 2.0).abs() <= 0.5, "two lines tall (taffy rounds): {r:?}");
    }

    #[test]
    fn viewport_kind_shares_frame_slot() {
        let slot = FrameSlot::new();
        slot.submit(crate::CpuFrame {
            width: 1,
            height: 1,
            format: crate::PixelFormat::Rgba8,
            data: vec![1, 2, 3, 4],
        });
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            Style {
                size: Size { width: Dimension::percent(1.0), height: Dimension::percent(1.0) },
                ..Default::default()
            },
            WidgetKind::Viewport { viewport_id: 1, frames: slot.clone() },
        );
        tree.add_child(root, id);
        tree.compute_layout(64.0, 32.0);
        let rect = tree.absolute_rect(id).expect("laid out");
        assert_eq!(rect.width, 64.0);
        assert_eq!(rect.height, 32.0);
        let WidgetKind::Viewport { frames, .. } = tree.kind(id).unwrap() else { panic!("kind") };
        let frame = frames.take_latest().expect("frame");
        assert_eq!(frame.data, vec![1, 2, 3, 4]);
    }

    /// Intrinsic measure without the root column stretching the cross axis.
    fn measure_style() -> Style {
        Style { align_self: Some(AlignSelf::START), ..Default::default() }
    }

    #[test]
    fn checkbox_measure_includes_box_and_label() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            measure_style(),
            WidgetKind::Checkbox {
                checked: false,
                label: "On".into(),
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                box_color: Color::TRANSPARENT,
                check_color: Color::TRANSPARENT,
                on_change: None,
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 200.0);
        let rect = tree.absolute_rect(id).expect("laid out");
        let text = measure_text("On", 14.0);
        // taffy may round final layout sizes; measure only has to land nearby.
        assert!((rect.width - (CHECK_SIZE + CHECK_LABEL_GAP + text.width)).abs() < 1.0);
        assert!((rect.height - CHECK_SIZE.max(text.height)).abs() < 1.0);
        assert!(tree.kind(id).unwrap().is_focusable());
    }

    #[test]
    fn toggle_and_spinbox_intrinsic_sizes() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let toggle = tree.new_node(
            measure_style(),
            WidgetKind::Toggle {
                checked: false,
                track_off: Color::TRANSPARENT,
                track_on: Color::TRANSPARENT,
                thumb_color: Color::TRANSPARENT,
                on_change: None,
            },
        );
        let spin = tree.new_node(
            measure_style(),
            WidgetKind::SpinBox {
                value: 0.0,
                min: 0.0,
                max: 10.0,
                step: 1.0,
                decimals: 0,
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                button_color: Color::TRANSPARENT,
                on_change: None,
                mirror: None,
            },
        );
        tree.add_child(root, toggle);
        tree.add_child(root, spin);
        tree.compute_layout(400.0, 200.0);
        let t = tree.absolute_rect(toggle).expect("toggle");
        assert!((t.width - TOGGLE_WIDTH).abs() < 1.0);
        assert!((t.height - TOGGLE_HEIGHT).abs() < 1.0);
        let s = tree.absolute_rect(spin).expect("spin");
        assert!((s.width - 100.0).abs() < 1.0);
        let expected_h = 14.0 * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING;
        assert!((s.height - expected_h).abs() < 1.0);
    }

    #[test]
    fn combo_box_measure_and_focusable() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let id = tree.new_node(
            measure_style(),
            WidgetKind::ComboBox {
                items: vec!["a".into(), "b".into()],
                selected: None,
                placeholder: "Pick…".into(),
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                border: Color::TRANSPARENT,
                on_change: None,
                mirror: None,
                popup_id: None,
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 200.0);
        let rect = tree.absolute_rect(id).expect("laid out");
        assert!((rect.width - 160.0).abs() < 1.0);
        let expected_h = 14.0 * LINE_HEIGHT_RATIO + 2.0 * TEXT_INPUT_VERTICAL_PADDING;
        assert!((rect.height - expected_h).abs() < 1.0);
        assert!(tree.kind(id).unwrap().is_focusable());
    }

    #[test]
    fn text_area_wheels_and_shows_a_scrollbar_when_it_overflows() {
        let mut tree = WidgetTree::new();
        let root = tree.root();
        let lines = (0..20).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n");
        let id = tree.new_node(
            Style {
                size: Size { width: Dimension::length(200.0), height: Dimension::length(80.0) },
                ..Default::default()
            },
            WidgetKind::TextArea {
                edit: TextEdit::new_multiline(&lines),
                placeholder: String::new(),
                font_size: 14.0,
                text_color: Color::TRANSPARENT,
                placeholder_color: Color::TRANSPARENT,
                background: Color::TRANSPARENT,
                selection_color: Color::TRANSPARENT,
                scroll_y: 0.0,
                preedit: None,
                on_change: None,
                on_submit: None,
                mirror: None,
            },
        );
        tree.add_child(root, id);
        tree.compute_layout(400.0, 300.0);
        let max = tree.scroll_extent(id).expect("text area scrolls").1;
        assert!(max > 0.0, "twenty lines in an 80-tall field must overflow");
        assert_eq!(tree.scroll_offset(id), Some((0.0, 0.0)));
        assert!(tree.scrollbar_thumbs(id).0.is_some(), "overflow draws a vertical thumb");
        assert!(tree.scroll_at(10.0, 10.0, 0.0, 40.0));
        let Some((_, y)) = tree.scroll_offset(id) else { panic!() };
        assert!((y - 40.0).abs() < 0.5, "wheel moved scroll_y to {y}");
        assert!(tree.set_scroll_offset(id, 0.0, 1e6));
        assert_eq!(tree.scroll_offset(id), Some((0.0, max)));
    }
}
