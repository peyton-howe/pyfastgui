use std::collections::HashMap;
use std::sync::Arc;

use taffy::prelude::*;

use crate::FrameSlot;

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
pub type TabSelectCallback = Arc<dyn Fn(usize) + Send + Sync>;
/// `(dragged_region_id, target_region_id, zone, float_rect)`.
/// `float_rect` is `Some((x, y, width, height))` in main-window client coords when `zone` is
/// `Float` (tear a docked panel out into an OS window); `None` for ordinary dock rearrange.
pub type PanelDropCallback =
    Arc<dyn Fn(u64, u64, DropZone, Option<(f32, f32, f32, f32)>) + Send + Sync>;
/// Fired when the user clicks a panel/tab close control — argument is that panel's region id.
pub type PanelCloseCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Width/height of the × hit target in a title bar or tab segment (layout units).
pub const CLOSE_BUTTON_SIZE: f32 = 22.0;

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
    /// Classify a point against `rect` for drag-and-drop purposes: the middle 50% (by area, via
    /// independent per-axis margins) is `Center`, the outer 25% margin on whichever axis the
    /// point is furthest out on picks an edge. Shared by hover-preview and drop-commit so they
    /// always agree on the same zone for the same cursor position.
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
    /// `first`/`second`'s `flex_grow` to match — see `fastgui-render-vk::app`'s
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
    /// other wrapper's to `Display::None` — see `fastgui-render-vk::app`'s tab click handling.
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
    /// an identity: dragging this bar (`fastgui-render-vk::app`'s panel-drag handling) and
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
    /// A GPU/CPU image rect composited on top of chrome. `frames` is the same latest-wins
    /// mailbox `Viewport.submit_frame` writes; `viewport_id` is a stable identity for the
    /// renderer's GPU texture cache across tree rebuilds (unlike `WidgetId`).
    Viewport {
        viewport_id: u64,
        frames: FrameSlot<crate::CpuFrame>,
    },
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
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

/// Close-button hit/draw rect on the right of a title bar or tab segment.
pub fn close_button_rect(bar: Rect) -> Rect {
    let size = CLOSE_BUTTON_SIZE.min(bar.height).min(bar.width.max(0.0) * 0.5);
    Rect {
        x: bar.x + bar.width - size,
        y: bar.y + (bar.height - size) * 0.5,
        width: size,
        height: size,
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
        Self { taffy, kinds, root, dirty: true, absolute_rects: HashMap::new() }
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
        self.taffy = TaffyTree::new();
        self.kinds.clear();
        self.root = self.taffy.new_leaf(root_style()).expect("creating the root node cannot fail");
        self.kinds.insert(self.root, WidgetKind::Container { background: Color::TRANSPARENT, region_id: None });
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

    /// Used by `Splitter` drag-resize (`fastgui-render-vk::app::update_dragged_splitter`) to
    /// resize a pane without needing to reconstruct its whole `Style` at the call site.
    pub fn set_flex_grow(&mut self, id: WidgetId, flex_grow: f32) {
        let Some(mut style) = self.taffy.style(id).ok().cloned() else { return };
        style.flex_grow = flex_grow;
        let _ = self.taffy.set_style(id, style);
        self.mark_dirty();
    }

    /// Used by `TabBar` click handling (`fastgui-render-vk::app`) to show/hide a tab's content
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
        let available = Size { width: AvailableSpace::Definite(width), height: AvailableSpace::Definite(height) };
        let kinds = &self.kinds;
        let _ = self.taffy.compute_layout_with_measure(
            self.root,
            available,
            |known_dimensions, _available_space, node_id, _node_context, _style| {
                measure_leaf(known_dimensions, node_id, kinds)
            },
        );

        self.absolute_rects.clear();
        let mut stack = vec![(self.root, 0.0f32, 0.0f32)];
        while let Some((id, parent_x, parent_y)) = stack.pop() {
            let Ok(layout) = self.taffy.layout(id) else { continue };
            let x = parent_x + layout.location.x;
            let y = parent_y + layout.location.y;
            self.absolute_rects.insert(id, Rect { x, y, width: layout.size.width, height: layout.size.height });
            if let Ok(children) = self.taffy.children(id) {
                for child in children {
                    stack.push((child, x, y));
                }
            }
        }
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
        self.walk().filter(|&id| self.absolute_rect(id).is_some_and(|r| r.contains(x, y))).last()
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

impl Default for WidgetTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Very rough text-extent estimate (average proportional-font advance width), used only to
/// give `Label`/`Button` a sane intrinsic size for `taffy`'s layout pass. Not a substitute for
/// real shaping — `fastgui-chrome` does that with `cosmic-text` when it actually rasterizes —
/// this only has to be good enough that layout doesn't look broken before that happens. Good
/// enough for M4; swap for measuring through `fastgui-chrome`'s font system if/when this
/// approximation visibly matters (e.g. non-Latin scripts, tight-fitting layouts).
fn measure_text(text: &str, font_size: f32) -> Size<f32> {
    const AVG_ADVANCE_RATIO: f32 = 0.55;
    const LINE_HEIGHT_RATIO: f32 = 1.3;
    let width = text.chars().count() as f32 * font_size * AVG_ADVANCE_RATIO;
    Size { width, height: font_size * LINE_HEIGHT_RATIO }
}

/// taffy calls this once per leaf node during layout. `known_dimensions` is already `Some` for
/// any axis the node's own style pins (explicit size, or a parent that stretched it) — we only
/// need to supply the *intrinsic content* size for whichever axes are still `None`.
fn measure_leaf(
    known_dimensions: Size<Option<f32>>,
    node_id: NodeId,
    kinds: &HashMap<WidgetId, WidgetKind>,
) -> Size<f32> {
    let content_size = match kinds.get(&node_id) {
        Some(WidgetKind::Label { text, font_size, .. }) => measure_text(text, *font_size),
        Some(WidgetKind::Button { text, font_size, .. }) => {
            let text_size = measure_text(text, *font_size);
            // Room for the button's own padding beyond the text itself; real padding is
            // applied via the node's taffy `Style`, this is just the intrinsic minimum.
            const BUTTON_PADDING: f32 = 16.0;
            Size { width: text_size.width + BUTTON_PADDING, height: text_size.height + BUTTON_PADDING }
        }
        Some(
            WidgetKind::Container { .. }
            | WidgetKind::Slider { .. }
            | WidgetKind::Splitter { .. }
            | WidgetKind::TabBar { .. }
            | WidgetKind::PanelTitleBar { .. }
            | WidgetKind::Viewport { .. },
        )
        | None => Size::ZERO,
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
}
