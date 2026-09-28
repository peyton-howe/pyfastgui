//! Widget chrome: a display list of solid rects, circles, and cached text-run sprites.
//!
//! Two draw paths share that list:
//! - **GPU (default in the app).** `build_quads` packs text-run sprites into a glyph atlas and
//!   emits instanced quads. Metal and Vulkan composite them; scrolling/resizing re-sends a few
//!   hundred KB of vertex data instead of a window-sized pixmap. Text is still shaped and
//!   rasterized on the CPU once per string (the text-run cache).
//! - **CPU (fallback, `FASTGUI_CHROME=cpu`).** `rasterize` paints into a retained window-sized
//!   RGBA8 buffer with dirty-rect diffs and partial upload — also the pixel reference the GPU
//!   path is compared against in tests.
//!
//! Shared pieces that keep both paths cheap:
//! - **Display list + diff.** Each frame builds per-widget ops and compares with the previous
//!   list. The CPU path repaints only changed bounds; the GPU path re-emits quads when ops
//!   change (and skips the frame when nothing did).
//! - **Text run cache.** A string is shaped and rasterized once into a premultiplied RGBA run,
//!   keyed by (text, size, box, color). Unchanged labels are never reshaped.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

mod gpu;
#[cfg(feature = "testing")]
pub mod testing;

use cosmic_text::{Attrs, Buffer, Color as CosmicColor, FontSystem, Metrics, Shaping, SwashCache, Wrap};
use fastgui_core::text_edit::{TextEdit, TextMeasure};
use fastgui_core::widget::{Color, DropZone, WidgetKind, WidgetTree, TEXT_INPUT_PADDING};
use fastgui_core::{ChromeFrame, PixelRect};
use tiny_skia::{Paint, Pixmap, Rect, Transform};

type WidgetRect = fastgui_core::widget::Rect;

/// The window background color chrome fills the whole pixmap with before drawing widgets on
/// top — a `Container` with a transparent background just lets this show through.
const BACKGROUND: Color = Color([0.10, 0.11, 0.13, 1.0]);

/// Drawn as the last step, on top of everything, while a `Panel` title bar is being dragged over
/// a drop-eligible region — see `rasterize`'s `drop_indicator` parameter.
const DROP_INDICATOR_COLOR: Color = Color([0.40, 0.65, 1.0, 0.35]);

/// `Item::key` for the drop indicator (widget keys are taffy node ids, which never reach this).
const DROP_INDICATOR_KEY: u64 = u64::MAX;

/// Flipped into a `ScrollArea`'s id for the key of its scrollbars' item (drawn after its
/// content, so the bars sit on top).
const SCROLLBAR_KEY_BIT: u64 = 1 << 63;

/// Once damage covers more than this share of the window, repaint (and upload) all of it: the
/// partial path's per-rect overhead stops paying for itself.
const FULL_REPAINT_FRACTION: f64 = 0.5;

/// More separate damage rects than this collapse into their bounding rect.
const MAX_DAMAGE_RECTS: usize = 8;

/// Above this many raw damage rects, skip pairwise merging and take their bounding rect.
const MERGE_PAIRWISE_LIMIT: usize = 256;

/// A cached text run survives this many `rasterize` calls without being drawn — enough for tab
/// switches to come back warm, while a label whose text changes every frame (a slider readout)
/// can't grow the cache without bound.
const TEXT_CACHE_KEEP_FRAMES: u64 = 120;

const SLIDER_THUMB_RADIUS: f32 = 8.0;

/// Dims everything behind a modal popup.
const MODAL_SCRIM: Color = Color([0.0, 0.0, 0.0, 0.45]);

/// Keyboard-focus outline, drawn just inside the focused widget's rect (points, before scale).
const FOCUS_RING_COLOR: Color = Color([0.4, 0.7, 1.0, 1.0]);
const FOCUS_RING_WIDTH: f32 = 2.0;

/// `(byte index, x)` caret positions along one shaped line; see `ChromeRenderer::line_stops`.
type CaretStops = Arc<[(usize, f32)]>;

/// Line height as a multiple of font size — cosmic-text `Metrics` everywhere in chrome.
const LINE_HEIGHT_RATIO: f32 = 1.25;

/// Text-input caret width (points, before scale).
const CARET_WIDTH: f32 = 1.5;

/// Past this many measured lines, the caret-stop cache starts over.
const LINE_CACHE_LIMIT: usize = 256;

pub struct ChromeRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_cache: TextCache,
    /// The last frame's pixels, kept so a frame with small damage only repaints that part.
    frame: Option<Pixmap>,
    /// The display list `frame` was painted from, diffed against the next one.
    items: Vec<Item>,
    /// Backing storage for the last returned `ChromeFrame::damage`.
    damage: Vec<PixelRect>,
    /// State of the GPU path (`build_quads`); the CPU path (`rasterize`) doesn't touch it.
    gpu: gpu::GpuState,
    /// Caret stops of recently measured single lines, keyed by (text, font size bits).
    line_cache: HashMap<(String, u32), CaretStops>,
    /// Index of the first popup (overlay) item in the last `build_items` list.
    overlay_items: usize,
}

impl ChromeRenderer {
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new(),
            swash_cache: SwashCache::new(),
            text_cache: TextCache::default(),
            frame: None,
            items: Vec::new(),
            damage: Vec::new(),
            gpu: gpu::GpuState::default(),
            line_cache: HashMap::new(),
            overlay_items: 0,
        }
    }

    /// Forget the retained frame, so the next `rasterize` repaints everything and reports it as
    /// fully damaged (a full upload), and the next `build_quads` re-sends every quad and
    /// re-uploads a fresh atlas. Call when the backend's chrome texture or atlas was lost.
    pub fn invalidate(&mut self) {
        self.frame = None;
        self.items.clear();
        self.gpu.force = true;
        self.gpu.atlas = gpu::Atlas::default();
    }

    /// Rasterize `tree` (whose layout must already be up to date — call
    /// `WidgetTree::compute_layout` first) into a `width`x`height` RGBA8 frame. Layout rects and
    /// font sizes are in the same units as `compute_layout` and are multiplied by `scale` so a
    /// HiDPI window can keep layout in points while the pixmap matches backing pixels.
    /// `drop_indicator` — `(region rect, zone)` — draws a translucent highlight over the sub-area
    /// of that region a dragged `Panel` title bar would land in if released right now; `None`
    /// when no drag is in progress.
    ///
    /// Returns `None` when nothing visible changed since the previous call, so there is nothing
    /// to upload. Otherwise the frame's `damage` lists the rects that changed, or is `None` when
    /// the whole frame was repainted.
    pub fn rasterize(
        &mut self,
        tree: &WidgetTree,
        width: u32,
        height: u32,
        drop_indicator: Option<(WidgetRect, DropZone)>,
        scale: f32,
    ) -> Option<ChromeFrame<'_>> {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let (width, height) = (width.max(1), height.max(1));
        let window = PixelRect { x: 0, y: 0, width, height };

        self.text_cache.generation += 1;
        let items = self.build_items(tree, drop_indicator, scale, window);
        self.text_cache.evict();

        let same_size = self.frame.as_ref().is_some_and(|f| f.width() == width && f.height() == height);
        let damage = if same_size { diff_items(&self.items, &items) } else { None };
        self.items = items;
        let damage = match damage {
            Some(rects) if rects.is_empty() => return None,
            Some(rects) => merge_damage(rects, window),
            None => None,
        };

        let partial = damage.is_some();
        match damage {
            Some(rects) => {
                let frame = self.frame.as_mut().expect("damage is only computed against a retained frame");
                for rect in &rects {
                    let mut patch = Pixmap::new(rect.width, rect.height).expect("damage rects are non-empty");
                    paint(&mut patch, &self.items, *rect);
                    copy_patch(frame, &patch, *rect);
                }
                self.damage = rects;
            }
            None => {
                if !same_size {
                    self.frame = Some(Pixmap::new(width, height).expect("nonzero dimensions"));
                }
                paint(self.frame.as_mut().expect("ensured above"), &self.items, window);
                self.damage.clear();
            }
        }

        let frame = self.frame.as_ref().expect("painted above");
        Some(ChromeFrame {
            width,
            height,
            data: frame.data(),
            damage: partial.then_some(self.damage.as_slice()),
        })
    }

    /// One `Item` per widget, in paint order (parent before children), plus the drop indicator.
    /// Widgets that draw nothing still get an (empty) item so the key sequence only changes
    /// when the tree's structure does.
    fn build_items(
        &mut self,
        tree: &WidgetTree,
        drop_indicator: Option<(WidgetRect, DropZone)>,
        scale: f32,
        window: PixelRect,
    ) -> Vec<Item> {
        let mut items = Vec::new();
        // Popups are the root's last children, so once the walk reaches one, everything after
        // it is overlay content; it's collected separately and appended after the drop
        // indicator (see `overlay_items`).
        let mut overlay = Vec::new();
        let mut in_overlay = false;
        let popups: Vec<_> = tree.walk().filter(|&id| matches!(tree.kind(id), Some(WidgetKind::Popup { .. }))).collect();
        for id in tree.walk() {
            in_overlay |= popups.contains(&id);
            let (Some(logical_rect), Some(kind)) = (tree.absolute_rect(id), tree.kind(id)) else {
                continue;
            };
            // The × geometry comes from `close_button_rect` on the *logical* rect (then scaled),
            // exactly like the hit test in `fastgui-app`, so drawn and clickable areas coincide
            // at any DPI.
            let rect = scale_rect(logical_rect, scale);
            let mut ops = Vec::new();
            match kind {
                WidgetKind::Container { background, .. } => push_fill(&mut ops, rect, *background),
                WidgetKind::Label { text, font_size, color } => {
                    self.push_text(&mut ops, rect, text, *font_size * scale, *color);
                }
                WidgetKind::Button { text, font_size, text_color, background, .. } => {
                    push_fill(&mut ops, rect, *background);
                    self.push_text(&mut ops, rect, text, *font_size * scale, *text_color);
                }
                WidgetKind::Slider { value, min, max, track_color, thumb_color, .. } => {
                    push_slider(&mut ops, rect, *value, *min, *max, *track_color, *thumb_color, scale);
                }
                WidgetKind::Splitter { bar_color, .. } => push_fill(&mut ops, rect, *bar_color),
                WidgetKind::ScrollArea { background, .. } => push_fill(&mut ops, rect, *background),
                WidgetKind::Popup { modal, background, border, .. } => {
                    if *modal {
                        let screen = WidgetRect { x: 0.0, y: 0.0, width: window.width as f32, height: window.height as f32 };
                        push_fill(&mut ops, screen, MODAL_SCRIM);
                    }
                    push_fill(&mut ops, rect, *background);
                    push_outline(&mut ops, rect, scale.max(1.0), *border);
                }
                WidgetKind::TabBar {
                    titles,
                    active,
                    font_size,
                    text_color,
                    active_color,
                    inactive_color,
                    on_close,
                    ..
                } => {
                    self.push_tab_bar(
                        &mut ops,
                        logical_rect,
                        scale,
                        titles,
                        *active,
                        *font_size * scale,
                        *text_color,
                        *active_color,
                        *inactive_color,
                        on_close.as_slice(),
                    );
                }
                WidgetKind::PanelTitleBar { title, font_size, text_color, background, on_close, .. } => {
                    push_fill(&mut ops, rect, *background);
                    let close = scale_rect(fastgui_core::widget::close_button_rect(logical_rect), scale);
                    let text_rect = if on_close.is_some() {
                        WidgetRect { width: (rect.width - close.width).max(0.0), ..rect }
                    } else {
                        rect
                    };
                    self.push_text(&mut ops, text_rect, title, *font_size * scale, *text_color);
                    if on_close.is_some() {
                        self.push_close_glyph(&mut ops, close, *font_size * scale, *text_color);
                    }
                }
                WidgetKind::TextInput {
                    edit,
                    placeholder,
                    font_size,
                    text_color,
                    placeholder_color,
                    background,
                    selection_color,
                    scroll,
                    preedit,
                    ..
                } => {
                    push_fill(&mut ops, rect, *background);
                    let field = TextField {
                        edit,
                        placeholder,
                        preedit: preedit.as_ref(),
                        focused: tree.focused() == Some(id),
                        font_size: *font_size,
                        text_color: *text_color,
                        placeholder_color: *placeholder_color,
                        selection_color: *selection_color,
                        caret_color: FOCUS_RING_COLOR,
                        scroll: *scroll,
                    };
                    self.push_text_field(&mut ops, logical_rect, scale, &field);
                }
                WidgetKind::Viewport { .. } => {
                    // Placeholder only — the GPU draws the real frame in this rect after chrome.
                    push_fill(&mut ops, rect, Color([0.05, 0.06, 0.08, 1.0]));
                }
            }
            if tree.focused() == Some(id) {
                push_outline(&mut ops, rect, FOCUS_RING_WIDTH * scale, FOCUS_RING_COLOR);
            }
            if let Some(clip) = tree.clip_rect(id) {
                ops = clip_ops(ops, Clip::from_rect(scale_rect(clip, scale)));
            }
            let target = if in_overlay { &mut overlay } else { &mut items };
            target.push(Item::new(u64::from(id), ops, window));
        }

        // Overlay scrollbars, after all content so nothing scrolled covers them.
        let mut in_overlay = false;
        for id in tree.walk() {
            in_overlay |= popups.contains(&id);
            let Some(WidgetKind::ScrollArea { bar_color, .. }) = tree.kind(id) else { continue };
            let mut ops = Vec::new();
            let (vertical, horizontal) = tree.scrollbar_thumbs(id);
            for thumb in [vertical, horizontal].into_iter().flatten() {
                push_fill(&mut ops, scale_rect(thumb, scale), *bar_color);
            }
            if let Some(clip) = tree.clip_rect(id) {
                ops = clip_ops(ops, Clip::from_rect(scale_rect(clip, scale)));
            }
            let target = if in_overlay { &mut overlay } else { &mut items };
            target.push(Item::new(u64::from(id) ^ SCROLLBAR_KEY_BIT, ops, window));
        }

        // Same rect the release will commit to (`DropZone::preview_rect`), drawn last so no
        // widget covers it. `fastgui-app` also skips GPU viewport layers under it. Always
        // present (empty without a drag) so showing/hiding it doesn't change the key sequence.
        let mut ops = Vec::new();
        if let Some((region_rect, zone)) = drop_indicator {
            push_fill(&mut ops, scale_rect(zone.preview_rect(region_rect), scale), DROP_INDICATOR_COLOR);
        }
        items.push(Item::new(DROP_INDICATOR_KEY, ops, window));
        self.overlay_items = items.len();
        items.extend(overlay);
        items
    }

    /// A `TextInput`'s contents inside its (already filled) `logical_rect`: selection highlight,
    /// one unwrapped line of text shifted left by `scroll` and clipped to the padded interior,
    /// an underline under any IME composition, and the caret while focused. Measured at the
    /// drawn (scaled) size so the caret sits exactly between the glyphs that are drawn.
    fn push_text_field(&mut self, ops: &mut Vec<Op>, logical_rect: WidgetRect, scale: f32, field: &TextField<'_>) {
        let rect = scale_rect(logical_rect, scale);
        let padding = TEXT_INPUT_PADDING * scale;
        let font_size = field.font_size * scale;
        let line_height = font_size * LINE_HEIGHT_RATIO;
        let inner = WidgetRect {
            x: rect.x + padding,
            y: (rect.y + (rect.height - line_height) / 2.0).round(),
            width: (rect.width - 2.0 * padding).max(0.0),
            height: line_height.min(rect.height),
        };
        if inner.width <= 0.0 || inner.height <= 0.0 {
            return;
        }
        let scroll = (field.scroll * scale).round();

        // While composing, the IME's text shows at the caret (and the selection isn't drawn).
        let (display, caret, composing) = field.edit.composed(field.preedit.filter(|_| field.focused));
        let selection = if composing.is_some() { 0..0 } else { field.edit.selection() };
        let stops = self.line_stops(&display, font_size);
        let x_at = |index: usize| inner.x + stop_x(&stops, index) - scroll;
        let clip = |x0: f32, x1: f32| (x0.max(inner.x), x1.min(inner.x + inner.width));

        if field.focused && !selection.is_empty() {
            let (x0, x1) = clip(x_at(selection.start), x_at(selection.end));
            push_fill(ops, WidgetRect { x: x0, width: x1 - x0, ..inner }, field.selection_color);
        }
        if display.is_empty() {
            if !field.placeholder.is_empty() {
                self.push_line_text(ops, inner, field.placeholder, font_size, 0.0, field.placeholder_color);
            }
        } else {
            self.push_line_text(ops, inner, &display, font_size, scroll, field.text_color);
        }
        if let Some(range) = composing {
            let (x0, x1) = clip(x_at(range.start), x_at(range.end));
            let thickness = scale.max(1.0);
            push_fill(
                ops,
                WidgetRect { x: x0, y: inner.y + inner.height - thickness, width: x1 - x0, height: thickness },
                field.text_color,
            );
        }
        if field.focused {
            let width = (CARET_WIDTH * scale).max(1.0);
            let x = x_at(caret).round().clamp(inner.x, inner.x + inner.width - width);
            push_fill(ops, WidgetRect { x, width, ..inner }, field.caret_color);
        }
    }

    /// One line of `text`, not wrapped, shifted left by `scroll` px and clipped to `rect`.
    fn push_line_text(&mut self, ops: &mut Vec<Op>, rect: WidgetRect, text: &str, font_size: f32, scroll: f32, color: Color) {
        let run = self.text_cache.get(
            &mut self.font_system,
            &mut self.swash_cache,
            text,
            font_size,
            rect,
            color,
            Some(scroll),
        );
        if run.width > 0 {
            ops.push(Op::Text { run, x: rect.x.round() as i32, y: rect.y.round() as i32, clip: None });
        }
    }

    /// `(byte index, x)` caret stops for `text` shaped as one unwrapped line at `font_size`:
    /// one per glyph cluster, in byte order, ending with `(text.len(), line width)`.
    fn line_stops(&mut self, text: &str, font_size: f32) -> CaretStops {
        let key = (text.to_owned(), font_size.to_bits());
        if let Some(stops) = self.line_cache.get(&key) {
            return stops.clone();
        }
        if self.line_cache.len() >= LINE_CACHE_LIMIT {
            self.line_cache.clear();
        }
        let buffer = shape_line(&mut self.font_system, text, font_size, None);
        let mut stops: Vec<(usize, f32)> = Vec::new();
        let mut width: f32 = 0.0;
        for run in buffer.layout_runs() {
            width = width.max(run.line_w);
            stops.extend(run.glyphs.iter().map(|glyph| (glyph.start, glyph.x)));
        }
        stops.sort_by_key(|&(index, _)| index);
        stops.dedup_by_key(|&mut (index, _)| index);
        stops.retain(|&(index, _)| index < text.len());
        stops.push((text.len(), width));
        let stops: CaretStops = stops.into();
        self.line_cache.insert(key, stops.clone());
        stops
    }

    fn push_text(&mut self, ops: &mut Vec<Op>, rect: WidgetRect, text: &str, font_size: f32, color: Color) {
        if text.is_empty() || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        let run = self.text_cache.get(&mut self.font_system, &mut self.swash_cache, text, font_size, rect, color, None);
        if run.width == 0 {
            return;
        }
        // Snapped to whole pixels: runs are blitted straight into the buffer, and a fractional
        // origin would only smear each glyph pixel across its neighbours.
        ops.push(Op::Text { run, x: rect.x.round() as i32, y: rect.y.round() as i32, clip: None });
    }

    /// "×" centred in `rect` (physical px). Text is laid out from the top-left, which would put
    /// the glyph in the top-left corner of its square, off-centre from the area that actually
    /// closes. Uses the same rough advance/line-height ratios as layout's `measure_text`, which
    /// is plenty for a single glyph.
    fn push_close_glyph(&mut self, ops: &mut Vec<Op>, rect: WidgetRect, font_size: f32, color: Color) {
        let glyph_width = font_size * 0.55;
        let line_height = font_size * 1.25;
        let inset = WidgetRect {
            x: rect.x + ((rect.width - glyph_width) * 0.5).max(0.0),
            y: rect.y + ((rect.height - line_height) * 0.5).max(0.0),
            width: rect.width.min(glyph_width * 2.0),
            height: rect.height.max(line_height),
        };
        self.push_text(ops, inset, "×", font_size, color);
    }

    /// Bakes its own header segments (equal-width, one per title) directly rather than composing
    /// child `Label` nodes — see `WidgetKind::TabBar`'s doc comment for why. Passes each segment's
    /// *full* rect as the text box (not a tightly text-measured sub-box), avoiding the clipping
    /// `Panel`'s title bar hit when `measure_text`'s heuristic underestimated short strings.
    #[allow(clippy::too_many_arguments)]
    fn push_tab_bar(
        &mut self,
        ops: &mut Vec<Op>,
        logical_rect: WidgetRect,
        scale: f32,
        titles: &[String],
        active: usize,
        font_size: f32,
        text_color: Color,
        active_color: Color,
        inactive_color: Color,
        on_close: &[Option<fastgui_core::widget::PanelCloseCallback>],
    ) {
        if titles.is_empty() || logical_rect.width <= 0.0 {
            return;
        }
        // Segments are computed in logical units, like `fastgui-app`'s tab hit testing, then
        // scaled for drawing.
        let segment_width = logical_rect.width / titles.len() as f32;
        for (index, title) in titles.iter().enumerate() {
            let logical_segment = WidgetRect {
                x: logical_rect.x + index as f32 * segment_width,
                y: logical_rect.y,
                width: segment_width,
                height: logical_rect.height,
            };
            let segment_rect = scale_rect(logical_segment, scale);
            let close = scale_rect(fastgui_core::widget::close_button_rect(logical_segment), scale);
            push_fill(ops, segment_rect, if index == active { active_color } else { inactive_color });
            let closable = on_close.get(index).is_some_and(|c| c.is_some());
            let text_rect = if closable {
                WidgetRect { width: (segment_rect.width - close.width).max(0.0), ..segment_rect }
            } else {
                segment_rect
            };
            self.push_text(ops, text_rect, title, font_size, text_color);
            if closable {
                self.push_close_glyph(ops, close, font_size, text_color);
            }
        }
    }
}

impl TextMeasure for ChromeRenderer {
    fn caret_x(&mut self, text: &str, font_size: f32, index: usize) -> f32 {
        stop_x(&self.line_stops(text, font_size), index)
    }

    fn index_at(&mut self, text: &str, font_size: f32, x: f32) -> usize {
        let stops = self.line_stops(text, font_size);
        stops
            .iter()
            .min_by(|a, b| (a.1 - x).abs().total_cmp(&(b.1 - x).abs()))
            .map_or(0, |&(index, _)| index)
    }
}

/// What `push_text_field` needs from a `WidgetKind::TextInput`.
struct TextField<'a> {
    edit: &'a TextEdit,
    placeholder: &'a str,
    preedit: Option<&'a (String, Option<(usize, usize)>)>,
    focused: bool,
    font_size: f32,
    text_color: Color,
    placeholder_color: Color,
    selection_color: Color,
    caret_color: Color,
    scroll: f32,
}

/// X of the caret before byte `index`: the last stop at or before it (inside a multi-glyph
/// cluster that's the cluster's start).
fn stop_x(stops: &[(usize, f32)], index: usize) -> f32 {
    let after = stops.partition_point(|&(stop, _)| stop <= index);
    stops[..after].last().map_or(0.0, |&(_, x)| x)
}

/// A cosmic-text buffer holding `text` as lines laid out in a `width`-wide box (`None`: one
/// unwrapped line).
fn shape_line(font_system: &mut FontSystem, text: &str, font_size: f32, height: Option<f32>) -> Buffer {
    let mut buffer = Buffer::new(font_system, Metrics::new(font_size, font_size * LINE_HEIGHT_RATIO));
    buffer.set_wrap(font_system, Wrap::None);
    buffer.set_size(font_system, None, height);
    buffer.set_text(font_system, text, &Attrs::new(), Shaping::Advanced);
    buffer
}

impl Default for ChromeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// One primitive draw, in physical pixels of the whole window.
///
/// Every op must paint the same pixels whether it lands in the full frame or in a damage patch
/// (shifted by the patch's whole-pixel origin). Rects are kept as edges, so shifting them is
/// exact f32 subtraction. Anything built from a path is not, because curve points round
/// differently at different magnitudes, so it's rasterized once into a `Sprite` and blitted.
enum Op {
    Fill { left: f32, top: f32, right: f32, bottom: f32, color: Color },
    /// `sprite` is the circle rasterized at its whole-pixel position, made on first CPU paint
    /// (the GPU path draws the circle analytically instead); only the parameters are compared.
    /// `clip` (set only when the circle is partly clipped): draw just the part inside it.
    Circle { cx: f32, cy: f32, radius: f32, color: Color, sprite: OnceLock<Sprite>, clip: Option<Clip> },
    /// `(x, y)` is the text box's snapped origin; the run's own offset is relative to it.
    /// `clip` as for `Circle`.
    Text { run: Arc<Sprite>, x: i32, y: i32, clip: Option<Clip> },
}

/// A whole-pixel rect (physical px, window space) that ops inside a `ScrollArea` are clipped
/// to. Whole pixels keep sprite clipping exact: a clipped run just copies fewer pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Clip {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl Clip {
    fn from_rect(rect: WidgetRect) -> Self {
        Self {
            left: rect.x.round() as i32,
            top: rect.y.round() as i32,
            right: (rect.x + rect.width).round() as i32,
            bottom: (rect.y + rect.height).round() as i32,
        }
    }

    /// Where `[left, top, right, bottom]` stands relative to this clip: `None` if they don't
    /// overlap, `Some(true)` if it's entirely inside, `Some(false)` if partly.
    fn classify(&self, left: f32, top: f32, right: f32, bottom: f32) -> Option<bool> {
        let (cl, ct, cr, cb) = (self.left as f32, self.top as f32, self.right as f32, self.bottom as f32);
        if right <= cl || bottom <= ct || left >= cr || top >= cb {
            return None;
        }
        Some(left >= cl && top >= ct && right <= cr && bottom <= cb)
    }

    fn shifted(self, dx: i32, dy: i32) -> Self {
        Self { left: self.left + dx, top: self.top + dy, right: self.right + dx, bottom: self.bottom + dy }
    }
}

/// Restrict `ops` to `clip`: fills are cut to it exactly, ops entirely outside are dropped,
/// and circles/text that straddle it carry it along (see `Op::Circle` / `Op::Text`).
fn clip_ops(ops: Vec<Op>, clip: Clip) -> Vec<Op> {
    let (cl, ct, cr, cb) = (clip.left as f32, clip.top as f32, clip.right as f32, clip.bottom as f32);
    ops.into_iter()
        .filter_map(|op| match op {
            Op::Fill { left, top, right, bottom, color } => {
                let (left, top, right, bottom) = (left.max(cl), top.max(ct), right.min(cr), bottom.min(cb));
                (right > left && bottom > top).then_some(Op::Fill { left, top, right, bottom, color })
            }
            // One pixel of anti-aliasing past the radius, as drawn.
            Op::Circle { cx, cy, radius, color, sprite, .. } => {
                let inside = clip.classify(cx - radius - 1.0, cy - radius - 1.0, cx + radius + 1.0, cy + radius + 1.0)?;
                Some(Op::Circle { cx, cy, radius, color, sprite, clip: (!inside).then_some(clip) })
            }
            Op::Text { run, x, y, .. } => {
                let (left, top) = ((x + run.x) as f32, (y + run.y) as f32);
                let inside = clip.classify(left, top, left + run.width as f32, top + run.height as f32)?;
                Some(Op::Text { run, x, y, clip: (!inside).then_some(clip) })
            }
        })
        .collect()
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        match (self, other) {
            (
                Op::Fill { left: l1, top: t1, right: r1, bottom: b1, color: c1 },
                Op::Fill { left: l2, top: t2, right: r2, bottom: b2, color: c2 },
            ) => bits(&[*l1, *t1, *r1, *b1]) == bits(&[*l2, *t2, *r2, *b2]) && bits(&c1.0) == bits(&c2.0),
            (
                Op::Circle { cx: x1, cy: y1, radius: r1, color: c1, clip: k1, .. },
                Op::Circle { cx: x2, cy: y2, radius: r2, color: c2, clip: k2, .. },
            ) => bits(&[*x1, *y1, *r1]) == bits(&[*x2, *y2, *r2]) && bits(&c1.0) == bits(&c2.0) && k1 == k2,
            // Pointer identity is exact here: both display lists being compared hold their
            // runs alive, so a run's allocation can't be reused for different text meanwhile.
            // The same text re-cached after eviction compares unequal, which only over-damages.
            (Op::Text { run: r1, x: x1, y: y1, clip: k1 }, Op::Text { run: r2, x: x2, y: y2, clip: k2 }) => {
                Arc::ptr_eq(r1, r2) && x1 == x2 && y1 == y2 && k1 == k2
            }
            _ => false,
        }
    }
}

impl Op {
    /// Pixels this op can touch, rounded out (plus 1px for anti-aliasing) and clipped to
    /// `window`.
    fn bounds(&self, window: PixelRect) -> PixelRect {
        let sprite_bounds = |sprite: &Sprite, x: i32, y: i32| {
            let (x0, y0) = ((x + sprite.x) as f32, (y + sprite.y) as f32);
            pixel_bounds(x0, y0, x0 + sprite.width as f32, y0 + sprite.height as f32, window)
        };
        let clipped = |bounds: PixelRect, clip: &Option<Clip>| match clip {
            Some(c) => bounds.intersect(&pixel_bounds(c.left as f32, c.top as f32, c.right as f32, c.bottom as f32, window)),
            None => bounds,
        };
        match self {
            Op::Fill { left, top, right, bottom, .. } => pixel_bounds(*left, *top, *right, *bottom, window),
            // Same extent `Sprite::circle` covers.
            Op::Circle { cx, cy, radius, clip, .. } => {
                clipped(pixel_bounds(cx - radius, cy - radius, cx + radius, cy + radius, window), clip)
            }
            Op::Text { run, x, y, clip } => clipped(sprite_bounds(run, *x, *y), clip),
        }
    }
}

/// Everything one widget draws, plus where.
struct Item {
    /// The widget's id (or `DROP_INDICATOR_KEY`): two display lists are diffed item by item
    /// only when their key sequences match.
    key: u64,
    bounds: PixelRect,
    ops: Vec<Op>,
}

impl Item {
    fn new(key: u64, ops: Vec<Op>, window: PixelRect) -> Self {
        let bounds = ops.iter().fold(PixelRect::default(), |acc, op| acc.union(&op.bounds(window)));
        Self { key, bounds, ops }
    }
}

/// Rects that differ between two display lists: each changed item's old and new bounds. `None`
/// when the lists aren't the same widgets in the same order (a tree rebuild, a rearrange), where
/// matching items up isn't worth it — those are rare and repaint everything, as before.
fn diff_items(old: &[Item], new: &[Item]) -> Option<Vec<PixelRect>> {
    if old.len() != new.len() || old.iter().zip(new).any(|(a, b)| a.key != b.key) {
        return None;
    }
    let mut damage = Vec::new();
    for (a, b) in old.iter().zip(new) {
        if a.ops != b.ops {
            damage.extend([a.bounds, b.bounds].into_iter().filter(|r| !r.is_empty()));
        }
    }
    Some(damage)
}

/// Merge overlapping or nearly-adjacent damage rects (so an item's old and new position — e.g.
/// a slider thumb — become one patch), capping the count. `None` means "repaint everything":
/// the damage covers enough of the window that one full pass is cheaper.
fn merge_damage(mut rects: Vec<PixelRect>, window: PixelRect) -> Option<Vec<PixelRect>> {
    // Pairwise merging is quadratic per pass; thousands of rects (every cell of a scrolled
    // table moved) would cost more than the repaint. Their bounding rect is the answer anyway.
    if rects.len() > MERGE_PAIRWISE_LIMIT {
        rects = vec![rects.iter().fold(PixelRect::default(), |acc, r| acc.union(r))];
    }
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let union = rects[i].union(&rects[j]);
                // Merge when the union wastes less than a quarter of its area over the two parts.
                if union.area() * 3 <= (rects[i].area() + rects[j].area()) * 4 {
                    rects[i] = union;
                    rects.swap_remove(j);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    if rects.len() > MAX_DAMAGE_RECTS {
        let union = rects.iter().fold(PixelRect::default(), |acc, r| acc.union(r));
        rects = vec![union];
    }
    let covered: u64 = rects.iter().map(PixelRect::area).sum();
    if covered as f64 > window.area() as f64 * FULL_REPAINT_FRACTION {
        return None;
    }
    Some(rects)
}

/// Paint every item overlapping `clip` into `target`, which covers exactly `clip` (a patch the
/// size of the damage rect, or the whole frame). Ops are translated by whole pixels, so a patch
/// gets exactly the pixels a full repaint would.
fn paint(target: &mut Pixmap, items: &[Item], clip: PixelRect) {
    target.fill(to_tiny_skia_color(BACKGROUND));
    let (dx, dy) = (clip.x as f32, clip.y as f32);
    let (ox, oy) = (clip.x as i32, clip.y as i32);
    for item in items.iter().filter(|item| item.bounds.intersects(&clip)) {
        for op in &item.ops {
            match op {
                Op::Fill { left, top, right, bottom, color } => {
                    // `right - dx` is exact whenever the rect reaches into the patch (dx is a
                    // whole number ≤ right). A left edge before the patch origin may round, so
                    // pin it just outside the patch, where it can't affect any pixel.
                    if *right <= dx || *bottom <= dy {
                        continue;
                    }
                    let (l, t) = ((left - dx).max(-1.0), (top - dy).max(-1.0));
                    fill_rect(target, l, t, right - dx, bottom - dy, *color);
                }
                Op::Circle { cx, cy, radius, color, sprite, clip } => {
                    let sprite = sprite.get_or_init(|| Sprite::circle(*cx, *cy, *radius, *color));
                    let clip = clip.map(|c| c.shifted(-ox, -oy));
                    blit_sprite(target, sprite, sprite.x - ox, sprite.y - oy, clip);
                }
                Op::Text { run, x, y, clip } => {
                    let clip = clip.map(|c| c.shifted(-ox, -oy));
                    blit_sprite(target, run, x + run.x - ox, y + run.y - oy, clip);
                }
            }
        }
    }
}

fn copy_patch(frame: &mut Pixmap, patch: &Pixmap, rect: PixelRect) {
    let frame_stride = frame.width() as usize * 4;
    let row_bytes = rect.width as usize * 4;
    let dst = frame.data_mut();
    for (row, src) in patch.data().chunks_exact(row_bytes).enumerate() {
        let start = (rect.y as usize + row) * frame_stride + rect.x as usize * 4;
        dst[start..start + row_bytes].copy_from_slice(src);
    }
}

/// Pre-rasterized premultiplied RGBA pixels, drawn by `blit_sprite` at whole-pixel positions.
/// `(x, y)` offsets the pixels from the owning op's origin: for text, the snapped text box
/// origin (the run covers exactly the glyphs' ink); for a circle, the window origin.
#[derive(Default)]
struct Sprite {
    /// Unique per sprite (0 only for empty ones), so the GPU atlas can track which sprites it
    /// already holds without keeping them alive.
    id: u64,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Sprite {
    /// A filled anti-aliased circle, rasterized at its position in the window (only the
    /// fractional part of the centre matters) and placed at whole pixels.
    fn circle(cx: f32, cy: f32, radius: f32, color: Color) -> Self {
        let (x0, y0) = ((cx - radius).floor() as i32 - 1, (cy - radius).floor() as i32 - 1);
        let (x1, y1) = ((cx + radius).ceil() as i32 + 1, (cy + radius).ceil() as i32 + 1);
        let (width, height) = ((x1 - x0).max(1) as u32, (y1 - y0).max(1) as u32);
        let Some(mut pixmap) = Pixmap::new(width, height) else { return Self::default() };
        fill_circle(&mut pixmap, cx - x0 as f32, cy - y0 as f32, radius, color);
        Self { id: next_sprite_id(), x: x0, y: y0, width, height, pixels: pixmap.take() }
    }

    /// `line_scroll: Some(px)` lays `text` out as one unwrapped line shifted left by `px` and
    /// clips it to `rect` (a text field); `None` wraps it in `rect` (a label).
    fn shape(
        font_system: &mut FontSystem,
        swash_cache: &mut SwashCache,
        text: &str,
        font_size: f32,
        rect: WidgetRect,
        color: CosmicColor,
        line_scroll: Option<f32>,
    ) -> Self {
        let buffer = match line_scroll {
            Some(_) => shape_line(font_system, text, font_size, Some(rect.height)),
            None => {
                let metrics = Metrics::new(font_size, font_size * LINE_HEIGHT_RATIO);
                let mut buffer = Buffer::new(font_system, metrics);
                buffer.set_size(font_system, Some(rect.width), Some(rect.height));
                buffer.set_text(font_system, text, &Attrs::new(), Shaping::Advanced);
                buffer
            }
        };

        // cosmic-text reports coverage one pixel span at a time; collect first to size the run.
        let mut spans = Vec::new();
        let (shift, clip) = match line_scroll {
            Some(scroll) => (scroll.round() as i32, Some((rect.width as i32, rect.height.ceil() as i32))),
            None => (0, None),
        };
        buffer.draw(font_system, swash_cache, color, |x, y, w, h, color| {
            if color.a() == 0 || w == 0 || h == 0 {
                return;
            }
            let (mut x0, mut x1, mut y0, mut y1) = (x - shift, x - shift + w as i32, y, y + h as i32);
            if let Some((width, height)) = clip {
                (x0, x1, y0, y1) = (x0.max(0), x1.min(width), y0.max(0), y1.min(height));
            }
            if x1 > x0 && y1 > y0 {
                spans.push((x0, y0, x1 - x0, y1 - y0, color));
            }
        });
        let Some(x0) = spans.iter().map(|s| s.0).min() else { return Self::default() };
        let y0 = spans.iter().map(|s| s.1).min().expect("non-empty");
        let x1 = spans.iter().map(|s| s.0 + s.2).max().expect("non-empty");
        let y1 = spans.iter().map(|s| s.1 + s.3).max().expect("non-empty");
        let (width, height) = ((x1 - x0) as u32, (y1 - y0) as u32);

        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        for (x, y, w, h, color) in spans {
            for py in y - y0..y - y0 + h {
                let row = py as usize * width as usize * 4;
                for px in x - x0..x - x0 + w {
                    let i = row + px as usize * 4;
                    blend_premultiplied(&mut pixels[i..i + 4], color.r(), color.g(), color.b(), color.a());
                }
            }
        }
        Self { id: next_sprite_id(), x: x0, y: y0, width, height, pixels }
    }
}

/// Everything but the string itself, so lookups can borrow the `&str` without allocating.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TextParams {
    font_size: u32,
    width: u32,
    height: u32,
    color: [u8; 4],
    /// `Some(scroll bits)` for a clipped single line (see `Sprite::shape`).
    line_scroll: Option<u32>,
}

struct CachedRun {
    run: Arc<Sprite>,
    last_used: u64,
}

#[derive(Default)]
struct TextCache {
    runs: HashMap<TextParams, HashMap<String, CachedRun>>,
    /// Bumped once per `rasterize`; entries remember the generation they were last drawn in.
    generation: u64,
}

impl TextCache {
    #[allow(clippy::too_many_arguments)]
    fn get(
        &mut self,
        font_system: &mut FontSystem,
        swash_cache: &mut SwashCache,
        text: &str,
        font_size: f32,
        rect: WidgetRect,
        color: Color,
        line_scroll: Option<f32>,
    ) -> Arc<Sprite> {
        let color = to_cosmic_color(color);
        let params = TextParams {
            font_size: font_size.to_bits(),
            width: rect.width.to_bits(),
            height: rect.height.to_bits(),
            color: [color.r(), color.g(), color.b(), color.a()],
            line_scroll: line_scroll.map(f32::to_bits),
        };
        let generation = self.generation;
        let by_text = self.runs.entry(params).or_default();
        if let Some(cached) = by_text.get_mut(text) {
            cached.last_used = generation;
            return cached.run.clone();
        }
        let run = Arc::new(Sprite::shape(font_system, swash_cache, text, font_size, rect, color, line_scroll));
        by_text.insert(text.to_owned(), CachedRun { run: run.clone(), last_used: generation });
        run
    }

    fn evict(&mut self) {
        let generation = self.generation;
        self.runs.retain(|_, by_text| {
            by_text.retain(|_, cached| generation - cached.last_used <= TEXT_CACHE_KEEP_FRAMES);
            !by_text.is_empty()
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn push_slider(
    ops: &mut Vec<Op>,
    rect: WidgetRect,
    value: f32,
    min: f32,
    max: f32,
    track_color: Color,
    thumb_color: Color,
    scale: f32,
) {
    // `Display::None` tabs leave a 0×0 rect at (0,0); the track fill already no-ops on that, but
    // the thumb is a circle centred on the rect and would otherwise paint a quarter-disk in the
    // window's top-left corner (tabs_demo's hidden Settings slider).
    if rect.width <= 0.0 || rect.height <= 0.0 {
        return;
    }
    let track_height = (rect.height * 0.3).max(2.0 * scale);
    let track_y = rect.y + (rect.height - track_height) / 2.0;
    push_fill(ops, WidgetRect { x: rect.x, y: track_y, width: rect.width, height: track_height }, track_color);

    let fraction = if max > min { ((value - min) / (max - min)).clamp(0.0, 1.0) } else { 0.0 };
    let (cx, cy, radius) = (rect.x + fraction * rect.width, rect.y + rect.height / 2.0, SLIDER_THUMB_RADIUS * scale);
    ops.push(Op::Circle { cx, cy, radius, color: thumb_color, sprite: OnceLock::new(), clip: None });
}

/// A `width`-thick border along the inside of `rect`, as four fills.
fn push_outline(ops: &mut Vec<Op>, rect: WidgetRect, width: f32, color: Color) {
    let width = width.min(rect.width / 2.0).min(rect.height / 2.0);
    if width <= 0.0 {
        return;
    }
    let WidgetRect { x, y, width: w, height: h } = rect;
    push_fill(ops, WidgetRect { x, y, width: w, height: width }, color);
    push_fill(ops, WidgetRect { x, y: y + h - width, width: w, height: width }, color);
    push_fill(ops, WidgetRect { x, y: y + width, width, height: h - 2.0 * width }, color);
    push_fill(ops, WidgetRect { x: x + w - width, y: y + width, width, height: h - 2.0 * width }, color);
}

fn push_fill(ops: &mut Vec<Op>, rect: WidgetRect, color: Color) {
    if color.0[3] > 0.0 && rect.width > 0.0 && rect.height > 0.0 {
        let (left, top) = (rect.x, rect.y);
        ops.push(Op::Fill { left, top, right: left + rect.width, bottom: top + rect.height, color });
    }
}

fn pixel_bounds(x0: f32, y0: f32, x1: f32, y1: f32, window: PixelRect) -> PixelRect {
    let clamp_x = |v: f32| v.clamp(0.0, window.width as f32) as u32;
    let clamp_y = |v: f32| v.clamp(0.0, window.height as f32) as u32;
    let (left, top) = (clamp_x(x0.floor() - 1.0), clamp_y(y0.floor() - 1.0));
    let (right, bottom) = (clamp_x(x1.ceil() + 1.0), clamp_y(y1.ceil() + 1.0));
    if right <= left || bottom <= top {
        return PixelRect::default();
    }
    PixelRect { x: left, y: top, width: right - left, height: bottom - top }
}

fn scale_rect(rect: WidgetRect, scale: f32) -> WidgetRect {
    WidgetRect { x: rect.x * scale, y: rect.y * scale, width: rect.width * scale, height: rect.height * scale }
}

fn fill_rect(pixmap: &mut Pixmap, left: f32, top: f32, right: f32, bottom: f32, color: Color) {
    let Some(sk_rect) = Rect::from_ltrb(left, top, right, bottom) else { return };
    let mut paint = Paint::default();
    paint.set_color(to_tiny_skia_color(color));
    pixmap.fill_rect(sk_rect, &paint, Transform::identity(), None);
}

fn fill_circle(pixmap: &mut Pixmap, cx: f32, cy: f32, radius: f32, color: Color) {
    let mut path_builder = tiny_skia::PathBuilder::new();
    path_builder.push_circle(cx, cy, radius);
    if let Some(path) = path_builder.finish() {
        let mut paint = Paint::default();
        let [r, g, b, a] = color.0;
        paint.set_color_rgba8((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8);
        paint.anti_alias = true;
        pixmap.fill_path(&path, &paint, tiny_skia::FillRule::Winding, Transform::identity(), None);
    }
}

/// Source-over composite of a premultiplied run onto the premultiplied pixmap, with the run's
/// top-left at `(x, y)`, clipped to the pixmap. For a pixel only one glyph touched this is
/// exactly what blending that glyph pixel straight into the pixmap gave.
/// Composite `run` with its top-left at `(x, y)`, only inside `clip` (pixmap coordinates) if given.
fn blit_sprite(pixmap: &mut Pixmap, run: &Sprite, x: i32, y: i32, clip: Option<Clip>) {
    let (pixmap_w, pixmap_h) = (pixmap.width() as i32, pixmap.height() as i32);
    let clip = clip.unwrap_or(Clip { left: 0, top: 0, right: pixmap_w, bottom: pixmap_h });
    let (x0, y0) = (x.max(0).max(clip.left), y.max(0).max(clip.top));
    let (x1, y1) = (
        (x + run.width as i32).min(pixmap_w).min(clip.right),
        (y + run.height as i32).min(pixmap_h).min(clip.bottom),
    );
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let data = pixmap.data_mut();
    for py in y0..y1 {
        let src_row = (py - y) as usize * run.width as usize * 4;
        let dst_row = py as usize * pixmap_w as usize * 4;
        for px in x0..x1 {
            let s = src_row + (px - x) as usize * 4;
            let src = &run.pixels[s..s + 4];
            let a = src[3];
            if a == 0 {
                continue;
            }
            let d = dst_row + px as usize * 4;
            let dst = &mut data[d..d + 4];
            if a == 255 {
                dst.copy_from_slice(src);
                continue;
            }
            let inv = 255 - a;
            for c in 0..4 {
                dst[c] = src[c].saturating_add(mul_u8(dst[c], inv));
            }
        }
    }
}

fn next_sprite_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn mul_u8(x: u8, y: u8) -> u8 {
    ((u16::from(x) * u16::from(y) + 127) / 255) as u8
}

/// Source-over blend of one straight-alpha RGBA8 color onto one premultiplied RGBA8 pixel
/// (tiny-skia's pixel format): `dst = src·a + dst·(1 − a)`.
fn blend_premultiplied(dst: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    if a == 255 {
        dst.copy_from_slice(&[r, g, b, 255]);
        return;
    }
    let inv = 255 - a;
    dst[0] = mul_u8(r, a) + mul_u8(dst[0], inv);
    dst[1] = mul_u8(g, a) + mul_u8(dst[1], inv);
    dst[2] = mul_u8(b, a) + mul_u8(dst[2], inv);
    dst[3] = a + mul_u8(dst[3], inv);
}

fn to_tiny_skia_color(color: Color) -> tiny_skia::Color {
    let [r, g, b, a] = color.0;
    tiny_skia::Color::from_rgba(r, g, b, a).unwrap_or(tiny_skia::Color::BLACK)
}

fn to_cosmic_color(color: Color) -> CosmicColor {
    let [r, g, b, a] = color.0;
    CosmicColor::rgba((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8)
}

#[cfg(test)]
mod tests;
