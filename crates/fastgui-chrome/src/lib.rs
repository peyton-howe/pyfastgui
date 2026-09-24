//! Widget chrome rasterizer: `cosmic-text` for shaping/layout, `tiny-skia` for CPU
//! rasterization into a single window-sized RGBA8 buffer. Deliberately not a dynamic-atlas /
//! partial-update scheme — the whole tree is small enough (a UI tree, not a scene graph) that
//! re-rasterizing it fully whenever it's dirty is simple and fast enough, and the result reuses
//! `fastgui-render-vk`'s existing CPU-texture-upload path (the same one `Viewport.submit_frame`
//! uses) rather than needing new Vulkan machinery.
//!
//! Everything drawn for M4 is fully opaque, so tiny-skia's premultiplied-alpha pixel format is
//! indistinguishable from straight alpha here — this will need real handling once translucent
//! widgets exist.

use cosmic_text::{Attrs, Buffer, Color as CosmicColor, FontSystem, Metrics, Shaping, SwashCache};
use fastgui_core::widget::{Color, DropZone, WidgetKind, WidgetTree};
use fastgui_core::{CpuFrame, PixelFormat};
use tiny_skia::{Paint, Pixmap, Rect, Transform};

/// The window background color chrome fills the whole pixmap with before drawing widgets on
/// top — since nothing here does alpha blending yet (see module docs), a `Container` with a
/// transparent background just lets this show through.
const BACKGROUND: Color = Color([0.10, 0.11, 0.13, 1.0]);

/// Drawn as the last step, on top of everything, while a `Panel` title bar is being dragged over
/// a drop-eligible region — see `rasterize`'s `drop_indicator` parameter.
const DROP_INDICATOR_COLOR: Color = Color([0.40, 0.65, 1.0, 0.35]);

pub struct ChromeRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
}

impl ChromeRenderer {
    pub fn new() -> Self {
        Self { font_system: FontSystem::new(), swash_cache: SwashCache::new() }
    }

    /// Rasterize `tree` (whose layout must already be up to date — call
    /// `WidgetTree::compute_layout` first) into a `width`x`height` RGBA8 frame. Layout rects and
    /// font sizes are in the same units as `compute_layout` and are multiplied by `scale` so a
    /// HiDPI window can keep layout in points while the pixmap matches backing pixels.
    /// `drop_indicator` — `(region rect, zone)` — draws a translucent highlight over the sub-area
    /// of that region a dragged `Panel` title bar would land in if released right now; `None`
    /// when no drag is in progress.
    pub fn rasterize(
        &mut self,
        tree: &WidgetTree,
        width: u32,
        height: u32,
        drop_indicator: Option<(fastgui_core::widget::Rect, DropZone)>,
        scale: f32,
    ) -> CpuFrame {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let mut pixmap = Pixmap::new(width.max(1), height.max(1)).expect("nonzero dimensions");
        pixmap.fill(to_tiny_skia_color(BACKGROUND));

        for id in tree.walk() {
            let (Some(rect), Some(kind)) = (tree.absolute_rect(id), tree.kind(id)) else {
                continue;
            };
            let rect = scale_rect(rect, scale);
            match kind {
                WidgetKind::Container { background, .. } => {
                    fill_rect(&mut pixmap, rect, *background);
                }
                WidgetKind::Label { text, font_size, color } => {
                    self.draw_text(&mut pixmap, rect, text, *font_size * scale, *color);
                }
                WidgetKind::Button { text, font_size, text_color, background, .. } => {
                    fill_rect(&mut pixmap, rect, *background);
                    self.draw_text(&mut pixmap, rect, text, *font_size * scale, *text_color);
                }
                WidgetKind::Slider { value, min, max, track_color, thumb_color, .. } => {
                    draw_slider(&mut pixmap, rect, *value, *min, *max, *track_color, *thumb_color, scale);
                }
                WidgetKind::Splitter { bar_color, .. } => {
                    fill_rect(&mut pixmap, rect, *bar_color);
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
                    self.draw_tab_bar(
                        &mut pixmap,
                        rect,
                        titles,
                        *active,
                        *font_size * scale,
                        *text_color,
                        *active_color,
                        *inactive_color,
                        on_close.as_slice(),
                    );
                }
                WidgetKind::PanelTitleBar {
                    title,
                    font_size,
                    text_color,
                    background,
                    on_close,
                    ..
                } => {
                    fill_rect(&mut pixmap, rect, *background);
                    let close = fastgui_core::widget::close_button_rect(rect);
                    let text_rect = if on_close.is_some() {
                        fastgui_core::widget::Rect {
                            width: (rect.width - close.width).max(0.0),
                            ..rect
                        }
                    } else {
                        rect
                    };
                    self.draw_text(&mut pixmap, text_rect, title, *font_size * scale, *text_color);
                    if on_close.is_some() {
                        self.draw_text(&mut pixmap, close, "×", *font_size * scale, *text_color);
                    }
                }
                WidgetKind::Viewport { .. } => {
                    // Placeholder only — the GPU draws the real frame in this rect after chrome.
                    fill_rect(&mut pixmap, rect, Color([0.05, 0.06, 0.08, 1.0]));
                }
            }
        }

        if let Some((region_rect, zone)) = drop_indicator {
            fill_rect(&mut pixmap, scale_rect(drop_zone_rect(region_rect, zone), scale), DROP_INDICATOR_COLOR);
        }

        CpuFrame { width, height, format: PixelFormat::Rgba8, data: pixmap.data().to_vec() }
    }

    fn draw_text(
        &mut self,
        pixmap: &mut Pixmap,
        rect: fastgui_core::widget::Rect,
        text: &str,
        font_size: f32,
        color: Color,
    ) {
        if text.is_empty() || rect.width <= 0.0 || rect.height <= 0.0 {
            return;
        }
        let metrics = Metrics::new(font_size, font_size * 1.25);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_size(&mut self.font_system, Some(rect.width), Some(rect.height));
        buffer.set_text(&mut self.font_system, text, &Attrs::new(), Shaping::Advanced);

        let text_color = to_cosmic_color(color);
        let (origin_x, origin_y) = (rect.x, rect.y);
        buffer.draw(&mut self.font_system, &mut self.swash_cache, text_color, |x, y, w, h, color| {
            if color.a() == 0 {
                return;
            }
            let mut paint = Paint::default();
            paint.set_color_rgba8(color.r(), color.g(), color.b(), color.a());
            paint.anti_alias = true;
            if let Some(px_rect) =
                Rect::from_xywh(origin_x + x as f32, origin_y + y as f32, w as f32, h as f32)
            {
                pixmap.fill_rect(px_rect, &paint, Transform::identity(), None);
            }
        });
    }

    /// Bakes its own header segments (equal-width, one per title) directly rather than composing
    /// child `Label` nodes — see `WidgetKind::TabBar`'s doc comment for why. Passes each segment's
    /// *full* rect to `draw_text` (not a tightly text-measured sub-box), avoiding the clipping
    /// `Panel`'s title bar hit when `measure_text`'s heuristic underestimated short strings.
    #[allow(clippy::too_many_arguments)]
    fn draw_tab_bar(
        &mut self,
        pixmap: &mut Pixmap,
        rect: fastgui_core::widget::Rect,
        titles: &[String],
        active: usize,
        font_size: f32,
        text_color: Color,
        active_color: Color,
        inactive_color: Color,
        on_close: &[Option<fastgui_core::widget::PanelCloseCallback>],
    ) {
        if titles.is_empty() || rect.width <= 0.0 {
            return;
        }
        let segment_width = rect.width / titles.len() as f32;
        for (index, title) in titles.iter().enumerate() {
            let segment_rect = fastgui_core::widget::Rect {
                x: rect.x + index as f32 * segment_width,
                y: rect.y,
                width: segment_width,
                height: rect.height,
            };
            fill_rect(pixmap, segment_rect, if index == active { active_color } else { inactive_color });
            let closable = on_close.get(index).is_some_and(|c| c.is_some());
            let text_rect = if closable {
                let close = fastgui_core::widget::close_button_rect(segment_rect);
                fastgui_core::widget::Rect {
                    width: (segment_rect.width - close.width).max(0.0),
                    ..segment_rect
                }
            } else {
                segment_rect
            };
            self.draw_text(pixmap, text_rect, title, font_size, text_color);
            if closable {
                let close = fastgui_core::widget::close_button_rect(segment_rect);
                self.draw_text(pixmap, close, "×", font_size, text_color);
            }
        }
    }
}

impl Default for ChromeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_slider(
    pixmap: &mut Pixmap,
    rect: fastgui_core::widget::Rect,
    value: f32,
    min: f32,
    max: f32,
    track_color: Color,
    thumb_color: Color,
    scale: f32,
) {
    const THUMB_RADIUS: f32 = 8.0;

    let track_height = (rect.height * 0.3).max(2.0 * scale);
    let track_y = rect.y + (rect.height - track_height) / 2.0;
    fill_rect(
        pixmap,
        fastgui_core::widget::Rect { x: rect.x, y: track_y, width: rect.width, height: track_height },
        track_color,
    );

    let fraction = if max > min { ((value - min) / (max - min)).clamp(0.0, 1.0) } else { 0.0 };
    let thumb_x = rect.x + fraction * rect.width;
    let thumb_y = rect.y + rect.height / 2.0;

    let mut path_builder = tiny_skia::PathBuilder::new();
    path_builder.push_circle(thumb_x, thumb_y, THUMB_RADIUS * scale);
    if let Some(path) = path_builder.finish() {
        let mut paint = Paint::default();
        let [r, g, b, a] = thumb_color.0;
        paint.set_color_rgba8((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8);
        paint.anti_alias = true;
        pixmap.fill_path(&path, &paint, tiny_skia::FillRule::Winding, Transform::identity(), None);
    }
}

/// The sub-area of `region` a drop indicator highlights for `zone` — half the region for an
/// edge zone (matching `DropZone::classify`'s own halves, not just its outer margin, so the
/// preview reads as "the target will end up this big", not just "you're near this edge"), the
/// whole region for `Center` ("this becomes a tab").
fn drop_zone_rect(region: fastgui_core::widget::Rect, zone: DropZone) -> fastgui_core::widget::Rect {
    use fastgui_core::widget::Rect;
    match zone {
        DropZone::Center | DropZone::Float => region,
        DropZone::Left => Rect { width: region.width / 2.0, ..region },
        DropZone::Right => Rect { x: region.x + region.width / 2.0, width: region.width / 2.0, ..region },
        DropZone::Top => Rect { height: region.height / 2.0, ..region },
        DropZone::Bottom => Rect { y: region.y + region.height / 2.0, height: region.height / 2.0, ..region },
    }
}

fn scale_rect(rect: fastgui_core::widget::Rect, scale: f32) -> fastgui_core::widget::Rect {
    fastgui_core::widget::Rect {
        x: rect.x * scale,
        y: rect.y * scale,
        width: rect.width * scale,
        height: rect.height * scale,
    }
}

fn fill_rect(pixmap: &mut Pixmap, rect: fastgui_core::widget::Rect, color: Color) {
    if color.0[3] <= 0.0 || rect.width <= 0.0 || rect.height <= 0.0 {
        return;
    }
    let Some(sk_rect) = Rect::from_xywh(rect.x, rect.y, rect.width, rect.height) else { return };
    let mut paint = Paint::default();
    paint.set_color(to_tiny_skia_color(color));
    pixmap.fill_rect(sk_rect, &paint, Transform::identity(), None);
}

fn to_tiny_skia_color(color: Color) -> tiny_skia::Color {
    let [r, g, b, a] = color.0;
    tiny_skia::Color::from_rgba(r, g, b, a).unwrap_or(tiny_skia::Color::BLACK)
}

fn to_cosmic_color(color: Color) -> CosmicColor {
    let [r, g, b, a] = color.0;
    CosmicColor::rgba((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, (a * 255.0) as u8)
}
