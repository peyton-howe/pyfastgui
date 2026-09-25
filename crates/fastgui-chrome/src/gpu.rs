//! The chrome as GPU draw data: the same display list the CPU path paints, emitted as instanced
//! quads (`ChromeQuad`) plus uploads into a glyph atlas. Text is still shaped and rasterized on
//! the CPU, once per string (the text run cache); the GPU composites. Per frame the CPU cost is
//! building the display list and ~64 bytes per quad, independent of window size, and a change
//! that moves everything (scroll, resize, splitter drag) re-sends quads, not pixels.
//!
//! What each quad kind does in the shader mirrors the CPU painter closely:
//! - solid rects get analytic box-filter coverage at fractional edges (tiny-skia's AA fill),
//! - circles get analytic edge coverage (close to, not identical with, tiny-skia's path AA),
//! - sprites (text runs) are copied 1:1 from the atlas at whole pixels, so text is identical.
//!
//! `emulate` (tests only) is a CPU reference of those shader rules, checked against the CPU
//! painter.

use std::collections::HashMap;
use std::sync::Arc;

use fastgui_core::{AtlasUpload, ChromeQuad, ChromeQuads, PixelRect, QUAD_CIRCLE, QUAD_SOLID, QUAD_SPRITE};

use super::{Color, Item, Op, Sprite, BACKGROUND};

const ATLAS_MIN_SIZE: u32 = 512;
/// 4096² RGBA8 = 64 MiB. Vulkan guarantees 2D images at least this large, and a window's text
/// fits many times over; past it, runs that don't fit are skipped (and reported once).
const ATLAS_MAX_SIZE: u32 = 4096;

/// Shelf packer: rows of sprites, each row as tall as the first sprite placed in it. Individual
/// slots are never freed; when the atlas fills, `ChromeRenderer::build_quads` repacks it from
/// scratch with only the sprites the current frame draws (growing it if they still don't fit).
#[derive(Default)]
pub(crate) struct Atlas {
    /// 0 before the first frame.
    size: u32,
    shelves: Vec<Shelf>,
    next_y: u32,
    /// Sprite id → top-left of its slot.
    slots: HashMap<u64, (u32, u32)>,
}

struct Shelf {
    y: u32,
    height: u32,
    next_x: u32,
}

impl Atlas {
    fn reset(&mut self, size: u32) {
        self.size = size;
        self.shelves.clear();
        self.next_y = 0;
        self.slots.clear();
    }

    fn alloc(&mut self, width: u32, height: u32) -> Option<(u32, u32)> {
        if width > self.size || height > self.size {
            return None;
        }
        // The shortest shelf it fits in, so short runs don't eat tall rows.
        let shelf = self
            .shelves
            .iter_mut()
            .filter(|s| s.height >= height && s.next_x + width <= self.size)
            .min_by_key(|s| s.height);
        if let Some(shelf) = shelf {
            let slot = (shelf.next_x, shelf.y);
            shelf.next_x += width;
            return Some(slot);
        }
        if self.next_y + height > self.size {
            return None;
        }
        let y = self.next_y;
        self.next_y += height;
        self.shelves.push(Shelf { y, height, next_x: width });
        Some((0, y))
    }
}

/// Per-renderer state of the GPU path.
#[derive(Default)]
pub(crate) struct GpuState {
    pub(crate) atlas: Atlas,
    pub(crate) quads: Vec<ChromeQuad>,
    uploads: Vec<(PixelRect, Arc<Sprite>)>,
    /// Physical size of the last emitted frame.
    size: Option<(u32, u32)>,
    /// Emit the next frame even if the display list didn't change (after `invalidate`).
    pub(crate) force: bool,
    reported_overflow: bool,
}

impl GpuState {
    /// Give every sprite the visible items draw an atlas slot, queuing uploads for new ones.
    /// `false` when the atlas ran out of room (some sprites have no slot).
    fn place_sprites(&mut self, items: &[Item]) -> bool {
        for op in items.iter().filter(|item| !item.bounds.is_empty()).flat_map(|item| &item.ops) {
            let Op::Text { run, .. } = op else { continue };
            if self.atlas.slots.contains_key(&run.id) {
                continue;
            }
            let Some((x, y)) = self.atlas.alloc(run.width, run.height) else { return false };
            self.atlas.slots.insert(run.id, (x, y));
            self.uploads.push((PixelRect { x, y, width: run.width, height: run.height }, run.clone()));
        }
        true
    }

    fn emit(&mut self, items: &[Item], width: u32, height: u32) {
        self.quads.clear();
        self.quads.push(solid([0.0, 0.0, width as f32, height as f32], BACKGROUND));
        for item in items.iter().filter(|item| !item.bounds.is_empty()) {
            for op in &item.ops {
                match op {
                    Op::Fill { left, top, right, bottom, color } => {
                        self.quads.push(solid([*left, *top, *right, *bottom], *color));
                    }
                    Op::Circle { cx, cy, radius, color, .. } => self.quads.push(ChromeQuad {
                        rect: [cx - radius, cy - radius, cx + radius, cy + radius],
                        color: color.0,
                        params: [*cx, *cy, *radius, 0.0],
                        kind: QUAD_CIRCLE,
                        _pad: [0; 3],
                    }),
                    Op::Text { run, x, y } => {
                        // No slot only when the atlas overflowed at its maximum size.
                        let Some(&(ax, ay)) = self.atlas.slots.get(&run.id) else { continue };
                        let (left, top) = ((x + run.x) as f32, (y + run.y) as f32);
                        self.quads.push(ChromeQuad {
                            rect: [left, top, left + run.width as f32, top + run.height as f32],
                            color: [1.0; 4],
                            params: [ax as f32, ay as f32, 0.0, 0.0],
                            kind: QUAD_SPRITE,
                            _pad: [0; 3],
                        });
                    }
                }
            }
        }
    }
}

fn solid(rect: [f32; 4], color: Color) -> ChromeQuad {
    ChromeQuad { rect, color: color.0, params: [0.0; 4], kind: QUAD_SOLID, _pad: [0; 3] }
}

impl super::ChromeRenderer {
    /// The GPU counterpart of `rasterize`: same inputs, but returns quads and atlas uploads
    /// for `fastgui-app`'s backends to draw instead of pixels. Returns `None` when nothing
    /// changed since the previous call (the backend keeps drawing the quads it has).
    pub fn build_quads(
        &mut self,
        tree: &fastgui_core::widget::WidgetTree,
        width: u32,
        height: u32,
        drop_indicator: Option<(super::WidgetRect, fastgui_core::widget::DropZone)>,
        scale: f32,
    ) -> Option<ChromeQuads<'_>> {
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let (width, height) = (width.max(1), height.max(1));
        let window = PixelRect { x: 0, y: 0, width, height };

        self.text_cache.generation += 1;
        let items = self.build_items(tree, drop_indicator, scale, window);
        self.text_cache.evict();

        let gpu = &mut self.gpu;
        let unchanged = !gpu.force
            && gpu.size == Some((width, height))
            && super::diff_items(&self.items, &items).is_some_and(|damage| damage.is_empty());
        self.items = items;
        if unchanged {
            return None;
        }
        gpu.size = Some((width, height));
        gpu.force = false;

        gpu.uploads.clear();
        let mut repacked = false;
        if !gpu.place_sprites(&self.items) {
            // Full: repack only what this frame draws, at the current size first, then bigger.
            repacked = true;
            let mut size = gpu.atlas.size.max(ATLAS_MIN_SIZE);
            loop {
                gpu.atlas.reset(size);
                gpu.uploads.clear();
                if gpu.place_sprites(&self.items) {
                    break;
                }
                if size >= ATLAS_MAX_SIZE {
                    if !gpu.reported_overflow {
                        gpu.reported_overflow = true;
                        eprintln!(
                            "fastgui: chrome text doesn't fit a {ATLAS_MAX_SIZE}² atlas; some labels won't be drawn"
                        );
                    }
                    break;
                }
                size *= 2;
            }
        }
        gpu.emit(&self.items, width, height);

        Some(ChromeQuads {
            width,
            height,
            quads: &gpu.quads,
            atlas_size: gpu.atlas.size,
            atlas_repacked: repacked,
            atlas_uploads: gpu
                .uploads
                .iter()
                .map(|(rect, sprite)| AtlasUpload { rect: *rect, pixels: &sprite.pixels })
                .collect(),
        })
    }
}

/// CPU reference of what the backends' quad shaders compute, for tests: draws `quads` into a
/// premultiplied RGBA8 `width`x`height` buffer with the same per-kind rules and ONE /
/// ONE_MINUS_SRC_ALPHA blending, sampling `atlas` (`atlas_size`² RGBA8).
#[cfg(test)]
pub(crate) fn emulate(quads: &[ChromeQuad], width: u32, height: u32, atlas: &[u8], atlas_size: u32) -> Vec<u8> {
    let mut out = vec![0f32; width as usize * height as usize * 4];
    for quad in quads {
        let pad = if quad.kind == QUAD_SPRITE { 0.0 } else { 1.0 };
        let [l, t, r, b] = quad.rect;
        let x0 = ((l - pad).floor().max(0.0)) as u32;
        let y0 = ((t - pad).floor().max(0.0)) as u32;
        let x1 = ((r + pad).ceil().min(width as f32)) as u32;
        let y1 = ((b + pad).ceil().min(height as f32)) as u32;
        for py in y0..y1 {
            for px in x0..x1 {
                let (fx, fy) = (px as f32, py as f32);
                let src = match quad.kind {
                    QUAD_SOLID => {
                        let cx = (r.min(fx + 1.0) - l.max(fx)).clamp(0.0, 1.0);
                        let cy = (b.min(fy + 1.0) - t.max(fy)).clamp(0.0, 1.0);
                        premultiply(quad.color, cx * cy)
                    }
                    QUAD_CIRCLE => {
                        let [ccx, ccy, radius, _] = quad.params;
                        let d = ((fx + 0.5 - ccx).powi(2) + (fy + 0.5 - ccy).powi(2)).sqrt();
                        premultiply(quad.color, (radius + 0.5 - d).clamp(0.0, 1.0))
                    }
                    _ => {
                        let ax = quad.params[0] as u32 + (px - l as u32);
                        let ay = quad.params[1] as u32 + (py - t as u32);
                        let i = (ay * atlas_size + ax) as usize * 4;
                        std::array::from_fn(|c| f32::from(atlas[i + c]) / 255.0)
                    }
                };
                let i = (py * width + px) as usize * 4;
                let inv = 1.0 - src[3];
                for c in 0..4 {
                    out[i + c] = src[c] + out[i + c] * inv;
                }
            }
        }
    }
    out.iter().map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8).collect()
}

#[cfg(test)]
fn premultiply(color: [f32; 4], coverage: f32) -> [f32; 4] {
    let a = color[3] * coverage;
    [color[0] * a, color[1] * a, color[2] * a, a]
}

#[cfg(test)]
pub(crate) fn apply_uploads(atlas: &mut Vec<u8>, atlas_size: u32, quads: &ChromeQuads<'_>) {
    if quads.atlas_size != atlas_size || atlas.len() != (atlas_size * atlas_size * 4) as usize {
        *atlas = vec![0; (quads.atlas_size * quads.atlas_size * 4) as usize];
    }
    let stride = quads.atlas_size as usize * 4;
    for upload in &quads.atlas_uploads {
        let row = upload.rect.width as usize * 4;
        for (dy, src) in upload.pixels.chunks_exact(row).enumerate() {
            let start = (upload.rect.y as usize + dy) * stride + upload.rect.x as usize * 4;
            atlas[start..start + row].copy_from_slice(src);
        }
    }
}
