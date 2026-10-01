//! CPU rasterizers for GPU-uploaded plot layers (`Image` / `Viewport` path).
//!
//! Line, scatter, and heatmap plots are drawn into an RGBA8 [`CpuFrame`] so the existing
//! Metal/Vulkan layer upload path composites them at interactive rates. Axes/grid are drawn
//! into the same buffer for a self-contained first slice of M7 7E.

use crate::{CpuFrame, PixelFormat};

/// Plot chrome colors and plot-area margins (pixels).
#[derive(Clone, Copy, Debug)]
pub struct PlotStyle {
    pub background: [u8; 4],
    pub grid: [u8; 4],
    pub axis: [u8; 4],
    pub series: [u8; 4],
    pub margin_left: u32,
    pub margin_right: u32,
    pub margin_top: u32,
    pub margin_bottom: u32,
}

impl Default for PlotStyle {
    fn default() -> Self {
        Self {
            background: [26, 28, 33, 255],
            grid: [55, 58, 66, 255],
            axis: [140, 145, 155, 255],
            series: [80, 170, 255, 255],
            margin_left: 48,
            margin_right: 12,
            margin_top: 12,
            margin_bottom: 28,
        }
    }
}

/// Named colormaps for heatmaps (256 RGB entries, sampled by normalized value).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    Viridis,
    Magma,
    Gray,
}

impl Colormap {
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "viridis" => Some(Self::Viridis),
            "magma" => Some(Self::Magma),
            "gray" | "grey" | "grayscale" | "greyscale" => Some(Self::Gray),
            _ => None,
        }
    }

    fn sample(self, t: f64) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        let i = (t * 255.0).round() as usize;
        match self {
            Self::Viridis => VIRIDIS[i.min(255)],
            Self::Magma => MAGMA[i.min(255)],
            Self::Gray => {
                let g = (t * 255.0).round() as u8;
                [g, g, g]
            }
        }
    }
}

/// Inclusive data range; `None` means auto from finite samples.
pub type AxisRange = Option<(f64, f64)>;

fn finite_range(values: &[f64]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if min.is_finite() && max.is_finite() {
        if (max - min).abs() < 1e-12 {
            Some((min - 1.0, max + 1.0))
        } else {
            Some((min, max))
        }
    } else {
        None
    }
}

fn resolve_range(explicit: AxisRange, values: &[f64], fallback: (f64, f64)) -> (f64, f64) {
    explicit
        .filter(|(a, b)| a.is_finite() && b.is_finite() && *a != *b)
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .or_else(|| finite_range(values))
        .unwrap_or(fallback)
}

struct PlotArea {
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    pw: u32,
    ph: u32,
}

impl PlotArea {
    fn new(width: u32, height: u32, style: &PlotStyle) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        let x0 = style.margin_left.min(width.saturating_sub(1));
        let y0 = style.margin_top.min(height.saturating_sub(1));
        let right = style.margin_right.min(width.saturating_sub(x0));
        let bottom = style.margin_bottom.min(height.saturating_sub(y0));
        let pw = width.saturating_sub(x0 + right).max(1);
        let ph = height.saturating_sub(y0 + bottom).max(1);
        Some(Self { width, height, x0, y0, pw, ph })
    }

    fn map_x(&self, v: f64, min: f64, max: f64) -> f32 {
        let t = ((v - min) / (max - min)).clamp(0.0, 1.0);
        self.x0 as f32 + t as f32 * (self.pw.saturating_sub(1) as f32)
    }

    fn map_y(&self, v: f64, min: f64, max: f64) -> f32 {
        // y grows up in data space, down in image space.
        let t = ((v - min) / (max - min)).clamp(0.0, 1.0);
        self.y0 as f32 + (1.0 - t as f32) * (self.ph.saturating_sub(1) as f32)
    }
}

fn fill(buf: &mut [u8], width: u32, color: [u8; 4]) {
    for px in buf.chunks_exact_mut(4) {
        px.copy_from_slice(&color);
    }
    let _ = width;
}

fn put(buf: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return;
    }
    let i = ((y as u32 * width + x as u32) * 4) as usize;
    buf[i..i + 4].copy_from_slice(&color);
}

fn blend(buf: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4], alpha: f32) {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 || alpha <= 0.0 {
        return;
    }
    let i = ((y as u32 * width + x as u32) * 4) as usize;
    let a = (alpha * (color[3] as f32 / 255.0)).clamp(0.0, 1.0);
    for c in 0..3 {
        let src = color[c] as f32;
        let dst = buf[i + c] as f32;
        buf[i + c] = (src * a + dst * (1.0 - a)).round().clamp(0.0, 255.0) as u8;
    }
    buf[i + 3] = 255;
}

fn fill_rect(buf: &mut [u8], width: u32, height: u32, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 4]) {
    let (x0, x1) = (x0.min(x1), x0.max(x1));
    let (y0, y1) = (y0.min(y1), y0.max(y1));
    for y in y0..=y1 {
        for x in x0..=x1 {
            put(buf, width, height, x, y, color);
        }
    }
}

fn draw_grid(buf: &mut [u8], area: &PlotArea, style: &PlotStyle) {
    // Plot background.
    fill_rect(
        buf,
        area.width,
        area.height,
        area.x0 as i32,
        area.y0 as i32,
        (area.x0 + area.pw - 1) as i32,
        (area.y0 + area.ph - 1) as i32,
        style.background,
    );
    for i in 0..=4 {
        let x = area.x0 as i32 + ((area.pw as i32 - 1) * i) / 4;
        fill_rect(buf, area.width, area.height, x, area.y0 as i32, x, (area.y0 + area.ph - 1) as i32, style.grid);
        let y = area.y0 as i32 + ((area.ph as i32 - 1) * i) / 4;
        fill_rect(buf, area.width, area.height, area.x0 as i32, y, (area.x0 + area.pw - 1) as i32, y, style.grid);
    }
    // Outer axis box.
    fill_rect(
        buf,
        area.width,
        area.height,
        area.x0 as i32,
        area.y0 as i32,
        (area.x0 + area.pw - 1) as i32,
        area.y0 as i32,
        style.axis,
    );
    fill_rect(
        buf,
        area.width,
        area.height,
        area.x0 as i32,
        (area.y0 + area.ph - 1) as i32,
        (area.x0 + area.pw - 1) as i32,
        (area.y0 + area.ph - 1) as i32,
        style.axis,
    );
    fill_rect(
        buf,
        area.width,
        area.height,
        area.x0 as i32,
        area.y0 as i32,
        area.x0 as i32,
        (area.y0 + area.ph - 1) as i32,
        style.axis,
    );
    fill_rect(
        buf,
        area.width,
        area.height,
        (area.x0 + area.pw - 1) as i32,
        area.y0 as i32,
        (area.x0 + area.pw - 1) as i32,
        (area.y0 + area.ph - 1) as i32,
        style.axis,
    );
}

fn draw_line_segment(
    buf: &mut [u8],
    width: u32,
    height: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [u8; 4],
    thickness: f32,
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = ((dx.abs().max(dy.abs())).ceil() as i32).max(1);
    let radius = (thickness * 0.5).max(0.5);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        let cx = x.round() as i32;
        let cy = y.round() as i32;
        let r = radius.ceil() as i32;
        for oy in -r..=r {
            for ox in -r..=r {
                let dist = ((ox as f32).hypot(oy as f32) - radius).abs();
                if dist < 1.0 {
                    blend(buf, width, height, cx + ox, cy + oy, color, 1.0 - dist);
                }
            }
        }
    }
}

fn draw_disc(buf: &mut [u8], width: u32, height: u32, x: f32, y: f32, radius: f32, color: [u8; 4]) {
    let cx = x.round() as i32;
    let cy = y.round() as i32;
    let r = radius.ceil() as i32;
    for oy in -r..=r {
        for ox in -r..=r {
            let dist = (ox as f32).hypot(oy as f32);
            if dist <= radius {
                put(buf, width, height, cx + ox, cy + oy, color);
            } else if dist < radius + 1.0 {
                blend(buf, width, height, cx + ox, cy + oy, color, radius + 1.0 - dist);
            }
        }
    }
}

fn empty_frame(width: u32, height: u32, style: &PlotStyle) -> CpuFrame {
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    if let Some(area) = PlotArea::new(width, height, style) {
        draw_grid(&mut data, &area, style);
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// Raster a polyline through `(x[i], y[i])` into an RGBA8 frame.
pub fn raster_line(
    width: u32,
    height: u32,
    x: &[f64],
    y: &[f64],
    x_range: AxisRange,
    y_range: AxisRange,
    style: PlotStyle,
    thickness: f32,
) -> CpuFrame {
    let n = x.len().min(y.len());
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    if n == 0 {
        return empty_frame(width, height, &style);
    }
    let xr = resolve_range(x_range, &x[..n], (0.0, 1.0));
    let yr = resolve_range(y_range, &y[..n], (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);

    let mut prev: Option<(f32, f32)> = None;
    for i in 0..n {
        if !(x[i].is_finite() && y[i].is_finite()) {
            prev = None;
            continue;
        }
        let px = area.map_x(x[i], xr.0, xr.1);
        let py = area.map_y(y[i], yr.0, yr.1);
        if let Some((ox, oy)) = prev {
            draw_line_segment(&mut data, width, height, ox, oy, px, py, style.series, thickness);
        }
        prev = Some((px, py));
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// Raster a scatter of points `(x[i], y[i])`.
pub fn raster_scatter(
    width: u32,
    height: u32,
    x: &[f64],
    y: &[f64],
    x_range: AxisRange,
    y_range: AxisRange,
    style: PlotStyle,
    point_radius: f32,
) -> CpuFrame {
    let n = x.len().min(y.len());
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    if n == 0 {
        return empty_frame(width, height, &style);
    }
    let xr = resolve_range(x_range, &x[..n], (0.0, 1.0));
    let yr = resolve_range(y_range, &y[..n], (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);
    for i in 0..n {
        if !(x[i].is_finite() && y[i].is_finite()) {
            continue;
        }
        let px = area.map_x(x[i], xr.0, xr.1);
        let py = area.map_y(y[i], yr.0, yr.1);
        draw_disc(&mut data, width, height, px, py, point_radius, style.series);
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// Raster a row-major `rows * cols` grid of values as a heatmap inside the plot frame.
pub fn raster_heatmap(
    width: u32,
    height: u32,
    values: &[f64],
    rows: usize,
    cols: usize,
    v_range: AxisRange,
    style: PlotStyle,
    colormap: Colormap,
) -> CpuFrame {
    if width == 0 || height == 0 || rows == 0 || cols == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let needed = rows.saturating_mul(cols);
    if values.len() < needed {
        return empty_frame(width, height, &style);
    }
    let vr = resolve_range(v_range, &values[..needed], (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);

    for py in 0..area.ph {
        // row 0 at top of data → top of plot area
        let row = ((py as f64 / area.ph.max(1) as f64) * rows as f64).floor() as usize;
        let row = row.min(rows - 1);
        for px in 0..area.pw {
            let col = ((px as f64 / area.pw.max(1) as f64) * cols as f64).floor() as usize;
            let col = col.min(cols - 1);
            let v = values[row * cols + col];
            let rgb = if v.is_finite() {
                let t = ((v - vr.0) / (vr.1 - vr.0)).clamp(0.0, 1.0);
                colormap.sample(t)
            } else {
                [0, 0, 0]
            };
            put(
                &mut data,
                width,
                height,
                (area.x0 + px) as i32,
                (area.y0 + py) as i32,
                [rgb[0], rgb[1], rgb[2], 255],
            );
        }
    }
    // Redraw axis box over the heatmap edge.
    fill_rect(
        &mut data,
        width,
        height,
        area.x0 as i32,
        area.y0 as i32,
        (area.x0 + area.pw - 1) as i32,
        area.y0 as i32,
        style.axis,
    );
    fill_rect(
        &mut data,
        width,
        height,
        area.x0 as i32,
        (area.y0 + area.ph - 1) as i32,
        (area.x0 + area.pw - 1) as i32,
        (area.y0 + area.ph - 1) as i32,
        style.axis,
    );
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

// Compact 256-stop Viridis / Magma samples (approximation of the matplotlib maps).
const VIRIDIS: [[u8; 3]; 256] = {
    let mut table = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        // Polynomial fit good enough for UI plots.
        let r = (0.267 + 0.005 * t + 2.1 * t * t - 1.6 * t * t * t).clamp(0.0, 1.0);
        let g = (0.005 + 1.4 * t - 0.55 * t * t).clamp(0.0, 1.0);
        let b = (0.329 + 1.4 * t - 2.4 * t * t + 1.2 * t * t * t).clamp(0.0, 1.0);
        table[i] = [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8];
        // Override ends to classic viridis endpoints.
        if i == 0 {
            table[i] = [68, 1, 84];
        }
        if i == 255 {
            table[i] = [253, 231, 37];
        }
        i += 1;
    }
    table
};

const MAGMA: [[u8; 3]; 256] = {
    let mut table = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        let r = (0.001 + 2.2 * t - 1.3 * t * t).clamp(0.0, 1.0);
        let g = (0.0 + 0.2 * t + 1.4 * t * t - 0.7 * t * t * t).clamp(0.0, 1.0);
        let b = (0.016 + 1.5 * t - 1.8 * t * t + 0.9 * t * t * t).clamp(0.0, 1.0);
        table[i] = [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8];
        if i == 0 {
            table[i] = [0, 0, 4];
        }
        if i == 255 {
            table[i] = [252, 253, 191];
        }
        i += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_raster_has_series_pixels() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let frame = raster_line(200, 120, &x, &y, None, None, PlotStyle::default(), 2.0);
        assert_eq!((frame.width, frame.height), (200, 120));
        assert_eq!(frame.data.len(), 200 * 120 * 4);
        let series = PlotStyle::default().series;
        let hits = frame.data.chunks_exact(4).filter(|px| px[0] == series[0] && px[1] == series[1]).count();
        assert!(hits > 50, "expected polyline pixels, got {hits}");
    }

    #[test]
    fn scatter_and_heatmap_smoke() {
        let x = [0.0, 1.0, 2.0];
        let y = [0.0, 1.0, 0.5];
        let scatter = raster_scatter(100, 80, &x, &y, None, None, PlotStyle::default(), 3.0);
        assert_eq!(scatter.data.len(), 100 * 80 * 4);
        let values: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let heat = raster_heatmap(120, 90, &values, 8, 8, None, PlotStyle::default(), Colormap::Viridis);
        assert_eq!(heat.data.len(), 120 * 90 * 4);
        assert_eq!(Colormap::parse("VIRIDIS"), Some(Colormap::Viridis));
        assert!(Colormap::parse("nope").is_none());
    }

    #[test]
    fn empty_data_still_draws_frame() {
        let frame = raster_line(64, 48, &[], &[], None, None, PlotStyle::default(), 1.0);
        assert_eq!(frame.data.len(), 64 * 48 * 4);
    }
}
