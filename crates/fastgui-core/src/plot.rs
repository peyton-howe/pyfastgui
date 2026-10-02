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

pub fn resolve_for_view(explicit: AxisRange, values: &[f64]) -> (f64, f64) {
    resolve_range(explicit, values, (0.0, 1.0))
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

/// One polyline or scatter series. `color` replaces [`PlotStyle::series`] for this series.
#[derive(Clone, Copy, Debug)]
pub struct PlotSeries<'a> {
    pub x: &'a [f64],
    pub y: &'a [f64],
    pub color: [u8; 4],
}

/// Raster one polyline. See [`raster_lines`] for several series on one set of axes.
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
    raster_lines(
        width,
        height,
        &[PlotSeries { x, y, color: style.series }],
        x_range,
        y_range,
        style,
        thickness,
    )
}

/// Raster several polylines into one frame. Axis ranges, when `None`, cover every finite sample.
pub fn raster_lines(
    width: u32,
    height: u32,
    series: &[PlotSeries<'_>],
    x_range: AxisRange,
    y_range: AxisRange,
    style: PlotStyle,
    thickness: f32,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for s in series {
        let n = s.x.len().min(s.y.len());
        xs.extend(s.x[..n].iter().copied());
        ys.extend(s.y[..n].iter().copied());
    }
    if xs.is_empty() {
        return empty_frame(width, height, &style);
    }
    let xr = resolve_range(x_range, &xs, (0.0, 1.0));
    let yr = resolve_range(y_range, &ys, (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);
    for s in series {
        let n = s.x.len().min(s.y.len());
        let mut prev: Option<(f32, f32)> = None;
        for i in 0..n {
            if !(s.x[i].is_finite() && s.y[i].is_finite()) {
                prev = None;
                continue;
            }
            let px = area.map_x(s.x[i], xr.0, xr.1);
            let py = area.map_y(s.y[i], yr.0, yr.1);
            if let Some((ox, oy)) = prev {
                draw_line_segment(&mut data, width, height, ox, oy, px, py, s.color, thickness);
            }
            prev = Some((px, py));
        }
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
    raster_scatters(
        width,
        height,
        &[PlotSeries { x, y, color: style.series }],
        x_range,
        y_range,
        style,
        point_radius,
    )
}

/// Raster several scatter series on one set of axes.
pub fn raster_scatters(
    width: u32,
    height: u32,
    series: &[PlotSeries<'_>],
    x_range: AxisRange,
    y_range: AxisRange,
    style: PlotStyle,
    point_radius: f32,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for s in series {
        let n = s.x.len().min(s.y.len());
        xs.extend(s.x[..n].iter().copied());
        ys.extend(s.y[..n].iter().copied());
    }
    if xs.is_empty() {
        return empty_frame(width, height, &style);
    }
    let xr = resolve_range(x_range, &xs, (0.0, 1.0));
    let yr = resolve_range(y_range, &ys, (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);
    for s in series {
        let n = s.x.len().min(s.y.len());
        for i in 0..n {
            if !(s.x[i].is_finite() && s.y[i].is_finite()) {
                continue;
            }
            let px = area.map_x(s.x[i], xr.0, xr.1);
            let py = area.map_y(s.y[i], yr.0, yr.1);
            draw_disc(&mut data, width, height, px, py, point_radius, s.color);
        }
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
    window: Option<(f64, f64, f64, f64)>,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    if rows == 0 || cols == 0 {
        return empty_frame(width, height, &style);
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

    let (col0, col1, row0, row1) = window.unwrap_or((0.0, 1.0, 0.0, 1.0));
    let (col0, col1) = if col1 > col0 { (col0, col1) } else { (0.0, 1.0) };
    let (row0, row1) = if row1 > row0 { (row0, row1) } else { (0.0, 1.0) };
    for py in 0..area.ph {
        // row 0 at top of data → top of plot area. `window` is fractions of the grid.
        let row_t = row0 + (py as f64 / area.ph.max(1) as f64) * (row1 - row0);
        let row = (row_t * rows as f64).floor() as usize;
        let row = row.min(rows - 1);
        for px in 0..area.pw {
            let col_t = col0 + (px as f64 / area.pw.max(1) as f64) * (col1 - col0);
            let col = (col_t * cols as f64).floor() as usize;
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

/// Zoom `range` around fraction `t` (0 = min, 1 = max). `factor` > 1 zooms in.
pub fn zoom_range(range: (f64, f64), t: f64, factor: f64) -> (f64, f64) {
    let (a, b) = range;
    let t = t.clamp(0.0, 1.0);
    let factor = if factor.is_finite() && factor > 1e-6 { factor } else { 1.0 };
    let anchor = a + t * (b - a);
    let span = ((b - a) / factor).abs().max(1e-12);
    (anchor - t * span, anchor + (1.0 - t) * span)
}

/// Shift a range by `delta_fraction` of its span (positive moves the window toward +data).
pub fn pan_range(range: (f64, f64), delta_fraction: f64) -> (f64, f64) {
    let (a, b) = range;
    let shift = (b - a) * delta_fraction;
    (a + shift, b + shift)
}

/// Map a widget fraction (0 at the left/top of the whole frame) into the plot area.
/// For the y axis the result is 0 at the data minimum (bottom of the plot).
pub fn axis_fraction(widget_t: f64, size: f64, margin_start: f64, margin_end: f64, y_axis: bool) -> f64 {
    let t = widget_t.clamp(0.0, 1.0);
    if size <= 1.0 {
        return if y_axis { 1.0 - t } else { t };
    }
    let pos = t * size;
    let start = margin_start.min(size);
    let end = margin_end.min((size - start).max(0.0));
    let span = (size - start - end).max(1.0);
    let along = ((pos - start) / span).clamp(0.0, 1.0);
    if y_axis { 1.0 - along } else { along }
}

fn clamp_unit_span(a: f64, b: f64) -> (f64, f64) {
    let span = (b - a).abs().clamp(1e-4, 1.0);
    let mid = ((a + b) * 0.5).clamp(span * 0.5, 1.0 - span * 0.5);
    (mid - span * 0.5, mid + span * 0.5)
}

/// Zoom a heatmap window `(col0, col1, row0, row1)` in unit grid fractions around `(tx, ty)`.
pub fn zoom_window(window: (f64, f64, f64, f64), tx: f64, ty: f64, factor: f64) -> (f64, f64, f64, f64) {
    let (x0, x1) = zoom_range((window.0, window.1), tx, factor);
    let (y0, y1) = zoom_range((window.2, window.3), ty, factor);
    let (x0, x1) = clamp_unit_span(x0, x1);
    let (y0, y1) = clamp_unit_span(y0, y1);
    (x0, x1, y0, y1)
}

/// Pan a heatmap window by fractions of its own spans, then clamp each span back into 0..1.
pub fn pan_window(window: (f64, f64, f64, f64), dx: f64, dy: f64) -> (f64, f64, f64, f64) {
    let (c0, c1) = pan_range((window.0, window.1), dx);
    let (r0, r1) = pan_range((window.2, window.3), dy);
    let (c0, c1) = clamp_unit_span(c0, c1);
    let (r0, r1) = clamp_unit_span(r0, r1);
    (c0, c1, r0, r1)
}

fn polar(cx: f32, cy: f32, r: f32, deg: f64) -> (f32, f32) {
    let rad = deg.to_radians();
    (cx + r * rad.cos() as f32, cy - r * rad.sin() as f32)
}

/// Semicircular gauge. `t` is 0..1 between `min` and `max` of the track (already normalized).
pub fn raster_gauge(width: u32, height: u32, t: f64, style: PlotStyle) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.72;
    let radius = (width.min(height) as f32) * 0.36;
    let t = t.clamp(0.0, 1.0);
    let a0 = 210.0;
    let a1 = -30.0;
    let steps = 48;
    let mut prev = polar(cx, cy, radius, a0);
    for i in 1..=steps {
        let a = a0 + (a1 - a0) * (i as f64 / steps as f64);
        let p = polar(cx, cy, radius, a);
        let along = i as f64 / steps as f64;
        let color = if along <= t + 1e-6 { style.series } else { style.grid };
        draw_line_segment(&mut data, width, height, prev.0, prev.1, p.0, p.1, color, 6.0);
        prev = p;
    }
    let needle = polar(cx, cy, radius * 0.82, a0 + (a1 - a0) * t);
    draw_line_segment(&mut data, width, height, cx, cy, needle.0, needle.1, style.axis, 2.0);
    draw_disc(&mut data, width, height, cx, cy, 4.0, style.axis);
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// One clip on a timeline track, in the same units as the timeline's duration.
#[derive(Clone, Copy, Debug)]
pub struct TimelineClip {
    pub start: f64,
    pub end: f64,
    pub color: [u8; 4],
}

/// Horizontal tracks with clips and a playhead at `time`.
pub fn raster_timeline(
    width: u32,
    height: u32,
    duration: f64,
    time: f64,
    tracks: &[Vec<TimelineClip>],
    style: PlotStyle,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    let duration = if duration.is_finite() && duration > 1e-9 { duration } else { 1.0 };
    let n = tracks.len().max(1);
    let label = 8u32;
    let top = 8u32.min(height.saturating_sub(1));
    let usable_h = height.saturating_sub(top + 8).max(1);
    let row_h = (usable_h / n as u32).max(1);
    let x0 = label.min(width.saturating_sub(1));
    let plot_w = width.saturating_sub(x0 + 8).max(1) as f64;
    for (i, clips) in tracks.iter().enumerate() {
        let y = top + i as u32 * row_h;
        let y1 = (y + row_h.saturating_sub(2)).min(height.saturating_sub(1));
        fill_rect(&mut data, width, height, x0 as i32, y as i32, (width.saturating_sub(8)) as i32, y1 as i32, style.grid);
        for clip in clips {
            if !clip.start.is_finite() || !clip.end.is_finite() || clip.end <= clip.start {
                continue;
            }
            let xa = x0 as f64 + (clip.start / duration).clamp(0.0, 1.0) * plot_w;
            let xb = x0 as f64 + (clip.end / duration).clamp(0.0, 1.0) * plot_w;
            if xb > xa {
                fill_rect(&mut data, width, height, xa.round() as i32, y as i32 + 2, xb.round() as i32, y1 as i32 - 1, clip.color);
            }
        }
    }
    let px = x0 as f64 + (time / duration).clamp(0.0, 1.0) * plot_w;
    fill_rect(&mut data, width, height, px.round() as i32, top as i32, px.round() as i32, (height.saturating_sub(6)) as i32, style.series);
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// A node in [`raster_graph`], in pixel coordinates of the frame.
#[derive(Clone, Debug)]
pub struct GraphNode {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub title: String,
    pub selected: bool,
}

fn glyph5x7(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        '-' => [0, 0, 0, 0b01110, 0, 0, 0],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

fn blit_text(buf: &mut [u8], width: u32, height: u32, x: i32, y: i32, text: &str, color: [u8; 4], scale: i32) {
    let scale = scale.max(1);
    let mut pen = x;
    for ch in text.chars().take(24) {
        let g = glyph5x7(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    let ox = pen + col * scale;
                    let oy = y + row as i32 * scale;
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put(buf, width, height, ox + sx, oy + sy, color);
                        }
                    }
                }
            }
        }
        pen += 6 * scale;
    }
}

/// Nodes and straight edges. `edges` indexes `nodes`.
pub fn raster_graph(width: u32, height: u32, nodes: &[GraphNode], edges: &[(usize, usize)], style: PlotStyle) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    for &(a, b) in edges {
        let (Some(na), Some(nb)) = (nodes.get(a), nodes.get(b)) else { continue };
        draw_line_segment(
            &mut data,
            width,
            height,
            na.x + na.w * 0.5,
            na.y + na.h * 0.5,
            nb.x + nb.w * 0.5,
            nb.y + nb.h * 0.5,
            style.axis,
            2.0,
        );
    }
    for node in nodes {
        let x0 = node.x.round() as i32;
        let y0 = node.y.round() as i32;
        let x1 = (node.x + node.w).round() as i32;
        let y1 = (node.y + node.h).round() as i32;
        let fill_color = if node.selected { style.series } else { style.grid };
        fill_rect(&mut data, width, height, x0, y0, x1, y1, fill_color);
        fill_rect(&mut data, width, height, x0, y0, x1, y0 + 2, style.axis);
        blit_text(&mut data, width, height, x0 + 8, y0 + 10, &node.title, [230, 232, 236, 255], 2);
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

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
        let heat = raster_heatmap(120, 90, &values, 8, 8, None, PlotStyle::default(), Colormap::Viridis, None);
        assert_eq!(heat.data.len(), 120 * 90 * 4);
        assert_eq!(Colormap::parse("VIRIDIS"), Some(Colormap::Viridis));
        assert!(Colormap::parse("nope").is_none());
    }

    #[test]
    fn empty_data_still_draws_frame() {
        let frame = raster_line(64, 48, &[], &[], None, None, PlotStyle::default(), 1.0);
        assert_eq!(frame.data.len(), 64 * 48 * 4);
    }

    #[test]
    fn two_series_use_their_own_colors() {
        let x = [0.0, 1.0];
        let y_lo = [0.0, 0.0];
        let y_hi = [1.0, 1.0];
        let red = [220, 40, 40, 255];
        let frame = raster_lines(
            80,
            60,
            &[
                PlotSeries { x: &x, y: &y_lo, color: PlotStyle::default().series },
                PlotSeries { x: &x, y: &y_hi, color: red },
            ],
            None,
            None,
            PlotStyle::default(),
            2.0,
        );
        let reds = frame.data.chunks_exact(4).filter(|px| px[0] == red[0] && px[1] == red[1]).count();
        assert!(reds > 5, "second series should paint its own color, got {reds}");
    }

    #[test]
    fn zoom_keeps_the_anchor_and_pan_shifts() {
        let (a, b) = zoom_range((0.0, 10.0), 0.0, 2.0);
        assert!((a - 0.0).abs() < 1e-9, "{a}");
        assert!((b - 5.0).abs() < 1e-9, "{b}");
        let (a, b) = pan_range((0.0, 10.0), 0.1);
        assert!((a - 1.0).abs() < 1e-9 && (b - 11.0).abs() < 1e-9);
        let (c0, c1, r0, r1) = zoom_window((0.0, 1.0, 0.0, 1.0), 0.5, 0.5, 2.0);
        assert!(c1 - c0 < 0.6 && r1 - r0 < 0.6);
        assert!(c0 > 0.1 && r0 > 0.1);
        let (c0, c1, _, _) = pan_window((0.0, 0.5, 0.0, 0.5), -1.0, 0.0);
        assert!(c0 >= -1e-9 && c1 <= 1.0 + 1e-9 && c1 > c0);
    }

    #[test]
    fn gauge_timeline_and_graph_paint() {
        let gauge = raster_gauge(80, 60, 0.5, PlotStyle::default());
        assert_eq!(gauge.data.len(), 80 * 60 * 4);
        let clips = vec![TimelineClip { start: 1.0, end: 3.0, color: [200, 80, 40, 255] }];
        let timeline = raster_timeline(100, 40, 10.0, 2.0, &[clips], PlotStyle::default());
        assert!(timeline.data.chunks_exact(4).any(|px| px[0] == 200));
        let nodes = vec![
            GraphNode { x: 8.0, y: 8.0, w: 70.0, h: 28.0, title: "SRC".into(), selected: true },
            GraphNode { x: 100.0, y: 20.0, w: 70.0, h: 28.0, title: "OUT".into(), selected: false },
        ];
        let graph = raster_graph(180, 70, &nodes, &[(0, 1)], PlotStyle::default());
        assert_eq!(graph.data.len(), 180 * 70 * 4);
        // Title pixels are near-white inside the first node.
        let titled = graph.data.chunks_exact(4).filter(|px| px[0] > 220 && px[1] > 220).count();
        assert!(titled > 10, "node title should be drawn, got {titled}");
    }
}
