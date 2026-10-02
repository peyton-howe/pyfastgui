//! CPU rasterizers for GPU-uploaded plot layers (`Image` / `Viewport` path).
//!
//! Line, scatter, bar, histogram, heatmap, contour, and projected 3-D surface/mesh plots are
//! drawn into an RGBA8 [`CpuFrame`]. The GPU only composites that image — there is no geometry
//! shader and no vertex pipeline for plots.

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

/// Bin finite samples into `bins` equal-width buckets. The last edge is inclusive.
/// Returns `(counts, edges)` with `edges.len() == counts.len() + 1`.
pub fn bin_counts(values: &[f64], bins: usize, range: Option<(f64, f64)>) -> (Vec<f64>, Vec<f64>) {
    let bins = bins.max(1);
    let (lo, hi) = range
        .filter(|(a, b)| a.is_finite() && b.is_finite() && a != b)
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .or_else(|| finite_range(values))
        .unwrap_or((0.0, 1.0));
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lo + (hi - lo) * (i as f64 / bins as f64));
    }
    let mut counts = vec![0.0; bins];
    let span = (hi - lo).max(1e-12);
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        let mut i = ((v - lo) / span * bins as f64).floor() as isize;
        if i == bins as isize {
            i -= 1;
        }
        if (0..bins as isize).contains(&i) {
            counts[i as usize] += 1.0;
        }
    }
    (counts, edges)
}

fn map_x_raw(area: &PlotArea, v: f64, min: f64, max: f64) -> f32 {
    let t = (v - min) / (max - min);
    area.x0 as f32 + t as f32 * (area.pw.saturating_sub(1) as f32)
}

fn map_y_raw(area: &PlotArea, v: f64, min: f64, max: f64) -> f32 {
    let t = (v - min) / (max - min);
    area.y0 as f32 + (1.0 - t as f32) * (area.ph.saturating_sub(1) as f32)
}

fn clipped_rect(x0: i32, y0: i32, x1: i32, y1: i32, width: u32, height: u32) -> Option<(i32, i32, i32, i32)> {
    let (mut x0, mut x1) = (x0.min(x1), x0.max(x1));
    let (mut y0, mut y1) = (y0.min(y1), y0.max(y1));
    x0 = x0.max(0);
    y0 = y0.max(0);
    x1 = x1.min(width as i32 - 1);
    y1 = y1.min(height as i32 - 1);
    if x0 > x1 || y0 > y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

/// Vertical bars from `x0[i]`..`x1[i]` up (or down) to `height[i]`, baseline at 0.
pub fn raster_bars(
    width: u32,
    height: u32,
    x0: &[f64],
    x1: &[f64],
    bar_height: &[f64],
    x_range: AxisRange,
    y_range: AxisRange,
    style: PlotStyle,
    color: [u8; 4],
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let n = x0.len().min(x1.len()).min(bar_height.len());
    if n == 0 {
        return empty_frame(width, height, &style);
    }
    let mut xs = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n + 1);
    ys.push(0.0);
    for i in 0..n {
        if x0[i].is_finite() {
            xs.push(x0[i]);
        }
        if x1[i].is_finite() {
            xs.push(x1[i]);
        }
        if bar_height[i].is_finite() {
            ys.push(bar_height[i]);
        }
    }
    let xr = resolve_range(x_range, &xs, (0.0, 1.0));
    let yr = resolve_range(y_range, &ys, (0.0, 1.0));
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    draw_grid(&mut data, &area, &style);
    let y_base = map_y_raw(&area, 0.0, yr.0, yr.1).round() as i32;
    for i in 0..n {
        if !(x0[i].is_finite() && x1[i].is_finite() && bar_height[i].is_finite()) {
            continue;
        }
        let xa = map_x_raw(&area, x0[i], xr.0, xr.1).round() as i32;
        let xb = map_x_raw(&area, x1[i], xr.0, xr.1).round() as i32;
        let yt = map_y_raw(&area, bar_height[i], yr.0, yr.1).round() as i32;
        let Some((cx0, cy0, cx1, cy1)) = clipped_rect(xa, y_base, xb, yt, width, height) else { continue };
        // Keep bars inside the plot frame so a zoomed-out bar cannot paint the margins.
        let px0 = cx0.max(area.x0 as i32);
        let py0 = cy0.max(area.y0 as i32);
        let px1 = cx1.min((area.x0 + area.pw - 1) as i32);
        let py1 = cy1.min((area.y0 + area.ph - 1) as i32);
        if px0 <= px1 && py0 <= py1 {
            fill_rect(&mut data, width, height, px0, py0, px1, py1, color);
        }
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

fn clip_segment(mut x0: f32, mut y0: f32, mut x1: f32, mut y1: f32, xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Option<(f32, f32, f32, f32)> {
    fn code(x: f32, y: f32, xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> u8 {
        let mut c = 0;
        if x < xmin {
            c |= 1;
        }
        if x > xmax {
            c |= 2;
        }
        if y < ymin {
            c |= 4;
        }
        if y > ymax {
            c |= 8;
        }
        c
    }
    for _ in 0..8 {
        let c0 = code(x0, y0, xmin, ymin, xmax, ymax);
        let c1 = code(x1, y1, xmin, ymin, xmax, ymax);
        if c0 | c1 == 0 {
            return Some((x0, y0, x1, y1));
        }
        if c0 & c1 != 0 {
            return None;
        }
        let c = if c0 != 0 { c0 } else { c1 };
        let (x, y) = if c & 1 != 0 || c & 2 != 0 {
            let dx = x1 - x0;
            if dx.abs() < 1e-6 {
                return None;
            }
            let edge = if c & 1 != 0 { xmin } else { xmax };
            let t = (edge - x0) / dx;
            (edge, y0 + t * (y1 - y0))
        } else {
            let dy = y1 - y0;
            if dy.abs() < 1e-6 {
                return None;
            }
            let edge = if c & 4 != 0 { ymin } else { ymax };
            let t = (edge - y0) / dy;
            (x0 + t * (x1 - x0), edge)
        };
        if c == c0 {
            x0 = x;
            y0 = y;
        } else {
            x1 = x;
            y1 = y;
        }
    }
    None
}

fn lerp_edge(a: f64, b: f64, va: f64, vb: f64, level: f64) -> f64 {
    let d = vb - va;
    if d.abs() < 1e-15 {
        return (a + b) * 0.5;
    }
    a + (level - va) / d * (b - a)
}

/// Isolines of a row-major grid (row 0 at the top, same window as [`raster_heatmap`]).
/// `levels` are data values. `filled` paints colormap bands under the lines.
pub fn raster_contour(
    width: u32,
    height: u32,
    values: &[f64],
    rows: usize,
    cols: usize,
    levels: &[f64],
    v_range: AxisRange,
    style: PlotStyle,
    colormap: Colormap,
    filled: bool,
    window: Option<(f64, f64, f64, f64)>,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    if rows < 2 || cols < 2 || values.len() < rows.saturating_mul(cols) {
        return empty_frame(width, height, &style);
    }
    let needed = rows * cols;
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

    let mut sorted: Vec<f64> = levels.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    if filled {
        for py in 0..area.ph {
            let row_t = row0 + (py as f64 / area.ph.max(1) as f64) * (row1 - row0);
            let row = ((row_t * rows as f64).floor() as usize).min(rows - 1);
            for px in 0..area.pw {
                let col_t = col0 + (px as f64 / area.pw.max(1) as f64) * (col1 - col0);
                let col = ((col_t * cols as f64).floor() as usize).min(cols - 1);
                let v = values[row * cols + col];
                let rgb = if v.is_finite() && !sorted.is_empty() {
                    let band = sorted.iter().position(|lvl| v < *lvl).unwrap_or(sorted.len());
                    let t = band as f64 / sorted.len() as f64;
                    colormap.sample(t)
                } else if v.is_finite() {
                    let t = ((v - vr.0) / (vr.1 - vr.0)).clamp(0.0, 1.0);
                    colormap.sample(t)
                } else {
                    [0, 0, 0]
                };
                put(&mut data, width, height, (area.x0 + px) as i32, (area.y0 + py) as i32, [rgb[0], rgb[1], rgb[2], 255]);
            }
        }
    }

    let to_screen = |gc: f64, gr: f64| -> (f32, f32) {
        let fx = (gc / cols as f64 - col0) / (col1 - col0);
        let fy = (gr / rows as f64 - row0) / (row1 - row0);
        let x = area.x0 as f32 + fx as f32 * (area.pw.saturating_sub(1) as f32);
        let y = area.y0 as f32 + fy as f32 * (area.ph.saturating_sub(1) as f32);
        (x, y)
    };
    let xmin = area.x0 as f32;
    let ymin = area.y0 as f32;
    let xmax = (area.x0 + area.pw - 1) as f32;
    let ymax = (area.y0 + area.ph - 1) as f32;
    let line_fallback = [230, 232, 236, 255];

    for &level in &sorted {
        let color = if filled {
            line_fallback
        } else {
            let t = ((level - vr.0) / (vr.1 - vr.0)).clamp(0.0, 1.0);
            let rgb = colormap.sample(t);
            [rgb[0], rgb[1], rgb[2], 255]
        };
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let tl = values[r * cols + c];
                let tr = values[r * cols + c + 1];
                let bl = values[(r + 1) * cols + c];
                let br = values[(r + 1) * cols + c + 1];
                if !(tl.is_finite() && tr.is_finite() && bl.is_finite() && br.is_finite()) {
                    continue;
                }
                let mut bits = 0u8;
                if tl >= level {
                    bits |= 1;
                }
                if tr >= level {
                    bits |= 2;
                }
                if br >= level {
                    bits |= 4;
                }
                if bl >= level {
                    bits |= 8;
                }
                if bits == 0 || bits == 15 {
                    continue;
                }
                let top = (lerp_edge(c as f64, c as f64 + 1.0, tl, tr, level), r as f64);
                let right = (c as f64 + 1.0, lerp_edge(r as f64, r as f64 + 1.0, tr, br, level));
                let bottom = (lerp_edge(c as f64, c as f64 + 1.0, bl, br, level), r as f64 + 1.0);
                let left = (c as f64, lerp_edge(r as f64, r as f64 + 1.0, tl, bl, level));
                let segs: &[((f64, f64), (f64, f64))] = match bits {
                    1 | 14 => &[(left, top)],
                    2 | 13 => &[(top, right)],
                    3 | 12 => &[(left, right)],
                    4 | 11 => &[(right, bottom)],
                    6 | 9 => &[(top, bottom)],
                    7 | 8 => &[(bottom, left)],
                    5 => &[(left, top), (right, bottom)],
                    10 => &[(top, right), (bottom, left)],
                    _ => &[],
                };
                for &(a, b) in segs {
                    let (x0, y0) = to_screen(a.0, a.1);
                    let (x1, y1) = to_screen(b.0, b.1);
                    if let Some((x0, y0, x1, y1)) = clip_segment(x0, y0, x1, y1, xmin, ymin, xmax, ymax) {
                        draw_line_segment(&mut data, width, height, x0, y0, x1, y1, color, 1.5);
                    }
                }
            }
        }
    }
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

/// Orbit camera for [`raster_surface`] and [`raster_mesh`]. Angles are radians.
#[derive(Clone, Copy, Debug)]
pub struct MeshView {
    pub yaw: f64,
    pub pitch: f64,
    pub zoom: f64,
}

impl Default for MeshView {
    fn default() -> Self {
        Self { yaw: -0.8, pitch: 0.55, zoom: 1.0 }
    }
}

/// Returns `(screen_x, screen_up, depth)`. Z is up: yaw spins around it, pitch tips it toward the camera.
fn rotate_view(x: f64, y: f64, z: f64, view: MeshView) -> (f64, f64, f64) {
    let (ys, yc) = view.yaw.sin_cos();
    let x1 = x * yc - y * ys;
    let y1 = x * ys + y * yc;
    let (ps, pc) = view.pitch.sin_cos();
    let up = z * pc - y1 * ps;
    let depth = z * ps + y1 * pc;
    (x1, up, depth)
}

fn shade(nx: f64, ny: f64, nz: f64) -> f32 {
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-12 {
        return 0.55;
    }
    let (lx, ly, lz) = (0.25_f64, 0.45_f64, 0.86_f64);
    let ll = (lx * lx + ly * ly + lz * lz).sqrt();
    let d = ((nx * lx + ny * ly + nz * lz) / (len * ll)).abs();
    (d as f32) * 0.72 + 0.28
}

fn fill_triangle(buf: &mut [u8], width: u32, height: u32, p: [(f32, f32); 3], color: [u8; 4], clip: (i32, i32, i32, i32)) {
    let (cx0, cy0, cx1, cy1) = clip;
    let mut minx = p[0].0.min(p[1].0).min(p[2].0).floor() as i32;
    let mut maxx = p[0].0.max(p[1].0).max(p[2].0).ceil() as i32;
    let mut miny = p[0].1.min(p[1].1).min(p[2].1).floor() as i32;
    let mut maxy = p[0].1.max(p[1].1).max(p[2].1).ceil() as i32;
    minx = minx.max(cx0);
    miny = miny.max(cy0);
    maxx = maxx.min(cx1);
    maxy = maxy.min(cy1);
    if minx > maxx || miny > maxy {
        return;
    }
    let edge = |a: (f32, f32), b: (f32, f32), x: f32, y: f32| (x - a.0) * (b.1 - a.1) - (y - a.1) * (b.0 - a.0);
    let area = edge(p[0], p[1], p[2].0, p[2].1);
    if area.abs() < 0.5 {
        return;
    }
    for y in miny..=maxy {
        for x in minx..=maxx {
            let xf = x as f32 + 0.5;
            let yf = y as f32 + 0.5;
            let w0 = edge(p[1], p[2], xf, yf);
            let w1 = edge(p[2], p[0], xf, yf);
            let w2 = edge(p[0], p[1], xf, yf);
            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                put(buf, width, height, x, y, color);
            }
        }
    }
}

struct ProjectedTri {
    p: [(f32, f32); 3],
    z: f32,
    color: [u8; 4],
}

fn raster_projected(
    width: u32,
    height: u32,
    verts: &[[f64; 3]],
    faces: &[[u32; 3]],
    style: PlotStyle,
    colormap: Colormap,
    view: MeshView,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    if verts.is_empty() || faces.is_empty() {
        draw_grid(&mut data, &area, &style);
        return CpuFrame { width, height, format: PixelFormat::Rgba8, data };
    }
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for v in verts {
        for k in 0..3 {
            if v[k].is_finite() {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    let extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]).max(1e-9);
    let zmin = min[2];
    let zspan = (max[2] - min[2]).max(1e-12);
    let zoom = if view.zoom.is_finite() && view.zoom > 0.0 { view.zoom } else { 1.0 };
    // Normalized coordinates span about 1 on the longest axis (-0.5..0.5), so this fills the frame.
    let scale = (area.pw.min(area.ph) as f64) * 0.78 * zoom;
    let ox = area.x0 as f64 + area.pw as f64 * 0.5;
    let oy = area.y0 as f64 + area.ph as f64 * 0.5;
    let project = |v: [f64; 3]| -> (f32, f32, f64) {
        let n = [(v[0] - center[0]) / extent, (v[1] - center[1]) / extent, (v[2] - center[2]) / extent];
        let (x, y, z) = rotate_view(n[0], n[1], n[2], view);
        ((ox + x * scale) as f32, (oy - y * scale) as f32, z)
    };
    let mut tris = Vec::with_capacity(faces.len());
    for face in faces {
        let Some(a) = verts.get(face[0] as usize) else { continue };
        let Some(b) = verts.get(face[1] as usize) else { continue };
        let Some(c) = verts.get(face[2] as usize) else { continue };
        if !(a.iter().all(|v| v.is_finite()) && b.iter().all(|v| v.is_finite()) && c.iter().all(|v| v.is_finite())) {
            continue;
        }
        let pa = project(*a);
        let pb = project(*b);
        let pc = project(*c);
        let ux = (pb.0 - pa.0) as f64;
        let uy = (pb.1 - pa.1) as f64;
        let uz = pb.2 - pa.2;
        let vx = (pc.0 - pa.0) as f64;
        let vy = (pc.1 - pa.1) as f64;
        let vz = pc.2 - pa.2;
        let nx = uy * vz - uz * vy;
        let ny = uz * vx - ux * vz;
        let nz = ux * vy - uy * vx;
        let s = shade(nx, ny, nz);
        let zc = (a[2] + b[2] + c[2]) / 3.0;
        let t = ((zc - zmin) / zspan).clamp(0.0, 1.0);
        let rgb = colormap.sample(t);
        let color = [
            (rgb[0] as f32 * s).round() as u8,
            (rgb[1] as f32 * s).round() as u8,
            (rgb[2] as f32 * s).round() as u8,
            255,
        ];
        tris.push(ProjectedTri {
            p: [(pa.0, pa.1), (pb.0, pb.1), (pc.0, pc.1)],
            z: ((pa.2 + pb.2 + pc.2) / 3.0) as f32,
            color,
        });
    }
    tris.sort_by(|a, b| a.z.partial_cmp(&b.z).unwrap_or(std::cmp::Ordering::Equal));
    let clip = (area.x0 as i32, area.y0 as i32, (area.x0 + area.pw - 1) as i32, (area.y0 + area.ph - 1) as i32);
    for tri in &tris {
        fill_triangle(&mut data, width, height, tri.p, tri.color, clip);
    }
    let (x0, y0, x1, y1) = clip;
    fill_rect(&mut data, width, height, x0, y0, x1, y0, style.axis);
    fill_rect(&mut data, width, height, x0, y1, x1, y1, style.axis);
    fill_rect(&mut data, width, height, x0, y0, x0, y1, style.axis);
    fill_rect(&mut data, width, height, x1, y0, x1, y1, style.axis);
    CpuFrame { width, height, format: PixelFormat::Rgba8, data }
}

const SURFACE_CAP: usize = 96;

fn axis_at(axis: Option<&[f64]>, rows: usize, cols: usize, r: usize, c: usize, along_cols: bool) -> f64 {
    let Some(axis) = axis else {
        return if along_cols { c as f64 } else { r as f64 };
    };
    if axis.len() == rows.saturating_mul(cols) {
        return axis[r * cols + c];
    }
    if along_cols && axis.len() == cols {
        return axis[c];
    }
    if !along_cols && axis.len() == rows {
        return axis[r];
    }
    f64::NAN
}

/// 3-D surface. `z` is row-major. `x`/`y` are omitted (column and row index), a 1-D axis
/// (`x` length `cols`, `y` length `rows`), or a grid of the same shape as `z`.
pub fn raster_surface(
    width: u32,
    height: u32,
    z: &[f64],
    rows: usize,
    cols: usize,
    x: Option<&[f64]>,
    y: Option<&[f64]>,
    style: PlotStyle,
    colormap: Colormap,
    view: MeshView,
) -> CpuFrame {
    if rows < 2 || cols < 2 || z.len() < rows.saturating_mul(cols) {
        return raster_projected(width, height, &[], &[], style, colormap, view);
    }
    let rs = (rows / SURFACE_CAP).max(1);
    let cs = (cols / SURFACE_CAP).max(1);
    let nr = ((rows - 1) / rs) + 1;
    let nc = ((cols - 1) / cs) + 1;
    let mut verts = Vec::with_capacity(nr * nc);
    let mut index_of = vec![u32::MAX; nr * nc];
    for ir in 0..nr {
        let r = (ir * rs).min(rows - 1);
        for ic in 0..nc {
            let c = (ic * cs).min(cols - 1);
            let px = axis_at(x, rows, cols, r, c, true);
            let py = axis_at(y, rows, cols, r, c, false);
            let h = z[r * cols + c];
            if !(px.is_finite() && py.is_finite() && h.is_finite()) {
                continue;
            }
            index_of[ir * nc + ic] = verts.len() as u32;
            verts.push([px, py, h]);
        }
    }
    let mut faces = Vec::new();
    for ir in 0..nr.saturating_sub(1) {
        for ic in 0..nc.saturating_sub(1) {
            let i00 = index_of[ir * nc + ic];
            let i10 = index_of[ir * nc + ic + 1];
            let i01 = index_of[(ir + 1) * nc + ic];
            let i11 = index_of[(ir + 1) * nc + ic + 1];
            if i00 == u32::MAX || i10 == u32::MAX || i01 == u32::MAX || i11 == u32::MAX {
                continue;
            }
            faces.push([i00, i10, i11]);
            faces.push([i00, i11, i01]);
        }
    }
    raster_projected(width, height, &verts, &faces, style, colormap, view)
}

/// Triangle mesh. `vertices` are xyz; `faces` indexes them. Colored by z and flat-shaded.
pub fn raster_mesh(
    width: u32,
    height: u32,
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    style: PlotStyle,
    colormap: Colormap,
    view: MeshView,
) -> CpuFrame {
    raster_projected(width, height, vertices, faces, style, colormap, view)
}

const SCATTER3D_CAP: usize = 8_000;

/// 3-D scatter. Points are `(x, y, z)`, colored by z, drawn back to front. More than
/// [`SCATTER3D_CAP`] points are strided so orbit stays interactive.
pub fn raster_scatter3d(
    width: u32,
    height: u32,
    points: &[[f64; 3]],
    radius: f32,
    style: PlotStyle,
    colormap: Colormap,
    view: MeshView,
) -> CpuFrame {
    if width == 0 || height == 0 {
        return CpuFrame { width: 0, height: 0, format: PixelFormat::Rgba8, data: Vec::new() };
    }
    let Some(area) = PlotArea::new(width, height, &style) else {
        return empty_frame(width, height, &style);
    };
    let finite: Vec<[f64; 3]> = points.iter().copied().filter(|p| p.iter().all(|v| v.is_finite())).collect();
    if finite.is_empty() {
        return raster_projected(width, height, &[], &[], style, colormap, view);
    }
    let stride = (finite.len() / SCATTER3D_CAP).max(1);
    let drawn: Vec<[f64; 3]> = finite.iter().step_by(stride).copied().collect();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for v in &drawn {
        for k in 0..3 {
            min[k] = min[k].min(v[k]);
            max[k] = max[k].max(v[k]);
        }
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
    let extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]).max(1e-9);
    let zmin = min[2];
    let zspan = (max[2] - min[2]).max(1e-12);
    let zoom = if view.zoom.is_finite() && view.zoom > 0.0 { view.zoom } else { 1.0 };
    let scale = (area.pw.min(area.ph) as f64) * 0.78 * zoom;
    let ox = area.x0 as f64 + area.pw as f64 * 0.5;
    let oy = area.y0 as f64 + area.ph as f64 * 0.5;
    let mut projected = Vec::with_capacity(drawn.len());
    for v in drawn {
        let n = [(v[0] - center[0]) / extent, (v[1] - center[1]) / extent, (v[2] - center[2]) / extent];
        let (x, y, z) = rotate_view(n[0], n[1], n[2], view);
        let t = ((v[2] - zmin) / zspan).clamp(0.0, 1.0);
        let rgb = colormap.sample(t);
        projected.push(((ox + x * scale) as f32, (oy - y * scale) as f32, z as f32, [rgb[0], rgb[1], rgb[2], 255]));
    }
    projected.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
    fill(&mut data, width, style.background);
    let radius = if radius.is_finite() && radius > 0.0 { radius } else { 3.0 };
    for (x, y, _, color) in projected {
        draw_disc(&mut data, width, height, x, y, radius, color);
    }
    let x0 = area.x0 as i32;
    let y0 = area.y0 as i32;
    let x1 = (area.x0 + area.pw - 1) as i32;
    let y1 = (area.y0 + area.ph - 1) as i32;
    fill_rect(&mut data, width, height, x0, y0, x1, y0, style.axis);
    fill_rect(&mut data, width, height, x0, y1, x1, y1, style.axis);
    fill_rect(&mut data, width, height, x0, y0, x0, y1, style.axis);
    fill_rect(&mut data, width, height, x1, y0, x1, y1, style.axis);
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

    #[test]
    fn histogram_bins_and_bars_paint() {
        let (counts, edges) = bin_counts(&[0.0, 0.0, 0.0, 1.0, 1.0], 2, Some((0.0, 1.0)));
        assert_eq!(edges.len(), 3);
        assert_eq!(counts, vec![3.0, 2.0]);
        let frame = raster_bars(
            120,
            80,
            &[0.0, 1.0],
            &[0.8, 1.8],
            &[1.0, 4.0],
            None,
            None,
            PlotStyle::default(),
            [220, 40, 40, 255],
        );
        let reds = frame.data.chunks_exact(4).filter(|px| px[0] == 220 && px[1] == 40).count();
        assert!(reds > 20, "bars should paint, got {reds}");
    }

    #[test]
    fn contour_lines_follow_a_ramp() {
        let mut values = vec![0.0; 16];
        for r in 0..4 {
            for c in 0..4 {
                values[r * 4 + c] = c as f64;
            }
        }
        let frame = raster_contour(
            100,
            80,
            &values,
            4,
            4,
            &[0.5, 1.5, 2.5],
            None,
            PlotStyle::default(),
            Colormap::Viridis,
            false,
            None,
        );
        let colored = frame.data.chunks_exact(4).filter(|px| px[0] != 26 || px[1] != 28).count();
        assert!(colored > 30, "isolines should paint, got {colored}");
    }

    #[test]
    fn surface_and_mesh_change_with_yaw() {
        let z: Vec<f64> = (0..64).map(|i| ((i % 8) as f64 / 7.0).sin()).collect();
        let xs: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let a = raster_surface(80, 60, &z, 8, 8, None, None, PlotStyle::default(), Colormap::Magma, MeshView::default());
        let b = raster_surface(
            80,
            60,
            &z,
            8,
            8,
            Some(&xs),
            Some(&ys),
            PlotStyle::default(),
            Colormap::Magma,
            MeshView { yaw: 1.4, ..MeshView::default() },
        );
        assert_ne!(a.data, b.data);
        let verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [0.2, 1.0, 0.8], [1.0, 1.0, 0.1]];
        let faces = [[0, 1, 2], [1, 3, 2]];
        let mesh = raster_mesh(80, 60, &verts, &faces, PlotStyle::default(), Colormap::Viridis, MeshView::default());
        let painted = mesh.data.chunks_exact(4).filter(|px| px[0] != 26 || px[1] != 28 || px[2] != 33).count();
        assert!(painted > 20, "mesh should cover pixels, got {painted}");
        let cloud = [[0.0, 0.0, 0.0], [1.0, 0.2, 0.5], [0.2, 1.0, 1.0], [-0.4, 0.3, 0.8]];
        let scatter = raster_scatter3d(80, 60, &cloud, 4.0, PlotStyle::default(), Colormap::Viridis, MeshView::default());
        let dots = scatter.data.chunks_exact(4).filter(|px| px[0] != 26 || px[1] != 28 || px[2] != 33).count();
        assert!(dots > 20, "3d scatter should paint discs, got {dots}");
    }

    #[test]
    fn z_is_the_up_axis() {
        let view = MeshView { yaw: 0.0, pitch: 0.6, zoom: 1.0 };
        let (_, up_z, _) = rotate_view(0.0, 0.0, 1.0, view);
        let (_, up_y, _) = rotate_view(0.0, 1.0, 0.0, view);
        assert!(up_z > up_y, "z should project above y, got z {up_z} y {up_y}");
        let (right, _, _) = rotate_view(1.0, 0.0, 0.0, view);
        assert!(right > 0.5, "x should stay to the right, got {right}");
    }
}
