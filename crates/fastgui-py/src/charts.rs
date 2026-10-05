//! Histogram, bar, contour, surface, and mesh plots. Same CPU-raster image layer as the other plots.

use std::sync::{Arc, Mutex};

use fastgui_core::plot::{
    axis_fraction, bin_counts, pan_range, pan_window, raster_bars, raster_contour, raster_mesh, raster_scatter3d,
    raster_surface, zoom_range, zoom_window, Colormap, MeshView, PlotStyle,
};
use fastgui_core::widget::{LayerFit, PointerCallback};
use fastgui_core::CpuFrame;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

use crate::backend::CommandDispatch;
use crate::next_viewport_id;
use crate::plots::{
    contained_pointer, describe_image, float_grid, float_series, pixel_size, style_from, submit_frame, IdCell,
    SenderCell,
};
use crate::widgets::DescribedWidget;

struct Shell {
    image_id: u64,
    frames: fastgui_core::FrameSlot<CpuFrame>,
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Arc<Mutex<PlotStyle>>,
    interactive: bool,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

impl Shell {
    fn new(
        background: Option<(f32, f32, f32, f32)>,
        color: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        Ok(Self {
            image_id: next_viewport_id(),
            frames: fastgui_core::FrameSlot::new(),
            dispatch: Arc::new(Mutex::new(None)),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Arc::new(Mutex::new(style_from(background, color))),
            interactive,
            width,
            height,
            flex_grow,
        })
    }

    fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    fn describe(&self, on_pointer: Option<PointerCallback>) -> DescribedWidget {
        describe_image(
            self.image_id,
            &self.frames,
            &self.id,
            &self.sender,
            self.flex_grow,
            self.width,
            self.height,
            LayerFit::Contain,
            on_pointer,
        )
    }
}

fn colormap_of(name: &str) -> PyResult<Colormap> {
    Colormap::parse(name).ok_or_else(|| PyValueError::new_err("colormap must be 'viridis', 'magma', or 'gray'"))
}

fn xy_zoom_pan(
    action: u8,
    dx: f32,
    dy: f32,
    lx: f32,
    ly: f32,
    w: f32,
    h: f32,
    pw: u32,
    ph: u32,
    style: PlotStyle,
    xr: (f64, f64),
    yr: (f64, f64),
) -> Option<((f64, f64), (f64, f64))> {
    match action {
        0 => {
            let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
            let (fx, fy, _, _) = contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
            let tx = axis_fraction(fx, f64::from(pw), f64::from(style.margin_left), f64::from(style.margin_right), false);
            let ty = axis_fraction(fy, f64::from(ph), f64::from(style.margin_top), f64::from(style.margin_bottom), true);
            Some((zoom_range(xr, tx, factor), zoom_range(yr, ty, factor)))
        }
        1 => {
            let (_, _, ddx, ddy) = contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
            Some((pan_range(xr, -ddx), pan_range(yr, ddy)))
        }
        _ => None,
    }
}

/// Histogram of a 1-D sample. `bin_range` is the domain the bars cover; values outside it are dropped.
#[pyclass]
pub(crate) struct PlotHistogram {
    shell: Shell,
    samples: Arc<Mutex<Vec<f64>>>,
    bins: Arc<Mutex<usize>>,
    bin_range: Arc<Mutex<Option<(f64, f64)>>>,
    x_view: Arc<Mutex<Option<(f64, f64)>>>,
    y_view: Arc<Mutex<Option<(f64, f64)>>>,
}

#[pymethods]
impl PlotHistogram {
    #[new]
    #[pyo3(signature = (
        values=None,
        *,
        bins=20,
        bin_range=None,
        color=None,
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        values: Option<&Bound<'_, PyAny>>,
        bins: usize,
        bin_range: Option<(f64, f64)>,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        if !(1..=2048).contains(&bins) {
            return Err(PyValueError::new_err("bins must be from 1 to 2048"));
        }
        let plot = Self {
            shell: Shell::new(background, color, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            samples: Arc::new(Mutex::new(Vec::new())),
            bins: Arc::new(Mutex::new(bins)),
            bin_range: Arc::new(Mutex::new(parse_pair(bin_range)?)),
            x_view: Arc::new(Mutex::new(None)),
            y_view: Arc::new(Mutex::new(None)),
        };
        if let Some(values) = values {
            plot.set_data(values)?;
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    /// Replace the samples. Keeps the current pan/zoom.
    fn set_data(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let samples = float_series(values)?;
        if samples.len() > 5_000_000 {
            return Err(PyValueError::new_err("histogram is limited to 5_000_000 samples"));
        }
        *self.samples.lock().unwrap_or_else(|p| p.into_inner()) = samples;
        self.redraw();
        Ok(())
    }

    fn set_bins(&self, bins: usize) -> PyResult<()> {
        if !(1..=2048).contains(&bins) {
            return Err(PyValueError::new_err("bins must be from 1 to 2048"));
        }
        *self.bins.lock().unwrap_or_else(|p| p.into_inner()) = bins;
        self.redraw();
        Ok(())
    }

    fn counts(&self) -> Vec<f64> {
        let (counts, _) = self.binned();
        counts
    }

    fn edges(&self) -> Vec<f64> {
        let (_, edges) = self.binned();
        edges
    }

    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let (xr, yr) = self.current_view();
        let tx = axis_fraction(fx, f64::from(self.shell.pixel_width), f64::from(style.margin_left), f64::from(style.margin_right), false);
        let ty = axis_fraction(fy, f64::from(self.shell.pixel_height), f64::from(style.margin_top), f64::from(style.margin_bottom), true);
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range(xr, tx, factor));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range(yr, ty, factor));
        self.redraw();
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let (xr, yr) = self.current_view();
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range(xr, -dx));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range(yr, dy));
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.redraw();
        Ok(())
    }

    fn view_range(&self) -> ((f64, f64), (f64, f64)) {
        self.current_view()
    }
}

impl PlotHistogram {
    fn binned(&self) -> (Vec<f64>, Vec<f64>) {
        let samples = self.samples.lock().unwrap_or_else(|p| p.into_inner());
        let bins = *self.bins.lock().unwrap_or_else(|p| p.into_inner());
        let range = *self.bin_range.lock().unwrap_or_else(|p| p.into_inner());
        bin_counts(&samples, bins, range)
    }

    fn current_view(&self) -> ((f64, f64), (f64, f64)) {
        let (counts, edges) = self.binned();
        let mut xs = Vec::new();
        let mut ys = vec![0.0];
        if edges.len() >= 2 {
            xs.push(edges[0]);
            xs.push(*edges.last().unwrap());
        }
        ys.extend(counts.iter().copied());
        let xv = *self.x_view.lock().unwrap_or_else(|p| p.into_inner());
        let yv = *self.y_view.lock().unwrap_or_else(|p| p.into_inner());
        (
            fastgui_core::plot::resolve_for_view(xv, &xs),
            fastgui_core::plot::resolve_for_view(yv, &ys),
        )
    }

    fn redraw(&self) {
        let (counts, edges) = self.binned();
        let (xr, yr) = self.current_view();
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let n = counts.len();
        let (x0, x1): (Vec<f64>, Vec<f64>) = if edges.len() == n + 1 {
            ((0..n).map(|i| edges[i]).collect(), (0..n).map(|i| edges[i + 1]).collect())
        } else {
            (Vec::new(), Vec::new())
        };
        let frame = raster_bars(
            self.shell.pixel_width,
            self.shell.pixel_height,
            &x0,
            &x1,
            &counts,
            Some(xr),
            Some(yr),
            style,
            style.series,
        );
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let samples = self.samples.clone();
        let bins = self.bins.clone();
        let bin_range = self.bin_range.clone();
        let x_view = self.x_view.clone();
        let y_view = self.y_view.clone();
        let style = self.shell.style.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let samples = samples.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let bins = *bins.lock().unwrap_or_else(|p| p.into_inner());
            let range = *bin_range.lock().unwrap_or_else(|p| p.into_inner());
            let (counts, edges) = bin_counts(&samples, bins, range);
            let n = counts.len();
            let mut xs = Vec::new();
            if edges.len() >= 2 {
                xs.push(edges[0]);
                xs.push(*edges.last().unwrap());
            }
            let mut ys = vec![0.0];
            ys.extend(counts.iter().copied());
            let mut xr = fastgui_core::plot::resolve_for_view(*x_view.lock().unwrap_or_else(|p| p.into_inner()), &xs);
            let mut yr = fastgui_core::plot::resolve_for_view(*y_view.lock().unwrap_or_else(|p| p.into_inner()), &ys);
            if action == 2 {
                *x_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
                *y_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
                xr = fastgui_core::plot::resolve_for_view(None, &xs);
                yr = fastgui_core::plot::resolve_for_view(None, &ys);
            } else if let Some((nx, ny)) = xy_zoom_pan(action, dx, dy, lx, ly, w, h, pw, ph, style_now, xr, yr) {
                xr = nx;
                yr = ny;
                *x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(xr);
                *y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(yr);
            } else {
                return;
            }
            let (x0, x1): (Vec<f64>, Vec<f64>) = if edges.len() == n + 1 {
                ((0..n).map(|i| edges[i]).collect(), (0..n).map(|i| edges[i + 1]).collect())
            } else {
                (Vec::new(), Vec::new())
            };
            let frame = raster_bars(pw, ph, &x0, &x1, &counts, Some(xr), Some(yr), style_now, style_now.series);
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

/// Vertical bars. `heights[i]` is drawn at category `i`, from a baseline of 0.
#[pyclass]
pub(crate) struct PlotBar {
    shell: Shell,
    heights: Arc<Mutex<Vec<f64>>>,
    x_view: Arc<Mutex<Option<(f64, f64)>>>,
    y_view: Arc<Mutex<Option<(f64, f64)>>>,
}

#[pymethods]
impl PlotBar {
    #[new]
    #[pyo3(signature = (
        heights=None,
        *,
        color=None,
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        heights: Option<&Bound<'_, PyAny>>,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        let plot = Self {
            shell: Shell::new(background, color, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            heights: Arc::new(Mutex::new(Vec::new())),
            x_view: Arc::new(Mutex::new(None)),
            y_view: Arc::new(Mutex::new(None)),
        };
        if let Some(heights) = heights {
            plot.set_data(heights)?;
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    fn set_data(&self, heights: &Bound<'_, PyAny>) -> PyResult<()> {
        let heights = float_series(heights)?;
        if heights.len() > 100_000 {
            return Err(PyValueError::new_err("bar chart is limited to 100_000 bars"));
        }
        *self.heights.lock().unwrap_or_else(|p| p.into_inner()) = heights;
        self.redraw();
        Ok(())
    }

    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let (xr, yr) = self.current_view();
        let tx = axis_fraction(fx, f64::from(self.shell.pixel_width), f64::from(style.margin_left), f64::from(style.margin_right), false);
        let ty = axis_fraction(fy, f64::from(self.shell.pixel_height), f64::from(style.margin_top), f64::from(style.margin_bottom), true);
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range(xr, tx, factor));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range(yr, ty, factor));
        self.redraw();
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let (xr, yr) = self.current_view();
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range(xr, -dx));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range(yr, dy));
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.redraw();
        Ok(())
    }

    fn view_range(&self) -> ((f64, f64), (f64, f64)) {
        self.current_view()
    }
}

impl PlotBar {
    fn spans(heights: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = heights.len();
        let x0: Vec<f64> = (0..n).map(|i| i as f64 - 0.4).collect();
        let x1: Vec<f64> = (0..n).map(|i| i as f64 + 0.4).collect();
        (x0, x1, heights.to_vec())
    }

    fn current_view(&self) -> ((f64, f64), (f64, f64)) {
        let heights = self.heights.lock().unwrap_or_else(|p| p.into_inner());
        let n = heights.len();
        let xs: Vec<f64> = if n == 0 { Vec::new() } else { vec![-0.6, n as f64 - 0.4] };
        let mut ys = vec![0.0];
        ys.extend(heights.iter().copied());
        let xv = *self.x_view.lock().unwrap_or_else(|p| p.into_inner());
        let yv = *self.y_view.lock().unwrap_or_else(|p| p.into_inner());
        (
            fastgui_core::plot::resolve_for_view(xv, &xs),
            fastgui_core::plot::resolve_for_view(yv, &ys),
        )
    }

    fn redraw(&self) {
        let heights = self.heights.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let (xr, yr) = self.current_view();
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let (x0, x1, h) = Self::spans(&heights);
        let frame = raster_bars(self.shell.pixel_width, self.shell.pixel_height, &x0, &x1, &h, Some(xr), Some(yr), style, style.series);
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let heights = self.heights.clone();
        let x_view = self.x_view.clone();
        let y_view = self.y_view.clone();
        let style = self.shell.style.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let heights = heights.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let n = heights.len();
            let xs: Vec<f64> = if n == 0 { Vec::new() } else { vec![-0.6, n as f64 - 0.4] };
            let mut ys = vec![0.0];
            ys.extend(heights.iter().copied());
            let mut xr = fastgui_core::plot::resolve_for_view(*x_view.lock().unwrap_or_else(|p| p.into_inner()), &xs);
            let mut yr = fastgui_core::plot::resolve_for_view(*y_view.lock().unwrap_or_else(|p| p.into_inner()), &ys);
            if action == 2 {
                *x_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
                *y_view.lock().unwrap_or_else(|p| p.into_inner()) = None;
                xr = fastgui_core::plot::resolve_for_view(None, &xs);
                yr = fastgui_core::plot::resolve_for_view(None, &ys);
            } else if let Some((nx, ny)) = xy_zoom_pan(action, dx, dy, lx, ly, w, h, pw, ph, style_now, xr, yr) {
                xr = nx;
                yr = ny;
                *x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(xr);
                *y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(yr);
            } else {
                return;
            }
            let (x0, x1, h) = PlotBar::spans(&heights);
            let frame = raster_bars(pw, ph, &x0, &x1, &h, Some(xr), Some(yr), style_now, style_now.series);
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

enum LevelSpec {
    Count(usize),
    Values(Vec<f64>),
}

fn parse_levels(obj: Option<&Bound<'_, PyAny>>) -> PyResult<LevelSpec> {
    let Some(obj) = obj else { return Ok(LevelSpec::Count(8)) };
    if let Ok(n) = obj.extract::<usize>() {
        if !(1..=64).contains(&n) {
            return Err(PyValueError::new_err("level count must be from 1 to 64"));
        }
        return Ok(LevelSpec::Count(n));
    }
    let values = float_series(obj)?;
    if values.is_empty() || values.len() > 64 || values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("levels must be 1 to 64 finite values, or a count"));
    }
    Ok(LevelSpec::Values(values))
}

/// Isolines of a 2-D grid. `filled` paints colormap bands under the lines (a banded heatmap).
#[pyclass]
pub(crate) struct PlotContour {
    shell: Shell,
    colormap: Colormap,
    filled: bool,
    levels: Arc<Mutex<LevelSpec>>,
    v_range: Arc<Mutex<Option<(f64, f64)>>>,
    grid: Arc<Mutex<(Vec<f64>, usize, usize)>>,
    window: Arc<Mutex<(f64, f64, f64, f64)>>,
}

#[pymethods]
impl PlotContour {
    #[new]
    #[pyo3(signature = (
        values=None,
        *,
        levels=None,
        filled=false,
        colormap="viridis",
        background=None,
        v_range=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        values: Option<&Bound<'_, PyAny>>,
        levels: Option<&Bound<'_, PyAny>>,
        filled: bool,
        colormap: &str,
        background: Option<(f32, f32, f32, f32)>,
        v_range: Option<(f64, f64)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        let plot = Self {
            shell: Shell::new(background, None, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            colormap: colormap_of(colormap)?,
            filled,
            levels: Arc::new(Mutex::new(parse_levels(levels)?)),
            v_range: Arc::new(Mutex::new(parse_pair(v_range)?)),
            grid: Arc::new(Mutex::new((Vec::new(), 0, 0))),
            window: Arc::new(Mutex::new((0.0, 1.0, 0.0, 1.0))),
        };
        if let Some(values) = values {
            plot.set_data(values)?;
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    fn set_data(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (grid, rows, cols) = float_grid(values)?;
        if rows < 2 || cols < 2 {
            return Err(PyValueError::new_err("contour grid must be at least 2x2"));
        }
        if rows.saturating_mul(cols) > 2_000_000 {
            return Err(PyValueError::new_err("contour grid is limited to 2_000_000 values"));
        }
        *self.grid.lock().unwrap_or_else(|p| p.into_inner()) = (grid, rows, cols);
        self.redraw();
        Ok(())
    }

    /// `levels` is a count (evenly spaced) or a sequence of data values.
    fn set_levels(&self, levels: &Bound<'_, PyAny>) -> PyResult<()> {
        *self.levels.lock().unwrap_or_else(|p| p.into_inner()) = parse_levels(Some(levels))?;
        self.redraw();
        Ok(())
    }

    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        self.apply_zoom(factor, fx, fy);
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        // Window row 0 is the top, so a downward drag (+dy) moves the window up: content follows the cursor.
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = pan_window(window, -dx, -dy);
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = (0.0, 1.0, 0.0, 1.0);
        self.redraw();
        Ok(())
    }

    fn window(&self) -> (f64, f64, f64, f64) {
        *self.window.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl PlotContour {
    fn level_values(&self, grid: &[f64]) -> Vec<f64> {
        let spec = self.levels.lock().unwrap_or_else(|p| p.into_inner());
        match &*spec {
            LevelSpec::Values(v) => v.clone(),
            LevelSpec::Count(n) => {
                let explicit = *self.v_range.lock().unwrap_or_else(|p| p.into_inner());
                let (lo, hi) = fastgui_core::plot::resolve_for_view(explicit, grid);
                let n = *n;
                (1..=n).map(|i| lo + (hi - lo) * (i as f64) / (n as f64 + 1.0)).collect()
            }
        }
    }

    fn redraw(&self) {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let (grid, rows, cols) = self.grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let levels = self.level_values(&grid);
        let vr = *self.v_range.lock().unwrap_or_else(|p| p.into_inner());
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_contour(
            self.shell.pixel_width,
            self.shell.pixel_height,
            &grid,
            rows,
            cols,
            &levels,
            vr,
            style,
            self.colormap,
            self.filled,
            Some(window),
        );
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn apply_zoom(&self, factor: f64, fx: f64, fy: f64) {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let tx = axis_fraction(fx, f64::from(self.shell.pixel_width), f64::from(style.margin_left), f64::from(style.margin_right), false);
        let ty = axis_fraction(fy, f64::from(self.shell.pixel_height), f64::from(style.margin_top), f64::from(style.margin_bottom), false);
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = zoom_window(window, tx, ty, factor);
        self.redraw();
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let style = self.shell.style.clone();
        let window = self.window.clone();
        let grid = self.grid.clone();
        let levels = self.levels.clone();
        let v_range = self.v_range.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let colormap = self.colormap;
        let filled = self.filled;
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let mut view = *window.lock().unwrap_or_else(|p| p.into_inner());
            match action {
                2 => view = (0.0, 1.0, 0.0, 1.0),
                0 => {
                    let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
                    let (fx, fy, _, _) = contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
                    let tx = axis_fraction(fx, f64::from(pw), f64::from(style_now.margin_left), f64::from(style_now.margin_right), false);
                    let ty = axis_fraction(fy, f64::from(ph), f64::from(style_now.margin_top), f64::from(style_now.margin_bottom), false);
                    view = zoom_window(view, tx, ty, factor);
                }
                1 => {
                    let (_, _, ddx, ddy) = contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
                    // Window row 0 is the top, so a downward drag (+dy) moves the window up: content follows the cursor.
                    view = pan_window(view, -ddx, -ddy);
                }
                _ => return,
            }
            *window.lock().unwrap_or_else(|p| p.into_inner()) = view;
            let (values, rows, cols) = grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let spec = levels.lock().unwrap_or_else(|p| p.into_inner());
            let vr = *v_range.lock().unwrap_or_else(|p| p.into_inner());
            let lv = match &*spec {
                LevelSpec::Values(v) => v.clone(),
                LevelSpec::Count(n) => {
                    let (lo, hi) = fastgui_core::plot::resolve_for_view(vr, &values);
                    let n = *n;
                    (1..=n).map(|i| lo + (hi - lo) * (i as f64) / (n as f64 + 1.0)).collect()
                }
            };
            let frame = raster_contour(pw, ph, &values, rows, cols, &lv, vr, style_now, colormap, filled, Some(view));
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

fn orbit_pointer(
    view: &Arc<Mutex<MeshView>>,
    action: u8,
    dx: f32,
    dy: f32,
    lx: f32,
    ly: f32,
    w: f32,
    h: f32,
    pw: u32,
    ph: u32,
) -> bool {
    let mut cam = *view.lock().unwrap_or_else(|p| p.into_inner());
    match action {
        2 => cam = MeshView::default(),
        0 => {
            let factor = if dy < 0.0 { 1.12 } else { 1.0 / 1.12 };
            cam.zoom = (cam.zoom * factor).clamp(0.25, 8.0);
        }
        1 => {
            let (_, _, ddx, ddy) = contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
            cam.yaw += ddx * 4.0;
            cam.pitch = (cam.pitch + ddy * 3.0).clamp(-1.2, 1.2);
        }
        _ => return false,
    }
    *view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
    true
}

/// 3-D height field. Drag orbits, the wheel zooms, double-click resets. Grids above 96² are strided.
#[pyclass]
pub(crate) struct PlotSurface {
    shell: Shell,
    colormap: Colormap,
    grid: Arc<Mutex<(Vec<f64>, usize, usize)>>,
    /// `None` means column/row index. Otherwise a 1-D axis or a grid the same shape as `z`.
    x: Arc<Mutex<Option<Vec<f64>>>>,
    y: Arc<Mutex<Option<Vec<f64>>>>,
    view: Arc<Mutex<MeshView>>,
}

#[pymethods]
impl PlotSurface {
    #[new]
    #[pyo3(signature = (
        z=None,
        *,
        x=None,
        y=None,
        colormap="viridis",
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        z: Option<&Bound<'_, PyAny>>,
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        let plot = Self {
            shell: Shell::new(background, None, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            colormap: colormap_of(colormap)?,
            grid: Arc::new(Mutex::new((Vec::new(), 0, 0))),
            x: Arc::new(Mutex::new(None)),
            y: Arc::new(Mutex::new(None)),
            view: Arc::new(Mutex::new(MeshView::default())),
        };
        if let Some(z) = z {
            plot.set_data(z, x, y)?;
        } else if x.is_some() || y.is_some() {
            return Err(PyValueError::new_err("surface x and y need a z grid"));
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    /// Replace the height field. `x` and `y` are optional axes (length `cols` / `rows`) or grids of the same shape.
    #[pyo3(signature = (z, x=None, y=None))]
    fn set_data(&self, z: &Bound<'_, PyAny>, x: Option<&Bound<'_, PyAny>>, y: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let (grid, rows, cols) = float_grid(z)?;
        if rows < 2 || cols < 2 {
            return Err(PyValueError::new_err("surface must be at least 2x2"));
        }
        if rows.saturating_mul(cols) > 2_000_000 {
            return Err(PyValueError::new_err("surface is limited to 2_000_000 heights"));
        }
        let xs = match x {
            Some(x) => Some(surface_axis(x, rows, cols, true)?),
            None => None,
        };
        let ys = match y {
            Some(y) => Some(surface_axis(y, rows, cols, false)?),
            None => None,
        };
        *self.grid.lock().unwrap_or_else(|p| p.into_inner()) = (grid, rows, cols);
        *self.x.lock().unwrap_or_else(|p| p.into_inner()) = xs;
        *self.y.lock().unwrap_or_else(|p| p.into_inner()) = ys;
        self.redraw();
        Ok(())
    }

    /// `factor` > 1 zooms in. `(fx, fy)` is ignored; the camera stays centered.
    fn zoom(&self, factor: f64, _fx: f64, _fy: f64) -> PyResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PyValueError::new_err("zoom factor must be positive"));
        }
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.zoom = (cam.zoom * factor).clamp(0.25, 8.0);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    /// Orbit: `dx` changes yaw, `dy` changes pitch (fractions of the widget, drag-down positive).
    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.yaw += dx * 4.0;
        cam.pitch = (cam.pitch + dy * 3.0).clamp(-1.2, 1.2);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = MeshView::default();
        self.redraw();
        Ok(())
    }

    /// `(yaw, pitch, zoom)` in radians and a scale.
    fn view(&self) -> (f64, f64, f64) {
        let cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        (cam.yaw, cam.pitch, cam.zoom)
    }
}

impl PlotSurface {
    fn redraw(&self) {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let (grid, rows, cols) = self.grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let xs = self.x.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let ys = self.y.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_surface(
            self.shell.pixel_width,
            self.shell.pixel_height,
            &grid,
            rows,
            cols,
            xs.as_deref(),
            ys.as_deref(),
            style,
            self.colormap,
            view,
        );
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let view = self.view.clone();
        let grid = self.grid.clone();
        let x = self.x.clone();
        let y = self.y.clone();
        let style = self.shell.style.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let colormap = self.colormap;
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            if !orbit_pointer(&view, action, dx, dy, lx, ly, w, h, pw, ph) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let cam = *view.lock().unwrap_or_else(|p| p.into_inner());
            let (values, rows, cols) = grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let xs = x.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let ys = y.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let frame = raster_surface(pw, ph, &values, rows, cols, xs.as_deref(), ys.as_deref(), style_now, colormap, cam);
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

fn surface_axis(obj: &Bound<'_, PyAny>, rows: usize, cols: usize, along_cols: bool) -> PyResult<Vec<f64>> {
    let values = if let Ok((grid, r, c)) = float_grid(obj) {
        if r == rows && c == cols {
            return Ok(grid);
        }
        if r == 1 {
            grid
        } else if c == 1 && r == if along_cols { cols } else { rows } {
            grid
        } else {
            return Err(PyValueError::new_err(if along_cols {
                "surface x must have length `cols` or the same shape as z"
            } else {
                "surface y must have length `rows` or the same shape as z"
            }));
        }
    } else {
        float_series(obj)?
    };
    let expect = if along_cols { cols } else { rows };
    if values.len() != expect && values.len() != rows.saturating_mul(cols) {
        return Err(PyValueError::new_err(if along_cols {
            "surface x must have length `cols` or the same shape as z"
        } else {
            "surface y must have length `rows` or the same shape as z"
        }));
    }
    Ok(values)
}

fn parse_xyz(
    x: &Bound<'_, PyAny>,
    y: Option<&Bound<'_, PyAny>>,
    z: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<[f64; 3]>> {
    match (y, z) {
        (None, None) => {
            let (values, rows, cols) = float_grid(x)?;
            if cols != 3 {
                return Err(PyValueError::new_err("points must be an (N, 3) array, or pass x, y, and z"));
            }
            if rows > 200_000 {
                return Err(PyValueError::new_err("3d scatter is limited to 200_000 points"));
            }
            let mut points = Vec::with_capacity(rows);
            for i in 0..rows {
                points.push([values[i * 3], values[i * 3 + 1], values[i * 3 + 2]]);
            }
            Ok(points)
        }
        (Some(y), Some(z)) => {
            let xs = float_series(x)?;
            let ys = float_series(y)?;
            let zs = float_series(z)?;
            if xs.len() != ys.len() || ys.len() != zs.len() {
                return Err(PyValueError::new_err("x, y, and z must have the same length"));
            }
            if xs.len() > 200_000 {
                return Err(PyValueError::new_err("3d scatter is limited to 200_000 points"));
            }
            Ok(xs.into_iter().zip(ys).zip(zs).map(|((x, y), z)| [x, y, z]).collect())
        }
        _ => Err(PyValueError::new_err("pass (N, 3) points, or x, y, and z")),
    }
}

/// 3-D scatter of `(x, y, z)` or one `(N, 3)` array. Drag orbits, the wheel zooms.
#[pyclass]
pub(crate) struct PlotScatter3D {
    shell: Shell,
    colormap: Colormap,
    radius: f32,
    points: Arc<Mutex<Vec<[f64; 3]>>>,
    view: Arc<Mutex<MeshView>>,
}

#[pymethods]
impl PlotScatter3D {
    #[new]
    #[pyo3(signature = (
        x=None,
        y=None,
        z=None,
        *,
        point_radius=3.5,
        colormap="viridis",
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        z: Option<&Bound<'_, PyAny>>,
        point_radius: f32,
        colormap: &str,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        if !point_radius.is_finite() || point_radius <= 0.0 {
            return Err(PyValueError::new_err("point_radius must be positive"));
        }
        let plot = Self {
            shell: Shell::new(background, None, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            colormap: colormap_of(colormap)?,
            radius: point_radius,
            points: Arc::new(Mutex::new(Vec::new())),
            view: Arc::new(Mutex::new(MeshView::default())),
        };
        if let Some(x) = x {
            plot.set_data(x, y, z)?;
        } else if y.is_some() || z.is_some() {
            return Err(PyValueError::new_err("pass (N, 3) points, or x, y, and z"));
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    /// Replace the points. One `(N, 3)` array, or three 1-D arrays. Keeps the camera.
    #[pyo3(signature = (x, y=None, z=None))]
    fn set_data(&self, x: &Bound<'_, PyAny>, y: Option<&Bound<'_, PyAny>>, z: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        *self.points.lock().unwrap_or_else(|p| p.into_inner()) = parse_xyz(x, y, z)?;
        self.redraw();
        Ok(())
    }

    fn zoom(&self, factor: f64, _fx: f64, _fy: f64) -> PyResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PyValueError::new_err("zoom factor must be positive"));
        }
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.zoom = (cam.zoom * factor).clamp(0.25, 8.0);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.yaw += dx * 4.0;
        cam.pitch = (cam.pitch + dy * 3.0).clamp(-1.2, 1.2);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = MeshView::default();
        self.redraw();
        Ok(())
    }

    fn view(&self) -> (f64, f64, f64) {
        let cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        (cam.yaw, cam.pitch, cam.zoom)
    }
}

impl PlotScatter3D {
    fn redraw(&self) {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let points = self.points.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_scatter3d(
            self.shell.pixel_width,
            self.shell.pixel_height,
            &points,
            self.radius,
            style,
            self.colormap,
            view,
        );
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let view = self.view.clone();
        let points = self.points.clone();
        let radius = self.radius;
        let style = self.shell.style.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let colormap = self.colormap;
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            if !orbit_pointer(&view, action, dx, dy, lx, ly, w, h, pw, ph) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let cam = *view.lock().unwrap_or_else(|p| p.into_inner());
            let points = points.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let frame = raster_scatter3d(pw, ph, &points, radius, style_now, colormap, cam);
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

fn parse_faces(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[u32; 3]>> {
    let seq = obj.cast::<PySequence>().map_err(|_| PyTypeError::new_err("faces must be a sequence of (i, j, k)"))?;
    let n = seq.len()? as usize;
    if n > 20_000 {
        return Err(PyValueError::new_err("mesh is limited to 20_000 triangles"));
    }
    let mut faces = Vec::with_capacity(n);
    for i in 0..n {
        let item = seq.get_item(i)?;
        let parts = item.cast::<PySequence>().map_err(|_| PyTypeError::new_err("each face is (i, j, k)"))?;
        if parts.len()? != 3 {
            return Err(PyTypeError::new_err("each face is (i, j, k)"));
        }
        let a: usize = parts.get_item(0)?.extract()?;
        let b: usize = parts.get_item(1)?.extract()?;
        let c: usize = parts.get_item(2)?.extract()?;
        if a > u32::MAX as usize || b > u32::MAX as usize || c > u32::MAX as usize {
            return Err(PyValueError::new_err("face index does not fit in u32"));
        }
        faces.push([a as u32, b as u32, c as u32]);
    }
    Ok(faces)
}

fn parse_vertices(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f64; 3]>> {
    let (values, rows, cols) = float_grid(obj)?;
    if cols != 3 {
        return Err(PyValueError::new_err("vertices must be an (N, 3) array"));
    }
    if rows > 20_000 {
        return Err(PyValueError::new_err("mesh is limited to 20_000 vertices"));
    }
    let mut verts = Vec::with_capacity(rows);
    for i in 0..rows {
        verts.push([values[i * 3], values[i * 3 + 1], values[i * 3 + 2]]);
    }
    Ok(verts)
}

/// Triangle mesh. `vertices` is `(N, 3)`, `faces` is `(M, 3)` indexes. Drag orbits, wheel zooms.
#[pyclass]
pub(crate) struct PlotMesh {
    shell: Shell,
    colormap: Colormap,
    vertices: Arc<Mutex<Vec<[f64; 3]>>>,
    faces: Arc<Mutex<Vec<[u32; 3]>>>,
    view: Arc<Mutex<MeshView>>,
}

#[pymethods]
impl PlotMesh {
    #[new]
    #[pyo3(signature = (
        vertices=None,
        faces=None,
        *,
        colormap="viridis",
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
        interactive=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vertices: Option<&Bound<'_, PyAny>>,
        faces: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
        interactive: bool,
    ) -> PyResult<Self> {
        let plot = Self {
            shell: Shell::new(background, None, pixel_width, pixel_height, width, height, flex_grow, interactive)?,
            colormap: colormap_of(colormap)?,
            vertices: Arc::new(Mutex::new(Vec::new())),
            faces: Arc::new(Mutex::new(Vec::new())),
            view: Arc::new(Mutex::new(MeshView::default())),
        };
        match (vertices, faces) {
            (Some(vertices), Some(faces)) => plot.set_data(vertices, faces)?,
            (None, None) => plot.redraw(),
            _ => return Err(PyValueError::new_err("pass both vertices and faces, or neither")),
        }
        Ok(plot)
    }

    fn set_data(&self, vertices: &Bound<'_, PyAny>, faces: &Bound<'_, PyAny>) -> PyResult<()> {
        let vertices = parse_vertices(vertices)?;
        let faces = parse_faces(faces)?;
        *self.vertices.lock().unwrap_or_else(|p| p.into_inner()) = vertices;
        *self.faces.lock().unwrap_or_else(|p| p.into_inner()) = faces;
        self.redraw();
        Ok(())
    }

    fn zoom(&self, factor: f64, _fx: f64, _fy: f64) -> PyResult<()> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PyValueError::new_err("zoom factor must be positive"));
        }
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.zoom = (cam.zoom * factor).clamp(0.25, 8.0);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        let mut cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        cam.yaw += dx * 4.0;
        cam.pitch = (cam.pitch + dy * 3.0).clamp(-1.2, 1.2);
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = cam;
        self.redraw();
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.view.lock().unwrap_or_else(|p| p.into_inner()) = MeshView::default();
        self.redraw();
        Ok(())
    }

    fn view(&self) -> (f64, f64, f64) {
        let cam = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        (cam.yaw, cam.pitch, cam.zoom)
    }
}

impl PlotMesh {
    fn redraw(&self) {
        let style = *self.shell.style.lock().unwrap_or_else(|p| p.into_inner());
        let vertices = self.vertices.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let faces = self.faces.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let view = *self.view.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_mesh(self.shell.pixel_width, self.shell.pixel_height, &vertices, &faces, style, self.colormap, view);
        submit_frame(&self.shell.frames, self.shell.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.shell.interactive {
            return None;
        }
        let view = self.view.clone();
        let vertices = self.vertices.clone();
        let faces = self.faces.clone();
        let style = self.shell.style.clone();
        let frames = self.shell.frames.clone();
        let dispatch = self.shell.dispatch.clone();
        let colormap = self.colormap;
        let pw = self.shell.pixel_width;
        let ph = self.shell.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            if !orbit_pointer(&view, action, dx, dy, lx, ly, w, h, pw, ph) {
                return;
            }
            let style_now = *style.lock().unwrap_or_else(|p| p.into_inner());
            let cam = *view.lock().unwrap_or_else(|p| p.into_inner());
            let vertices = vertices.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let faces = faces.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let frame = raster_mesh(pw, ph, &vertices, &faces, style_now, colormap, cam);
            submit_frame(&frames, dispatch.as_ref(), frame);
        }))
    }

    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        self.shell.bind_dispatch(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        self.shell.describe(self.pointer())
    }
}

fn parse_pair(range: Option<(f64, f64)>) -> PyResult<Option<(f64, f64)>> {
    match range {
        None => Ok(None),
        Some((a, b)) if a.is_finite() && b.is_finite() && a != b => Ok(Some(if a < b { (a, b) } else { (b, a) })),
        Some(_) => Err(PyValueError::new_err("range bounds must be finite and unequal")),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlotHistogram>()?;
    m.add_class::<PlotBar>()?;
    m.add_class::<PlotContour>()?;
    m.add_class::<PlotSurface>()?;
    m.add_class::<PlotScatter3D>()?;
    m.add_class::<PlotMesh>()?;
    Ok(())
}
