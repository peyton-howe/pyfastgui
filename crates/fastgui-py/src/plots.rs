//! GPU plot widgets (M7 7E): rasterize numpy series into RGBA frames uploaded via the Image layer path.

use std::sync::{Arc, Mutex};

use fastgui_core::plot::{raster_heatmap, raster_line, raster_scatter, Colormap, PlotStyle};
use fastgui_core::{CpuFrame, FrameSlot, MAX_CPU_FRAME_EXTENT};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

use crate::backend::CommandDispatch;
use crate::widgets::{DescribedWidget, StyleParams};
use crate::next_viewport_id;
use fastgui_core::widget::WidgetKind;

type IdCell = Arc<Mutex<Option<fastgui_core::widget::WidgetId>>>;
type SenderCell = Arc<Mutex<Option<CommandDispatch>>>;

fn rgba_u8(color: (f32, f32, f32, f32)) -> [u8; 4] {
    [
        (color.0.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.1.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.2.clamp(0.0, 1.0) * 255.0).round() as u8,
        (color.3.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

fn parse_range(range: Option<(f64, f64)>) -> PyResult<Option<(f64, f64)>> {
    match range {
        None => Ok(None),
        Some((a, b)) if a.is_finite() && b.is_finite() && a != b => Ok(Some(if a < b { (a, b) } else { (b, a) })),
        Some(_) => Err(PyValueError::new_err("range bounds must be finite and unequal")),
    }
}

/// Accept a 1-D float buffer (numpy) or any sequence of numbers.
fn float_series(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        if buf.dimensions() == 1 && buf.is_c_contiguous() {
            return buf.to_vec(obj.py());
        }
    }
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if buf.dimensions() == 1 && buf.is_c_contiguous() {
            let v = buf.to_vec(obj.py())?;
            return Ok(v.into_iter().map(|x| x as f64).collect());
        }
    }
    if let Ok(tolist) = obj.call_method0("tolist") {
        return float_series(&tolist);
    }
    let seq = obj.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("expected a 1-D float array or sequence of numbers")
    })?;
    let mut out = Vec::with_capacity(seq.len()? as usize);
    for i in 0..seq.len()? {
        out.push(seq.get_item(i)?.extract::<f64>()?);
    }
    Ok(out)
}

/// Accept a 2-D float buffer (numpy) or nested sequences → (values row-major, rows, cols).
fn float_grid(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        if buf.dimensions() == 2 && buf.is_c_contiguous() {
            let shape = buf.shape();
            let rows = shape[0];
            let cols = shape[1];
            return Ok((buf.to_vec(obj.py())?, rows, cols));
        }
    }
    if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        if buf.dimensions() == 2 && buf.is_c_contiguous() {
            let shape = buf.shape();
            let rows = shape[0];
            let cols = shape[1];
            let v = buf.to_vec(obj.py())?;
            return Ok((v.into_iter().map(|x| x as f64).collect(), rows, cols));
        }
    }
    if let Ok(tolist) = obj.call_method0("tolist") {
        return float_grid(&tolist);
    }
    let seq = obj.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("expected a 2-D float array or nested sequence")
    })?;
    let rows = seq.len()? as usize;
    if rows == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    let first = seq.get_item(0)?;
    let first_row = float_series(&first)?;
    let cols = first_row.len();
    let mut values = Vec::with_capacity(rows * cols);
    values.extend(first_row);
    for i in 1..rows {
        let row = float_series(&seq.get_item(i)?)?;
        if row.len() != cols {
            return Err(PyValueError::new_err("heatmap rows must all have the same length"));
        }
        values.extend(row);
    }
    Ok((values, rows, cols))
}

fn submit_frame(frames: &FrameSlot<CpuFrame>, dispatch: &Mutex<Option<CommandDispatch>>, frame: CpuFrame) {
    frames.submit(frame);
    if let Some(dispatch) = dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        dispatch.waker.wake();
    }
}

fn pixel_size(pixel_width: u32, pixel_height: u32) -> PyResult<(u32, u32)> {
    if pixel_width == 0 || pixel_height == 0 {
        return Err(PyValueError::new_err("pixel_width and pixel_height must be positive"));
    }
    if pixel_width > MAX_CPU_FRAME_EXTENT || pixel_height > MAX_CPU_FRAME_EXTENT {
        return Err(PyValueError::new_err(format!(
            "plot edge must be <= {MAX_CPU_FRAME_EXTENT} pixels"
        )));
    }
    Ok((pixel_width, pixel_height))
}

fn style_from(
    background: Option<(f32, f32, f32, f32)>,
    series: Option<(f32, f32, f32, f32)>,
) -> PlotStyle {
    let mut style = PlotStyle::default();
    if let Some(c) = background {
        style.background = rgba_u8(c);
    }
    if let Some(c) = series {
        style.series = rgba_u8(c);
    }
    style
}

fn describe_image(
    image_id: u64,
    frames: &FrameSlot<CpuFrame>,
    id: &IdCell,
    sender: &SenderCell,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
) -> DescribedWidget {
    DescribedWidget::leaf_image(
        StyleParams::leaf(flex_grow, width, height),
        WidgetKind::Image {
            image_id,
            frames: frames.clone(),
        },
        id.clone(),
        sender.clone(),
    )
}

/// Polyline plot uploaded as a GPU image layer. Call `set_data` from any thread.
#[pyclass]
pub(crate) struct PlotLine {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Mutex<PlotStyle>,
    thickness: f32,
    x_range: Mutex<Option<(f64, f64)>>,
    y_range: Mutex<Option<(f64, f64)>>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl PlotLine {
    #[new]
    #[pyo3(signature = (
        x=None,
        y=None,
        *,
        color=None,
        background=None,
        thickness=2.0,
        x_range=None,
        y_range=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        thickness: f32,
        x_range: Option<(f64, f64)>,
        y_range: Option<(f64, f64)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        if thickness <= 0.0 {
            return Err(PyValueError::new_err("thickness must be positive"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Mutex::new(style_from(background, color)),
            thickness,
            x_range: Mutex::new(parse_range(x_range)?),
            y_range: Mutex::new(parse_range(y_range)?),
            width,
            height,
            flex_grow,
        };
        match (x, y) {
            (Some(x), Some(y)) => plot.set_data(x, y)?,
            (None, None) => {
                let style = *plot.style.lock().unwrap_or_else(|p| p.into_inner());
                let frame = raster_line(
                    plot.pixel_width,
                    plot.pixel_height,
                    &[],
                    &[],
                    None,
                    None,
                    style,
                    plot.thickness,
                );
                submit_frame(&plot.frames, &plot.dispatch, frame);
            }
            _ => return Err(PyValueError::new_err("provide both x and y, or neither")),
        }
        Ok(plot)
    }

    /// Replace the series (any thread). `x`/`y` are 1-D float arrays or sequences.
    fn set_data(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let xv = float_series(x)?;
        let yv = float_series(y)?;
        if xv.len() != yv.len() {
            return Err(PyValueError::new_err("x and y must have the same length"));
        }
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let x_range = *self.x_range.lock().unwrap_or_else(|p| p.into_inner());
        let y_range = *self.y_range.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_line(
            self.pixel_width,
            self.pixel_height,
            &xv,
            &yv,
            x_range,
            y_range,
            style,
            self.thickness,
        );
        submit_frame(&self.frames, &self.dispatch, frame);
        Ok(())
    }

    #[pyo3(signature = (x_range=None, y_range=None))]
    fn set_range(&self, x_range: Option<(f64, f64)>, y_range: Option<(f64, f64)>) -> PyResult<()> {
        *self.x_range.lock().unwrap_or_else(|p| p.into_inner()) = parse_range(x_range)?;
        *self.y_range.lock().unwrap_or_else(|p| p.into_inner()) = parse_range(y_range)?;
        Ok(())
    }
}

impl PlotLine {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        describe_image(
            self.image_id,
            &self.frames,
            &self.id,
            &self.sender,
            self.flex_grow,
            self.width,
            self.height,
        )
    }
}

/// Scatter plot uploaded as a GPU image layer.
#[pyclass]
pub(crate) struct PlotScatter {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Mutex<PlotStyle>,
    point_radius: f32,
    x_range: Mutex<Option<(f64, f64)>>,
    y_range: Mutex<Option<(f64, f64)>>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl PlotScatter {
    #[new]
    #[pyo3(signature = (
        x=None,
        y=None,
        *,
        color=None,
        background=None,
        point_radius=3.0,
        x_range=None,
        y_range=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        x: Option<&Bound<'_, PyAny>>,
        y: Option<&Bound<'_, PyAny>>,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        point_radius: f32,
        x_range: Option<(f64, f64)>,
        y_range: Option<(f64, f64)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        if point_radius <= 0.0 {
            return Err(PyValueError::new_err("point_radius must be positive"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Mutex::new(style_from(background, color)),
            point_radius,
            x_range: Mutex::new(parse_range(x_range)?),
            y_range: Mutex::new(parse_range(y_range)?),
            width,
            height,
            flex_grow,
        };
        match (x, y) {
            (Some(x), Some(y)) => plot.set_data(x, y)?,
            (None, None) => {
                let style = *plot.style.lock().unwrap_or_else(|p| p.into_inner());
                let frame = raster_scatter(
                    plot.pixel_width,
                    plot.pixel_height,
                    &[],
                    &[],
                    None,
                    None,
                    style,
                    plot.point_radius,
                );
                submit_frame(&plot.frames, &plot.dispatch, frame);
            }
            _ => return Err(PyValueError::new_err("provide both x and y, or neither")),
        }
        Ok(plot)
    }

    fn set_data(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let xv = float_series(x)?;
        let yv = float_series(y)?;
        if xv.len() != yv.len() {
            return Err(PyValueError::new_err("x and y must have the same length"));
        }
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let x_range = *self.x_range.lock().unwrap_or_else(|p| p.into_inner());
        let y_range = *self.y_range.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_scatter(
            self.pixel_width,
            self.pixel_height,
            &xv,
            &yv,
            x_range,
            y_range,
            style,
            self.point_radius,
        );
        submit_frame(&self.frames, &self.dispatch, frame);
        Ok(())
    }
}

impl PlotScatter {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        describe_image(
            self.image_id,
            &self.frames,
            &self.id,
            &self.sender,
            self.flex_grow,
            self.width,
            self.height,
        )
    }
}

/// 2-D heatmap uploaded as a GPU image layer (`viridis` / `magma` / `gray`).
#[pyclass]
pub(crate) struct PlotHeatmap {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Mutex<PlotStyle>,
    colormap: Colormap,
    v_range: Mutex<Option<(f64, f64)>>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl PlotHeatmap {
    #[new]
    #[pyo3(signature = (
        values=None,
        *,
        colormap="viridis",
        background=None,
        v_range=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        values: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        background: Option<(f32, f32, f32, f32)>,
        v_range: Option<(f64, f64)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        let colormap = Colormap::parse(colormap).ok_or_else(|| {
            PyValueError::new_err("colormap must be 'viridis', 'magma', or 'gray'")
        })?;
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Mutex::new(style_from(background, None)),
            colormap,
            v_range: Mutex::new(parse_range(v_range)?),
            width,
            height,
            flex_grow,
        };
        if let Some(values) = values {
            plot.set_data(values)?;
        } else {
            let style = *plot.style.lock().unwrap_or_else(|p| p.into_inner());
            let frame = raster_heatmap(
                plot.pixel_width,
                plot.pixel_height,
                &[],
                0,
                0,
                None,
                style,
                plot.colormap,
            );
            // empty heatmap returns 0-size when rows=0 — force empty frame with axes
            let frame = if frame.width == 0 {
                raster_line(plot.pixel_width, plot.pixel_height, &[], &[], None, None, style, 1.0)
            } else {
                frame
            };
            submit_frame(&plot.frames, &plot.dispatch, frame);
        }
        Ok(plot)
    }

    /// Replace the grid. `values` is a 2-D float array or nested sequences (row-major).
    fn set_data(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (grid, rows, cols) = float_grid(values)?;
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("heatmap must be at least 1x1"));
        }
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let v_range = *self.v_range.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_heatmap(
            self.pixel_width,
            self.pixel_height,
            &grid,
            rows,
            cols,
            v_range,
            style,
            self.colormap,
        );
        submit_frame(&self.frames, &self.dispatch, frame);
        Ok(())
    }
}

impl PlotHeatmap {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe(&self) -> DescribedWidget {
        describe_image(
            self.image_id,
            &self.frames,
            &self.id,
            &self.sender,
            self.flex_grow,
            self.width,
            self.height,
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlotLine>()?;
    m.add_class::<PlotScatter>()?;
    m.add_class::<PlotHeatmap>()?;
    Ok(())
}
