//! GPU plot widgets (M7 7E): rasterize numpy series into RGBA frames uploaded via the Image layer path.

use std::sync::{Arc, Mutex};

use fastgui_core::plot::{
    axis_fraction, pan_range, pan_window, raster_heatmap, raster_line, raster_lines, raster_scatters, zoom_range,
    zoom_window, Colormap, PlotSeries, PlotStyle,
};
use fastgui_core::widget::{LayerFit, PointerCallback};
use fastgui_core::{CpuFrame, FrameSlot, MAX_CPU_FRAME_EXTENT};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

use crate::backend::CommandDispatch;
use crate::widgets::{DescribedWidget, StyleParams};
use crate::next_viewport_id;
use fastgui_core::widget::WidgetKind;

pub(crate) type IdCell = Arc<Mutex<Option<fastgui_core::widget::WidgetId>>>;
pub(crate) type SenderCell = Arc<Mutex<Option<CommandDispatch>>>;

pub(crate) fn rgba_u8(color: (f32, f32, f32, f32)) -> [u8; 4] {
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

/// `__cuda_array_interface__` (`<f4` / `<f8`, C-contiguous) copied with `cuMemcpyDtoH`.
/// Dtype and stride checks fail before any driver call. macOS has no CUDA interop.
/// Unverified against a real NVIDIA driver — same caveat as the rest of `fastgui-interop-cuda`.
fn cuda_array(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Vec<f64>, Vec<usize>)>> {
    if !obj.hasattr("__cuda_array_interface__").unwrap_or(false) {
        return Ok(None);
    }
    let iface = obj.getattr("__cuda_array_interface__")?;
    if iface.is_none() {
        return Ok(None);
    }
    let typestr: String = iface.get_item("typestr")?.extract()?;
    let elem = match typestr.as_str() {
        "<f4" => 4usize,
        "<f8" => 8usize,
        other => {
            return Err(PyValueError::new_err(format!(
                "CUDA array dtype '{other}' is not supported (need <f4 or <f8)"
            )));
        }
    };
    let shape: Vec<usize> = iface.get_item("shape")?.extract()?;
    if let Ok(strides_obj) = iface.get_item("strides") {
        if !strides_obj.is_none() {
            let strides: Vec<isize> = strides_obj.extract()?;
            if strides.len() != shape.len() {
                return Err(PyValueError::new_err("CUDA array strides do not match shape"));
            }
            let mut expect = elem as isize;
            for (&dim, &stride) in shape.iter().rev().zip(strides.iter().rev()) {
                if stride != expect {
                    return Err(PyValueError::new_err(
                        "CUDA array must be C-contiguous (<f4 or <f8); non-contiguous strides are not supported",
                    ));
                }
                expect = expect.saturating_mul(dim as isize);
            }
        }
    }
    let n = shape.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim)).ok_or_else(|| {
        PyValueError::new_err("CUDA array shape is too large")
    })?;
    let nbytes = n.checked_mul(elem).ok_or_else(|| PyValueError::new_err("CUDA array shape is too large"))?;
    if nbytes > 64 * 1024 * 1024 {
        return Err(PyValueError::new_err("CUDA array is larger than 64 MiB"));
    }
    if nbytes == 0 {
        return Ok(Some((Vec::new(), shape)));
    }
    let data = iface.get_item("data")?;
    let ptr = data.get_item(0)?.extract::<u64>()?;
    #[cfg(target_os = "macos")]
    {
        let _ = ptr;
        return Err(PyRuntimeError::new_err("CUDA arrays are not available on macOS"));
    }
    #[cfg(not(target_os = "macos"))]
    {
        let ctx = fastgui_interop_cuda::CudaContext::new().map_err(|err| {
            PyRuntimeError::new_err(format!("CUDA device array ingest failed: {err}"))
        })?;
        let mut bytes = vec![0u8; nbytes];
        ctx.copy_device_to_host(ptr, &mut bytes).map_err(|err| {
            PyRuntimeError::new_err(format!("CUDA device array ingest failed: {err}"))
        })?;
        let values = if elem == 4 {
            bytes
                .chunks_exact(4)
                .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect()
        } else {
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect()
        };
        Ok(Some((values, shape)))
    }
}

/// Accept a 1-D float buffer (numpy), a CUDA array, or any sequence of numbers.
fn float_series(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Some((values, shape)) = cuda_array(obj)? {
        if shape.len() != 1 {
            return Err(PyValueError::new_err("expected a 1-D CUDA float array"));
        }
        return Ok(values);
    }
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

/// Accept a 2-D float buffer (numpy), a CUDA array, or nested sequences → (values row-major, rows, cols).
fn float_grid(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Some((values, shape)) = cuda_array(obj)? {
        if shape.len() != 2 {
            return Err(PyValueError::new_err("expected a 2-D CUDA float array"));
        }
        return Ok((values, shape[0], shape[1]));
    }
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

/// `(fx, fy)` of a point on the letterboxed texture, plus drag deltas as fractions of that
/// drawn rect. The texture keeps its pixel aspect inside the widget, so widget fractions are
/// not texture fractions.
pub(crate) fn contained_pointer(lx: f32, ly: f32, dx: f32, dy: f32, w: f32, h: f32, pw: u32, ph: u32) -> (f64, f64, f64, f64) {
    let (ox, oy, dw, dh) = if w > 0.0 && h > 0.0 && pw > 0 && ph > 0 {
        let scale = (w / pw as f32).min(h / ph as f32);
        if scale.is_finite() && scale > 0.0 {
            let dw = pw as f32 * scale;
            let dh = ph as f32 * scale;
            ((w - dw) * 0.5, (h - dh) * 0.5, dw, dh)
        } else {
            (0.0, 0.0, w, h)
        }
    } else {
        (0.0, 0.0, w.max(1.0), h.max(1.0))
    };
    let fx = if dw > 0.0 { f64::from((lx - ox) / dw) } else { 0.5 };
    let fy = if dh > 0.0 { f64::from((ly - oy) / dh) } else { 0.5 };
    let ddx = if dw > 0.0 { f64::from(dx / dw) } else { 0.0 };
    let ddy = if dh > 0.0 { f64::from(dy / dh) } else { 0.0 };
    (fx, fy, ddx, ddy)
}

fn submit_frame(frames: &FrameSlot<CpuFrame>, dispatch: &Mutex<Option<CommandDispatch>>, frame: CpuFrame) {
    frames.submit(frame);
    if let Some(dispatch) = dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        dispatch.waker.wake();
    }
}

pub(crate) fn pixel_size(pixel_width: u32, pixel_height: u32) -> PyResult<(u32, u32)> {
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

fn parse_series(series: &Bound<'_, PyAny>, fallback: [u8; 4]) -> PyResult<Vec<(Vec<f64>, Vec<f64>, [u8; 4])>> {
    let seq = series.cast::<PySequence>().map_err(|_| {
        PyTypeError::new_err("series must be a sequence of (x, y) or (x, y, color)")
    })?;
    let n = seq.len()?;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let item = seq.get_item(i)?;
        let item_seq = item.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("each series is (x, y) or (x, y, color)")
        })?;
        let len = item_seq.len()?;
        if len != 2 && len != 3 {
            return Err(PyTypeError::new_err("each series is (x, y) or (x, y, color)"));
        }
        let x = float_series(&item_seq.get_item(0)?)?;
        let y = float_series(&item_seq.get_item(1)?)?;
        if x.len() != y.len() {
            return Err(PyValueError::new_err("x and y must have the same length"));
        }
        let color = if len == 3 { rgba_u8(item_seq.get_item(2)?.extract()?) } else { fallback };
        out.push((x, y, color));
    }
    Ok(out)
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

pub(crate) fn describe_image(
    image_id: u64,
    frames: &FrameSlot<CpuFrame>,
    id: &IdCell,
    sender: &SenderCell,
    flex_grow: f32,
    width: Option<f32>,
    height: Option<f32>,
    fit: LayerFit,
    on_pointer: Option<PointerCallback>,
) -> DescribedWidget {
    DescribedWidget::leaf_image(
        StyleParams::leaf(flex_grow, width, height),
        WidgetKind::Image {
            image_id,
            frames: frames.clone(),
            fit,
            on_pointer,
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
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Arc<Mutex<PlotStyle>>,
    thickness: f32,
    x_range: Arc<Mutex<Option<(f64, f64)>>>,
    y_range: Arc<Mutex<Option<(f64, f64)>>>,
    x_view: Arc<Mutex<Option<(f64, f64)>>>,
    y_view: Arc<Mutex<Option<(f64, f64)>>>,
    series: Arc<Mutex<Vec<(Vec<f64>, Vec<f64>, [u8; 4])>>>,
    interactive: bool,
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
        interactive=true,
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
        interactive: bool,
    ) -> PyResult<Self> {
        if thickness <= 0.0 {
            return Err(PyValueError::new_err("thickness must be positive"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Arc::new(Mutex::new(None)),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Arc::new(Mutex::new(style_from(background, color))),
            thickness,
            x_range: Arc::new(Mutex::new(parse_range(x_range)?)),
            y_range: Arc::new(Mutex::new(parse_range(y_range)?)),
            x_view: Arc::new(Mutex::new(parse_range(x_range)?)),
            y_view: Arc::new(Mutex::new(parse_range(y_range)?)),
            series: Arc::new(Mutex::new(Vec::new())),
            interactive,
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
                submit_frame(&plot.frames, plot.dispatch.as_ref(), frame);
            }
            _ => return Err(PyValueError::new_err("provide both x and y, or neither")),
        }
        Ok(plot)
    }

    /// Replace the series (any thread). `x`/`y` are 1-D float arrays or sequences.
    /// Keeps the current zoom; use `reset_view` to auto-range again.
    fn set_data(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let color = self.style.lock().unwrap_or_else(|p| p.into_inner()).series;
        self.replace_series(vec![(float_series(x)?, float_series(y)?, color)])
    }

    /// Several series. Each item is `(x, y)` or `(x, y, rgba)`.
    fn set_series(&self, series: &Bound<'_, PyAny>) -> PyResult<()> {
        let seq = series.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("series must be a sequence of (x, y) or (x, y, color)")
        })?;
        let n = seq.len()?;
        let mut out = Vec::with_capacity(n as usize);
        let fallback = self.style.lock().unwrap_or_else(|p| p.into_inner()).series;
        for i in 0..n {
            let item = seq.get_item(i)?;
            let item_seq = item.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err("each series is (x, y) or (x, y, color)")
            })?;
            let len = item_seq.len()?;
            if len != 2 && len != 3 {
                return Err(PyTypeError::new_err("each series is (x, y) or (x, y, color)"));
            }
            let x = float_series(&item_seq.get_item(0)?)?;
            let y = float_series(&item_seq.get_item(1)?)?;
            if x.len() != y.len() {
                return Err(PyValueError::new_err("x and y must have the same length"));
            }
            let color = if len == 3 {
                rgba_u8(item_seq.get_item(2)?.extract()?)
            } else {
                fallback
            };
            out.push((x, y, color));
        }
        self.replace_series(out)
    }

    #[pyo3(signature = (x_range=None, y_range=None))]
    fn set_range(&self, x_range: Option<(f64, f64)>, y_range: Option<(f64, f64)>) -> PyResult<()> {
        let xr = parse_range(x_range)?;
        let yr = parse_range(y_range)?;
        *self.x_range.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_range.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        self.redraw();
        Ok(())
    }

    /// Zoom by `factor` (>1 zooms in) around a point `(fx, fy)` in 0..1 of the widget.
    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        self.apply_zoom(factor, fx, fy);
        Ok(())
    }

    /// Pan by fractions of the widget (`dx` right, `dy` down are positive).
    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.apply_pan(dx, dy);
        Ok(())
    }

    /// Back to the explicit range, or auto-range when none was set.
    fn reset_view(&self) -> PyResult<()> {
        let xr = *self.x_range.lock().unwrap_or_else(|p| p.into_inner());
        let yr = *self.y_range.lock().unwrap_or_else(|p| p.into_inner());
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        self.redraw();
        Ok(())
    }

    /// The range currently drawn, `(x_min, x_max), (y_min, y_max)`.
    fn view_range(&self) -> ((f64, f64), (f64, f64)) {
        self.current_view()
    }
}

#[derive(Clone)]
struct XyState {
    series: Arc<Mutex<Vec<(Vec<f64>, Vec<f64>, [u8; 4])>>>,
    x_range: Arc<Mutex<Option<(f64, f64)>>>,
    y_range: Arc<Mutex<Option<(f64, f64)>>>,
    x_view: Arc<Mutex<Option<(f64, f64)>>>,
    y_view: Arc<Mutex<Option<(f64, f64)>>>,
    style: Arc<Mutex<PlotStyle>>,
    frames: FrameSlot<CpuFrame>,
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    pixel_width: u32,
    pixel_height: u32,
    stroke: f32,
    scatter: bool,
}

impl XyState {
    fn current_view(&self) -> ((f64, f64), (f64, f64)) {
        let series = self.series.lock().unwrap_or_else(|p| p.into_inner());
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (x, y, _) in series.iter() {
            xs.extend(x.iter().copied());
            ys.extend(y.iter().copied());
        }
        let home_x = *self.x_range.lock().unwrap_or_else(|p| p.into_inner());
        let home_y = *self.y_range.lock().unwrap_or_else(|p| p.into_inner());
        let view_x = *self.x_view.lock().unwrap_or_else(|p| p.into_inner());
        let view_y = *self.y_view.lock().unwrap_or_else(|p| p.into_inner());
        (
            fastgui_core::plot::resolve_for_view(view_x.or(home_x), &xs),
            fastgui_core::plot::resolve_for_view(view_y.or(home_y), &ys),
        )
    }

    fn raster(&self, style: PlotStyle, xr: (f64, f64), yr: (f64, f64)) {
        let series = self.series.lock().unwrap_or_else(|p| p.into_inner());
        let drawn: Vec<PlotSeries<'_>> = series
            .iter()
            .map(|(x, y, color)| PlotSeries { x: x.as_slice(), y: y.as_slice(), color: *color })
            .collect();
        let frame = if self.scatter {
            raster_scatters(self.pixel_width, self.pixel_height, &drawn, Some(xr), Some(yr), style, self.stroke)
        } else {
            raster_lines(self.pixel_width, self.pixel_height, &drawn, Some(xr), Some(yr), style, self.stroke)
        };
        submit_frame(&self.frames, self.dispatch.as_ref(), frame);
    }

    fn redraw(&self) {
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let (xr, yr) = self.current_view();
        self.raster(style, xr, yr);
    }

    fn apply_zoom(&self, factor: f64, fx: f64, fy: f64) {
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let ((x0, x1), (y0, y1)) = self.current_view();
        let tx = axis_fraction(fx, f64::from(self.pixel_width), f64::from(style.margin_left), f64::from(style.margin_right), false);
        let ty = axis_fraction(fy, f64::from(self.pixel_height), f64::from(style.margin_top), f64::from(style.margin_bottom), true);
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range((x0, x1), tx, factor));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(zoom_range((y0, y1), ty, factor));
        self.redraw();
    }

    fn apply_pan(&self, dx: f64, dy: f64) {
        let ((x0, x1), (y0, y1)) = self.current_view();
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range((x0, x1), -dx));
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(pan_range((y0, y1), dy));
        self.redraw();
    }

    fn pointer(&self, interactive: bool) -> Option<PointerCallback> {
        if !interactive {
            return None;
        }
        let state = self.clone();
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if matches!(action, 3 | 4 | 5) {
                return;
            }
            let style_now = *state.style.lock().unwrap_or_else(|p| p.into_inner());
            let (xs, ys, home_x, home_y, mut xr, mut yr) = {
                let series = state.series.lock().unwrap_or_else(|p| p.into_inner());
                let mut xs = Vec::new();
                let mut ys = Vec::new();
                for (x, y, _) in series.iter() {
                    xs.extend(x.iter().copied());
                    ys.extend(y.iter().copied());
                }
                let home_x = *state.x_range.lock().unwrap_or_else(|p| p.into_inner());
                let home_y = *state.y_range.lock().unwrap_or_else(|p| p.into_inner());
                let view_x = *state.x_view.lock().unwrap_or_else(|p| p.into_inner());
                let view_y = *state.y_view.lock().unwrap_or_else(|p| p.into_inner());
                let xr = fastgui_core::plot::resolve_for_view(view_x.or(home_x), &xs);
                let yr = fastgui_core::plot::resolve_for_view(view_y.or(home_y), &ys);
                (xs, ys, home_x, home_y, xr, yr)
            };
            match action {
                2 => {
                    xr = fastgui_core::plot::resolve_for_view(home_x, &xs);
                    yr = fastgui_core::plot::resolve_for_view(home_y, &ys);
                    *state.x_view.lock().unwrap_or_else(|p| p.into_inner()) = home_x;
                    *state.y_view.lock().unwrap_or_else(|p| p.into_inner()) = home_y;
                }
                0 => {
                    let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
                    let (fx, fy, _, _) = contained_pointer(lx, ly, dx, dy, w, h, state.pixel_width, state.pixel_height);
                    let tx = axis_fraction(fx, f64::from(state.pixel_width), f64::from(style_now.margin_left), f64::from(style_now.margin_right), false);
                    let ty = axis_fraction(fy, f64::from(state.pixel_height), f64::from(style_now.margin_top), f64::from(style_now.margin_bottom), true);
                    xr = zoom_range(xr, tx, factor);
                    yr = zoom_range(yr, ty, factor);
                    *state.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(xr);
                    *state.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(yr);
                }
                1 => {
                    let (_, _, ddx, ddy) = contained_pointer(lx, ly, dx, dy, w, h, state.pixel_width, state.pixel_height);
                    xr = pan_range(xr, -ddx);
                    yr = pan_range(yr, ddy);
                    *state.x_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(xr);
                    *state.y_view.lock().unwrap_or_else(|p| p.into_inner()) = Some(yr);
                }
                _ => return,
            }
            state.raster(style_now, xr, yr);
        }))
    }
}

impl PlotLine {
    fn state(&self) -> XyState {
        XyState {
            series: self.series.clone(),
            x_range: self.x_range.clone(),
            y_range: self.y_range.clone(),
            x_view: self.x_view.clone(),
            y_view: self.y_view.clone(),
            style: self.style.clone(),
            frames: self.frames.clone(),
            dispatch: self.dispatch.clone(),
            pixel_width: self.pixel_width,
            pixel_height: self.pixel_height,
            stroke: self.thickness,
            scatter: false,
        }
    }

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
            LayerFit::Contain,
            self.pointer(),
        )
    }

    fn replace_series(&self, series: Vec<(Vec<f64>, Vec<f64>, [u8; 4])>) -> PyResult<()> {
        for (x, y, _) in &series {
            if x.len() != y.len() {
                return Err(PyValueError::new_err("x and y must have the same length"));
            }
        }
        *self.series.lock().unwrap_or_else(|p| p.into_inner()) = series;
        self.state().redraw();
        Ok(())
    }

    fn current_view(&self) -> ((f64, f64), (f64, f64)) {
        self.state().current_view()
    }

    fn redraw(&self) {
        self.state().redraw();
    }

    fn apply_zoom(&self, factor: f64, fx: f64, fy: f64) {
        self.state().apply_zoom(factor, fx, fy);
    }

    fn apply_pan(&self, dx: f64, dy: f64) {
        self.state().apply_pan(dx, dy);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        self.state().pointer(self.interactive)
    }
}

/// Scatter plot uploaded as a GPU image layer.
#[pyclass]
pub(crate) struct PlotScatter {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Arc<Mutex<PlotStyle>>,
    point_radius: f32,
    x_range: Arc<Mutex<Option<(f64, f64)>>>,
    y_range: Arc<Mutex<Option<(f64, f64)>>>,
    x_view: Arc<Mutex<Option<(f64, f64)>>>,
    y_view: Arc<Mutex<Option<(f64, f64)>>>,
    series: Arc<Mutex<Vec<(Vec<f64>, Vec<f64>, [u8; 4])>>>,
    interactive: bool,
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
        interactive=true,
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
        interactive: bool,
    ) -> PyResult<Self> {
        if point_radius <= 0.0 {
            return Err(PyValueError::new_err("point_radius must be positive"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let xr = parse_range(x_range)?;
        let yr = parse_range(y_range)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Arc::new(Mutex::new(None)),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Arc::new(Mutex::new(style_from(background, color))),
            point_radius,
            x_range: Arc::new(Mutex::new(xr)),
            y_range: Arc::new(Mutex::new(yr)),
            x_view: Arc::new(Mutex::new(xr)),
            y_view: Arc::new(Mutex::new(yr)),
            series: Arc::new(Mutex::new(Vec::new())),
            interactive,
            width,
            height,
            flex_grow,
        };
        match (x, y) {
            (Some(x), Some(y)) => plot.set_data(x, y)?,
            (None, None) => plot.state().redraw(),
            _ => return Err(PyValueError::new_err("provide both x and y, or neither")),
        }
        Ok(plot)
    }

    /// Replace the points. Keeps the current zoom.
    fn set_data(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let color = self.style.lock().unwrap_or_else(|p| p.into_inner()).series;
        self.replace_series(vec![(float_series(x)?, float_series(y)?, color)])
    }

    /// Several series. Each item is `(x, y)` or `(x, y, rgba)`.
    fn set_series(&self, series: &Bound<'_, PyAny>) -> PyResult<()> {
        self.replace_series(parse_series(series, self.style.lock().unwrap_or_else(|p| p.into_inner()).series)?)
    }

    #[pyo3(signature = (x_range=None, y_range=None))]
    fn set_range(&self, x_range: Option<(f64, f64)>, y_range: Option<(f64, f64)>) -> PyResult<()> {
        let xr = parse_range(x_range)?;
        let yr = parse_range(y_range)?;
        *self.x_range.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_range.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        self.state().redraw();
        Ok(())
    }

    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        self.state().apply_zoom(factor, fx, fy);
        Ok(())
    }

    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.state().apply_pan(dx, dy);
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        let xr = *self.x_range.lock().unwrap_or_else(|p| p.into_inner());
        let yr = *self.y_range.lock().unwrap_or_else(|p| p.into_inner());
        *self.x_view.lock().unwrap_or_else(|p| p.into_inner()) = xr;
        *self.y_view.lock().unwrap_or_else(|p| p.into_inner()) = yr;
        self.state().redraw();
        Ok(())
    }

    fn view_range(&self) -> ((f64, f64), (f64, f64)) {
        self.state().current_view()
    }
}

impl PlotScatter {
    fn state(&self) -> XyState {
        XyState {
            series: self.series.clone(),
            x_range: self.x_range.clone(),
            y_range: self.y_range.clone(),
            x_view: self.x_view.clone(),
            y_view: self.y_view.clone(),
            style: self.style.clone(),
            frames: self.frames.clone(),
            dispatch: self.dispatch.clone(),
            pixel_width: self.pixel_width,
            pixel_height: self.pixel_height,
            stroke: self.point_radius,
            scatter: true,
        }
    }

    fn replace_series(&self, series: Vec<(Vec<f64>, Vec<f64>, [u8; 4])>) -> PyResult<()> {
        for (x, y, _) in &series {
            if x.len() != y.len() {
                return Err(PyValueError::new_err("x and y must have the same length"));
            }
        }
        *self.series.lock().unwrap_or_else(|p| p.into_inner()) = series;
        self.state().redraw();
        Ok(())
    }

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
            LayerFit::Contain,
            self.state().pointer(self.interactive),
        )
    }
}

/// 2-D heatmap uploaded as a GPU image layer (`viridis` / `magma` / `gray`).
#[pyclass]
pub(crate) struct PlotHeatmap {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: Arc<Mutex<PlotStyle>>,
    colormap: Colormap,
    v_range: Arc<Mutex<Option<(f64, f64)>>>,
    grid: Arc<Mutex<(Vec<f64>, usize, usize)>>,
    /// Visible window `(col0, col1, row0, row1)` in unit grid fractions. `row0` is the top.
    window: Arc<Mutex<(f64, f64, f64, f64)>>,
    interactive: bool,
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
        interactive=true,
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
        interactive: bool,
    ) -> PyResult<Self> {
        let colormap = Colormap::parse(colormap).ok_or_else(|| {
            PyValueError::new_err("colormap must be 'viridis', 'magma', or 'gray'")
        })?;
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let plot = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Arc::new(Mutex::new(None)),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: Arc::new(Mutex::new(style_from(background, None))),
            colormap,
            v_range: Arc::new(Mutex::new(parse_range(v_range)?)),
            grid: Arc::new(Mutex::new((Vec::new(), 0, 0))),
            window: Arc::new(Mutex::new((0.0, 1.0, 0.0, 1.0))),
            interactive,
            width,
            height,
            flex_grow,
        };
        if let Some(values) = values {
            plot.set_data(values)?;
        } else {
            plot.redraw();
        }
        Ok(plot)
    }

    /// Replace the grid. `values` is a 2-D float array or nested sequences (row-major).
    /// Keeps the current pan/zoom window.
    fn set_data(&self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (grid, rows, cols) = float_grid(values)?;
        if rows == 0 || cols == 0 {
            return Err(PyValueError::new_err("heatmap must be at least 1x1"));
        }
        *self.grid.lock().unwrap_or_else(|p| p.into_inner()) = (grid, rows, cols);
        self.redraw();
        Ok(())
    }

    /// Zoom by `factor` (>1 zooms in) around `(fx, fy)` in 0..1 of the widget.
    fn zoom(&self, factor: f64, fx: f64, fy: f64) -> PyResult<()> {
        self.apply_zoom(factor, fx, fy);
        Ok(())
    }

    /// Pan by fractions of the widget (`dx` right, `dy` down are positive).
    fn pan(&self, dx: f64, dy: f64) -> PyResult<()> {
        self.apply_pan(dx, dy);
        Ok(())
    }

    fn reset_view(&self) -> PyResult<()> {
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = (0.0, 1.0, 0.0, 1.0);
        self.redraw();
        Ok(())
    }

    /// Visible window `(col0, col1, row0, row1)` in unit grid fractions (`row0` is the top).
    fn window(&self) -> (f64, f64, f64, f64) {
        *self.window.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl PlotHeatmap {
    fn redraw(&self) {
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let (grid, rows, cols) = self.grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let v_range = *self.v_range.lock().unwrap_or_else(|p| p.into_inner());
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        let frame = raster_heatmap(
            self.pixel_width,
            self.pixel_height,
            &grid,
            rows,
            cols,
            v_range,
            style,
            self.colormap,
            Some(window),
        );
        let frame = if frame.width == 0 {
            raster_line(self.pixel_width, self.pixel_height, &[], &[], None, None, style, 1.0)
        } else {
            frame
        };
        submit_frame(&self.frames, self.dispatch.as_ref(), frame);
    }

    fn apply_zoom(&self, factor: f64, fx: f64, fy: f64) {
        let style = *self.style.lock().unwrap_or_else(|p| p.into_inner());
        let tx = axis_fraction(fx, f64::from(self.pixel_width), f64::from(style.margin_left), f64::from(style.margin_right), false);
        // `row0` is the top of the window, so fraction 0 is the top of the plot, not the bottom.
        let ty = axis_fraction(fy, f64::from(self.pixel_height), f64::from(style.margin_top), f64::from(style.margin_bottom), false);
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = zoom_window(window, tx, ty, factor);
        self.redraw();
    }

    fn apply_pan(&self, dx: f64, dy: f64) {
        let window = *self.window.lock().unwrap_or_else(|p| p.into_inner());
        *self.window.lock().unwrap_or_else(|p| p.into_inner()) = pan_window(window, -dx, dy);
        self.redraw();
    }

    fn pointer(&self) -> Option<PointerCallback> {
        if !self.interactive {
            return None;
        }
        let style = self.style.clone();
        let window = self.window.clone();
        let grid = self.grid.clone();
        let v_range = self.v_range.clone();
        let frames = self.frames.clone();
        let dispatch = self.dispatch.clone();
        let colormap = self.colormap;
        let pw = self.pixel_width;
        let ph = self.pixel_height;
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
                    view = pan_window(view, -ddx, ddy);
                }
                _ => return,
            }
            *window.lock().unwrap_or_else(|p| p.into_inner()) = view;
            let (values, rows, cols) = grid.lock().unwrap_or_else(|p| p.into_inner()).clone();
            let vr = *v_range.lock().unwrap_or_else(|p| p.into_inner());
            let frame = raster_heatmap(pw, ph, &values, rows, cols, vr, style_now, colormap, Some(view));
            if frame.width != 0 {
                submit_frame(&frames, dispatch.as_ref(), frame);
            }
        }))
    }

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
            LayerFit::Contain,
            self.pointer(),
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlotLine>()?;
    m.add_class::<PlotScatter>()?;
    m.add_class::<PlotHeatmap>()?;
    Ok(())
}
