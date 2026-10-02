//! Gauge, timeline, and node-graph widgets (M7 7E). Rasterized to RGBA and uploaded as image layers.

use std::sync::{Arc, Mutex};

use fastgui_core::plot::{raster_gauge, raster_graph, raster_timeline, GraphNode, PlotStyle, TimelineClip};
use fastgui_core::widget::PointerCallback;
use fastgui_core::{CpuFrame, FrameSlot};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PySequence};

use crate::backend::CommandDispatch;
use crate::next_viewport_id;
use crate::plots::{describe_image, pixel_size, rgba_u8};
use crate::widgets::DescribedWidget;

type IdCell = Arc<Mutex<Option<fastgui_core::widget::WidgetId>>>;
type SenderCell = Arc<Mutex<Option<CommandDispatch>>>;

fn submit(frames: &FrameSlot<CpuFrame>, dispatch: &Mutex<Option<CommandDispatch>>, frame: CpuFrame) {
    frames.submit(frame);
    if let Some(dispatch) = dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
        dispatch.waker.wake();
    }
}

fn style_of(background: Option<(f32, f32, f32, f32)>, series: Option<(f32, f32, f32, f32)>) -> PlotStyle {
    let mut style = PlotStyle::default();
    if let Some(c) = background {
        style.background = rgba_u8(c);
    }
    if let Some(c) = series {
        style.series = rgba_u8(c);
    }
    style
}

struct NodeRec {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    title: String,
    selected: bool,
}

/// Semicircular gauge. `value` is clamped into `min`..`max`.
#[pyclass]
pub(crate) struct Gauge {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: PlotStyle,
    value: Mutex<f64>,
    min: f64,
    max: f64,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl Gauge {
    #[new]
    #[pyo3(signature = (
        value=0.0,
        min=0.0,
        max=1.0,
        *,
        color=None,
        background=None,
        pixel_width=320,
        pixel_height=180,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        value: f64,
        min: f64,
        max: f64,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        if !value.is_finite() || !min.is_finite() || !max.is_finite() || min == max {
            return Err(PyValueError::new_err("gauge value, min, and max must be finite and min != max"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let gauge = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: style_of(background, color),
            value: Mutex::new(value),
            min,
            max,
            width,
            height,
            flex_grow,
        };
        gauge.redraw();
        Ok(gauge)
    }

    #[getter]
    fn value(&self) -> f64 {
        *self.value.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn set_value(&self, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(PyValueError::new_err("gauge value must be finite"));
        }
        *self.value.lock().unwrap_or_else(|p| p.into_inner()) = value;
        self.redraw();
        Ok(())
    }
}

impl Gauge {
    fn t(&self) -> f64 {
        let value = *self.value.lock().unwrap_or_else(|p| p.into_inner());
        let span = self.max - self.min;
        if span.abs() < 1e-12 {
            0.0
        } else {
            ((value - self.min) / span).clamp(0.0, 1.0)
        }
    }

    fn redraw(&self) {
        let frame = raster_gauge(self.pixel_width, self.pixel_height, self.t(), self.style);
        submit(&self.frames, &self.dispatch, frame);
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
            fastgui_core::widget::LayerFit::Contain,
            None,
        )
    }
}

/// Horizontal clip tracks and a playhead.
#[pyclass]
pub(crate) struct Timeline {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Mutex<Option<CommandDispatch>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: PlotStyle,
    duration: Mutex<f64>,
    time: Mutex<f64>,
    tracks: Mutex<Vec<Vec<TimelineClip>>>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl Timeline {
    #[new]
    #[pyo3(signature = (
        tracks=None,
        duration=10.0,
        time=0.0,
        *,
        color=None,
        background=None,
        pixel_width=640,
        pixel_height=160,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tracks: Option<&Bound<'_, PyAny>>,
        duration: f64,
        time: f64,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        if !duration.is_finite() || duration <= 0.0 || !time.is_finite() {
            return Err(PyValueError::new_err("duration must be positive and time finite"));
        }
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let style = style_of(background, color);
        let parsed = match tracks {
            Some(tracks) => parse_tracks(tracks, [255, 150, 70, 255])?,
            None => Vec::new(),
        };
        let timeline = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Mutex::new(None),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style,
            duration: Mutex::new(duration),
            time: Mutex::new(time),
            tracks: Mutex::new(parsed),
            width,
            height,
            flex_grow,
        };
        timeline.redraw();
        Ok(timeline)
    }

    #[getter]
    fn time(&self) -> f64 {
        *self.time.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn set_time(&self, time: f64) -> PyResult<()> {
        if !time.is_finite() {
            return Err(PyValueError::new_err("time must be finite"));
        }
        *self.time.lock().unwrap_or_else(|p| p.into_inner()) = time;
        self.redraw();
        Ok(())
    }

    fn set_duration(&self, duration: f64) -> PyResult<()> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(PyValueError::new_err("duration must be positive"));
        }
        *self.duration.lock().unwrap_or_else(|p| p.into_inner()) = duration;
        self.redraw();
        Ok(())
    }

    /// `tracks` is a sequence of tracks; each track is a sequence of `(start, end)` or
    /// `(start, end, rgba)` clips, in the same units as `duration`.
    fn set_tracks(&self, tracks: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = parse_tracks(tracks, [255, 150, 70, 255])?;
        *self.tracks.lock().unwrap_or_else(|p| p.into_inner()) = parsed;
        self.redraw();
        Ok(())
    }
}

impl Timeline {
    fn redraw(&self) {
        let duration = *self.duration.lock().unwrap_or_else(|p| p.into_inner());
        let time = *self.time.lock().unwrap_or_else(|p| p.into_inner());
        let tracks = self.tracks.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let frame = raster_timeline(self.pixel_width, self.pixel_height, duration, time, &tracks, self.style);
        submit(&self.frames, &self.dispatch, frame);
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
            fastgui_core::widget::LayerFit::Stretch,
            None,
        )
    }
}

fn parse_tracks(obj: &Bound<'_, PyAny>, fallback: [u8; 4]) -> PyResult<Vec<Vec<TimelineClip>>> {
    let seq = obj.cast::<PySequence>().map_err(|_| PyTypeError::new_err("tracks must be a sequence of clip lists"))?;
    let mut tracks = Vec::with_capacity(seq.len()? as usize);
    for i in 0..seq.len()? {
        let track = seq.get_item(i)?;
        let clips_seq = track.cast::<PySequence>().map_err(|_| PyTypeError::new_err("each track is a sequence of clips"))?;
        let mut clips = Vec::with_capacity(clips_seq.len()? as usize);
        for j in 0..clips_seq.len()? {
            let clip = clips_seq.get_item(j)?;
            let parts = clip.cast::<PySequence>().map_err(|_| {
                PyTypeError::new_err("each clip is (start, end) or (start, end, color)")
            })?;
            let len = parts.len()?;
            if len != 2 && len != 3 {
                return Err(PyTypeError::new_err("each clip is (start, end) or (start, end, color)"));
            }
            let start: f64 = parts.get_item(0)?.extract()?;
            let end: f64 = parts.get_item(1)?.extract()?;
            let color = if len == 3 { rgba_u8(parts.get_item(2)?.extract()?) } else { fallback };
            clips.push(TimelineClip { start, end, color });
        }
        tracks.push(clips);
    }
    Ok(tracks)
}

/// Nodes you can drag. Positions are pixels of the raster frame, which keeps its aspect inside the widget.
#[pyclass]
pub(crate) struct NodeGraph {
    image_id: u64,
    frames: FrameSlot<CpuFrame>,
    dispatch: Arc<Mutex<Option<CommandDispatch>>>,
    id: IdCell,
    sender: SenderCell,
    pixel_width: u32,
    pixel_height: u32,
    style: PlotStyle,
    nodes: Arc<Mutex<Vec<NodeRec>>>,
    edges: Arc<Mutex<Vec<(usize, usize)>>>,
    drag: Arc<Mutex<Option<usize>>>,
    width: Option<f32>,
    height: Option<f32>,
    flex_grow: f32,
}

#[pymethods]
impl NodeGraph {
    #[new]
    #[pyo3(signature = (
        nodes=None,
        edges=None,
        *,
        color=None,
        background=None,
        pixel_width=640,
        pixel_height=360,
        width=None,
        height=None,
        flex_grow=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nodes: Option<&Bound<'_, PyAny>>,
        edges: Option<&Bound<'_, PyAny>>,
        color: Option<(f32, f32, f32, f32)>,
        background: Option<(f32, f32, f32, f32)>,
        pixel_width: u32,
        pixel_height: u32,
        width: Option<f32>,
        height: Option<f32>,
        flex_grow: f32,
    ) -> PyResult<Self> {
        let (pixel_width, pixel_height) = pixel_size(pixel_width, pixel_height)?;
        let graph = Self {
            image_id: next_viewport_id(),
            frames: FrameSlot::new(),
            dispatch: Arc::new(Mutex::new(None)),
            id: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            pixel_width,
            pixel_height,
            style: style_of(background, color),
            nodes: Arc::new(Mutex::new(Vec::new())),
            edges: Arc::new(Mutex::new(Vec::new())),
            drag: Arc::new(Mutex::new(None)),
            width,
            height,
            flex_grow,
        };
        if let Some(nodes) = nodes {
            graph.set_nodes(nodes)?;
        } else {
            graph.redraw();
        }
        if let Some(edges) = edges {
            graph.set_edges(edges)?;
        }
        Ok(graph)
    }

    /// Each node is `(x, y, title)` or `(x, y, w, h, title)`, in frame pixels.
    fn set_nodes(&self, nodes: &Bound<'_, PyAny>) -> PyResult<()> {
        *self.nodes.lock().unwrap_or_else(|p| p.into_inner()) = parse_nodes(nodes)?;
        *self.drag.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self.redraw();
        Ok(())
    }

    /// Each edge is `(from_index, to_index)`.
    fn set_edges(&self, edges: &Bound<'_, PyAny>) -> PyResult<()> {
        *self.edges.lock().unwrap_or_else(|p| p.into_inner()) = parse_edges(edges)?;
        self.redraw();
        Ok(())
    }

    /// Current nodes as `(x, y, w, h, title, selected)`.
    fn nodes(&self) -> Vec<(f32, f32, f32, f32, String, bool)> {
        self.nodes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|n| (n.x, n.y, n.w, n.h, n.title.clone(), n.selected))
            .collect()
    }
}

impl NodeGraph {
    fn redraw(&self) {
        let nodes = self.nodes.lock().unwrap_or_else(|p| p.into_inner());
        let edges = self.edges.lock().unwrap_or_else(|p| p.into_inner());
        let drawn: Vec<GraphNode> = nodes
            .iter()
            .map(|n| GraphNode {
                x: n.x,
                y: n.y,
                w: n.w,
                h: n.h,
                title: n.title.clone(),
                selected: n.selected,
            })
            .collect();
        let frame = raster_graph(self.pixel_width, self.pixel_height, &drawn, &edges, self.style);
        drop(nodes);
        drop(edges);
        submit(&self.frames, self.dispatch.as_ref(), frame);
    }

    fn pointer(&self) -> Option<PointerCallback> {
        let nodes = self.nodes.clone();
        let edges = self.edges.clone();
        let drag = self.drag.clone();
        let frames = self.frames.clone();
        let dispatch = self.dispatch.clone();
        let style = self.style;
        let pw = self.pixel_width;
        let ph = self.pixel_height;
        Some(Arc::new(move |action, dx, dy, lx, ly, w, h| {
            if action == 3 {
                return;
            }
            // The frame is letterboxed, so widget fractions are not frame pixels.
            let (fx, fy, ddx, ddy) = crate::plots::contained_pointer(lx, ly, dx, dy, w, h, pw, ph);
            let px = fx as f32 * pw as f32;
            let py = fy as f32 * ph as f32;
            let dpx = ddx as f32 * pw as f32;
            let dpy = ddy as f32 * ph as f32;
            {
                let mut nodes = nodes.lock().unwrap_or_else(|p| p.into_inner());
                match action {
                    4 => {
                        let hit = nodes.iter().enumerate().rev().find(|(_, n)| {
                            px >= n.x && px <= n.x + n.w && py >= n.y && py <= n.y + n.h
                        }).map(|(i, _)| i);
                        for (i, node) in nodes.iter_mut().enumerate() {
                            node.selected = Some(i) == hit;
                        }
                        *drag.lock().unwrap_or_else(|p| p.into_inner()) = hit;
                    }
                    1 => {
                        if let Some(index) = *drag.lock().unwrap_or_else(|p| p.into_inner()) {
                            if let Some(node) = nodes.get_mut(index) {
                                node.x += dpx;
                                node.y += dpy;
                            }
                        } else {
                            return;
                        }
                    }
                    5 => {
                        *drag.lock().unwrap_or_else(|p| p.into_inner()) = None;
                        return;
                    }
                    _ => return,
                }
                let drawn: Vec<GraphNode> = nodes
                    .iter()
                    .map(|n| GraphNode {
                        x: n.x,
                        y: n.y,
                        w: n.w,
                        h: n.h,
                        title: n.title.clone(),
                        selected: n.selected,
                    })
                    .collect();
                let edges = edges.lock().unwrap_or_else(|p| p.into_inner()).clone();
                let frame = raster_graph(pw, ph, &drawn, &edges, style);
                drop(nodes);
                submit(&frames, dispatch.as_ref(), frame);
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
            fastgui_core::widget::LayerFit::Contain,
            self.pointer(),
        )
    }
}

fn parse_nodes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<NodeRec>> {
    let seq = obj.cast::<PySequence>().map_err(|_| PyTypeError::new_err("nodes must be a sequence"))?;
    let mut nodes = Vec::with_capacity(seq.len()? as usize);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let parts = item.cast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("each node is (x, y, title) or (x, y, w, h, title)")
        })?;
        let len = parts.len()?;
        let (x, y, w, h, title) = if len == 3 {
            (
                parts.get_item(0)?.extract::<f64>()? as f32,
                parts.get_item(1)?.extract::<f64>()? as f32,
                120.0,
                40.0,
                parts.get_item(2)?.extract::<String>()?,
            )
        } else if len == 5 {
            (
                parts.get_item(0)?.extract::<f64>()? as f32,
                parts.get_item(1)?.extract::<f64>()? as f32,
                parts.get_item(2)?.extract::<f64>()? as f32,
                parts.get_item(3)?.extract::<f64>()? as f32,
                parts.get_item(4)?.extract::<String>()?,
            )
        } else {
            return Err(PyTypeError::new_err("each node is (x, y, title) or (x, y, w, h, title)"));
        };
        if !x.is_finite() || !y.is_finite() || w <= 0.0 || h <= 0.0 {
            return Err(PyValueError::new_err("node position must be finite and size positive"));
        }
        nodes.push(NodeRec { x, y, w, h, title, selected: false });
    }
    Ok(nodes)
}

fn parse_edges(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let seq = obj.cast::<PySequence>().map_err(|_| PyTypeError::new_err("edges must be a sequence of (a, b)"))?;
    let mut edges = Vec::with_capacity(seq.len()? as usize);
    for i in 0..seq.len()? {
        let item = seq.get_item(i)?;
        let parts = item.cast::<PySequence>().map_err(|_| PyTypeError::new_err("each edge is (from, to)"))?;
        if parts.len()? != 2 {
            return Err(PyTypeError::new_err("each edge is (from, to)"));
        }
        let a: usize = parts.get_item(0)?.extract()?;
        let b: usize = parts.get_item(1)?.extract()?;
        edges.push((a, b));
    }
    Ok(edges)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Gauge>()?;
    m.add_class::<Timeline>()?;
    m.add_class::<NodeGraph>()?;
    Ok(())
}
