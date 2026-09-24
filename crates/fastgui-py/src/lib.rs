mod backend;
mod widgets;

use std::sync::atomic::Ordering;
use std::sync::Mutex;
#[cfg(not(target_os = "macos"))]
use std::sync::Arc;

use std::sync::atomic::AtomicU64;

use backend::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
use fastgui_core::{command_channel, CommandReceiver, CpuFrame, FrameSlot, PixelFormat, Readback};
#[cfg(not(target_os = "macos"))]
use fastgui_core::oneshot_channel;
#[cfg(not(target_os = "macos"))]
use fastgui_interop_cuda::{CudaContext, CudaStream, ExternalMemory, ExternalSemaphore};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.06, 0.07, 0.09, 1.0];

static NEXT_VIEWPORT_ID: AtomicU64 = AtomicU64::new(1);
fn next_viewport_id() -> u64 {
    NEXT_VIEWPORT_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(not(target_os = "macos"))]
fn cuda_err(err: fastgui_interop_cuda::CudaError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// A GPU-displayed image, fed either CPU-side pixel data (`submit_frame`) or, once attached to
/// a `Window` via `set_viewport`, a CUDA-writable surface (`create_cuda_surface`).
#[pyclass]
pub(crate) struct Viewport {
    viewport_id: u64,
    frame_slot: FrameSlot<CpuFrame>,
    // Set by `Window.set_viewport` / `set_content` as soon as this viewport is attached to a
    // window, so `submit_frame` can wake the idle event loop and `create_cuda_surface` can
    // reach the right render thread.
    dispatch: Mutex<Option<CommandDispatch>>,
}

#[pymethods]
impl Viewport {
    #[new]
    fn new() -> Self {
        Self {
            viewport_id: next_viewport_id(),
            frame_slot: FrameSlot::new(),
            dispatch: Mutex::new(None),
        }
    }

    /// Submit a `(height, width, 3-or-4)` uint8 array (or any other object implementing the
    /// buffer protocol) as the viewport's next frame. Three-channel (RGB) input is expanded
    /// to RGBA with alpha=255. Safe to call from any thread, as often as you like — only the
    /// most recent unconsumed frame is ever displayed.
    fn submit_frame(&self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let buffer = PyBuffer::<u8>::get(data)?;
        let shape = buffer.shape();
        if shape.len() != 3 || !(shape[2] == 3 || shape[2] == 4) {
            return Err(PyValueError::new_err(
                "expected a (height, width, 3-or-4) uint8 array",
            ));
        }
        if !buffer.is_c_contiguous() {
            return Err(PyValueError::new_err("frame buffer must be C-contiguous"));
        }

        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let channels = shape[2];
        let raw = buffer.to_vec(data.py())?;

        let rgba = if channels == 4 {
            raw
        } else {
            let mut out = Vec::with_capacity(raw.len() / 3 * 4);
            for pixel in raw.chunks_exact(3) {
                out.extend_from_slice(pixel);
                out.push(255);
            }
            out
        };

        self.frame_slot.submit(CpuFrame { width, height, format: PixelFormat::Rgba8, data: rgba });
        if let Some(dispatch) = self.dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            dispatch.waker.wake();
        }
        Ok(())
    }

    /// Allocate a `width`x`height` RGBA8 texture that a CUDA kernel can write into directly
    /// (zero-copy — the same GPU memory Vulkan samples from), and return a `CudaSurface`
    /// exposing its raw device pointer. Must be called after `window.set_viewport(viewport)`.
    ///
    /// **Unverified**: written against the CUDA driver API and Vulkan external-memory specs,
    /// but never run against a real CUDA-capable GPU during development. See
    /// `fastgui_interop_cuda`'s crate docs (in the Rust source) for the full caveat.
    ///
    /// Not available on macOS — see the `#[cfg(target_os = "macos")]` override below.
    #[cfg(not(target_os = "macos"))]
    fn create_cuda_surface(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<CudaSurface> {
        let dispatch = self
            .dispatch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "call window.set_viewport(viewport) or window.set_content(...) before viewport.create_cuda_surface()",
                )
            })?;

        let (respond, response) = oneshot_channel();
        dispatch
            .send(Command::CreateCudaSurface {
                viewport_id: self.viewport_id,
                width,
                height,
                respond,
            })
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))?;

        let handles = py
            .detach(|| response.recv())
            .map_err(|_| PyRuntimeError::new_err("window closed before creating the CUDA surface"))?
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        py.detach(|| {
            let ctx = CudaContext::new().map_err(cuda_err)?;
            let memory = ExternalMemory::import_win32(&ctx, handles.memory_win32_handle, handles.memory_size)
                .map_err(cuda_err)?;
            let device_ptr = memory.device_ptr();
            let semaphore = ExternalSemaphore::import_win32_timeline(&ctx, handles.semaphore_win32_handle)
                .map_err(cuda_err)?;
            let stream = CudaStream::new(&ctx).map_err(cuda_err)?;

            Ok(CudaSurface {
                _context: ctx,
                _memory: memory,
                inner: Mutex::new(CudaSurfaceInner { semaphore, stream, next_value: 0 }),
                device_ptr,
                pitch: handles.row_pitch,
                width: handles.width,
                height: handles.height,
                target_value: handles.target_value,
            })
        })
    }

    /// Apple hasn't shipped an NVIDIA GPU since ~2019, so there is no zero-copy CUDA<->Metal
    /// surface to allocate here — see `fastgui_render_mtl::Command`'s doc comment for the same
    /// reasoning one layer down. Use `Viewport.submit_frame()` (CPU copy) instead.
    #[cfg(target_os = "macos")]
    fn create_cuda_surface(&self, _py: Python<'_>, _width: u32, _height: u32) -> PyResult<CudaSurface> {
        Err(PyRuntimeError::new_err(
            "CUDA interop is not supported on macOS -- Apple has not shipped an NVIDIA GPU \
             since ~2019, so there is no zero-copy CUDA<->Metal path. Use \
             Viewport.submit_frame() (CPU copy) instead.",
        ))
    }
}

impl Viewport {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe_widget(&self) -> widgets::DescribedWidget {
        widgets::described_viewport(self.viewport_id, self.frame_slot.clone())
    }
}

#[cfg(not(target_os = "macos"))]
struct CudaSurfaceInner {
    semaphore: ExternalSemaphore,
    stream: CudaStream,
    next_value: u64,
}

/// A CUDA-writable GPU surface backing a `Viewport`, returned by `Viewport.create_cuda_surface`.
/// Write into `device_ptr` (e.g. via cupy, numba, a raw ctypes CUDA kernel launch — anything
/// that accepts a raw CUDA device pointer) respecting `pitch`-byte rows, then call
/// `signal_ready()` once a frame is complete.
///
/// **Unverified** — see `Viewport.create_cuda_surface`.
///
/// Not available on macOS — see the stub definition below.
#[cfg(not(target_os = "macos"))]
#[pyclass]
struct CudaSurface {
    _context: CudaContext,
    _memory: ExternalMemory,
    inner: Mutex<CudaSurfaceInner>,
    device_ptr: u64,
    pitch: u64,
    width: u32,
    height: u32,
    target_value: Arc<std::sync::atomic::AtomicU64>,
}

#[cfg(not(target_os = "macos"))]
#[pymethods]
impl CudaSurface {
    #[getter]
    fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    #[getter]
    fn pitch(&self) -> u64 {
        self.pitch
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    /// Tell the render thread a frame is ready: signals the shared timeline semaphore (waiting
    /// for CUDA to confirm it actually completed) and only then publishes the new value for
    /// the render thread to wait on before its next sample. Safe to call from any thread, but
    /// calls serialize against each other (they share one CUDA stream).
    fn signal_ready(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            inner.next_value += 1;
            let value = inner.next_value;
            inner.semaphore.signal(&inner.stream, value).map_err(cuda_err)?;
            inner.stream.synchronize().map_err(cuda_err)?;
            self.target_value.store(value, Ordering::Release);
            Ok(())
        })
    }
}

/// Never constructed — `Viewport::create_cuda_surface` always errors before reaching one on
/// macOS (see its `#[cfg(target_os = "macos")]` override). Exists only so this type name and
/// `_fastgui`'s `m.add_class::<CudaSurface>()` registration stay the same across platforms.
#[cfg(target_os = "macos")]
#[pyclass]
struct CudaSurface;

#[cfg(target_os = "macos")]
#[pymethods]
impl CudaSurface {}

/// A top-level application window. `run()` opens it and blocks the calling thread, rendering
/// until the user closes it.
///
/// Every mutator (`set_clear_color`, `set_viewport`) is safe to call from any Python thread,
/// at any time, including before `run()` starts or after the window has closed: calls are
/// enqueued and applied by the render thread once per frame, never touching render state
/// directly. Getters (`clear_color`) read a cache the render thread publishes after applying
/// each command, so they never block on frame timing.
#[pyclass]
struct Window {
    title: String,
    width: u32,
    height: u32,
    dispatch: CommandDispatch,
    receiver: Mutex<Option<CommandReceiver<Command>>>,
    clear_color: Readback<[f32; 4]>,
    /// Every `Panel` ever passed to `add_floating_panel`, with its initial `(x, y, width,
    /// height)`. Each lives in its own OS window (not in the main widget tree). Removed by
    /// `_take_floating_panel` when dropped back into a `DockArea`.
    floating_panels: Mutex<Vec<(Py<PyAny>, f32, f32, f32, f32)>>,
    /// The widget last passed to `set_content`. Used so `add_floating_panel` (and a later
    /// `set_content` that replaces the tree) can bind each floating panel's rearrange handler
    /// to a `DockArea._on_rearrange` when that's the window content.
    content: Mutex<Option<Py<PyAny>>>,
}

#[pymethods]
impl Window {
    #[new]
    #[pyo3(signature = (title="fast-gui", width=1280, height=720))]
    fn new(title: &str, width: u32, height: u32) -> Self {
        let (sender, receiver) = command_channel();
        let waker = EventWaker::default();
        Self {
            title: title.to_owned(),
            width,
            height,
            dispatch: CommandDispatch { sender, waker, floating_region: None },
            receiver: Mutex::new(Some(receiver)),
            clear_color: Readback::new(DEFAULT_CLEAR_COLOR),
            floating_panels: Mutex::new(Vec::new()),
            content: Mutex::new(None),
        }
    }

    /// Queue a clear-color change. Returns immediately; the render thread applies it before
    /// the next frame. Safe to call from any thread.
    fn set_clear_color(&self, r: f32, g: f32, b: f32, a: f32) -> PyResult<()> {
        self.dispatch
            .send(Command::SetClearColor([r, g, b, a]))
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))
    }

    /// Display `viewport` as the window's full-window content. Equivalent to
    /// `set_content(viewport)` — a `Viewport` is a widget and can also live inside a `Panel`.
    fn set_viewport(self_: &Bound<'_, Self>, viewport: &Bound<'_, Viewport>) -> PyResult<()> {
        Window::set_content(self_, &viewport.clone().into_any())
    }

    /// The last clear color the render thread actually applied.
    #[getter]
    fn clear_color(&self) -> [f32; 4] {
        self.clear_color.get()
    }

    /// Replace the window's content with `widget` (a `Box`, `Label`, `Button`, or `Slider`),
    /// full-window, behind (and once it has content, over) the clear color. Returns
    /// immediately — like every other mutator, this just enqueues the change; the render
    /// thread builds it before the next frame. Safe to call from any thread, including
    /// (in fact, typically) before `run()` starts.
    ///
    /// Note this means a widget's `id_cell` isn't populated until the render thread actually
    /// gets to it, so calling e.g. `child.set_text(...)` immediately afterward *on the same
    /// thread, before `run()` starts* will raise ("not attached yet") rather than block —
    /// there's nothing running yet to attach it. Calling such mutators from inside a widget's
    /// own callback (which only ever fires once the window is already running) is always fine.
    fn set_content(self_: &Bound<'_, Self>, widget: &Bound<'_, PyAny>) -> PyResult<()> {
        widgets::bind_dispatch(widget, &self_.borrow().dispatch);
        {
            let window = self_.borrow();
            *window.content.lock().unwrap_or_else(|p| p.into_inner()) = Some(widget.clone().unbind());
        }
        // If this content is a `DockArea`, floating panels need its rearrange handler so they
        // can be dropped back in. Bind *before* re-describing them so the attached title bars
        // carry `on_drop`.
        bind_floating_panels_to_content(self_, widget);

        let mut described = widgets::describe(widget)?;
        described.force_fill();

        // Floating panels are real OS windows (see `add_floating_panel`), not overlays in this
        // tree — a rearrange-triggered `set_content` must rebuild only the main dock content.
        let dispatch = self_.borrow().dispatch.clone();
        let closure_dispatch = dispatch.clone();
        dispatch
            .send(Command::MutateWidgetTree(Box::new(move |tree| {
                tree.reset();
                let root = tree.root();
                widgets::attach(tree, root, described, &closure_dispatch);
            })))
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))?;

        // Back-reference so composite widgets (e.g. `DockArea`) can trigger a full rebuild
        // later on their own — needed for drag-to-rearrange: Python restructures its tree in
        // response to a drop, then must ask *this* window to re-attach it, and nothing else in
        // this architecture gives it a route back to do that. Silently no-ops for anything that
        // doesn't support arbitrary attributes (e.g. a bare pyclass widget passed directly).
        let _ = widget.setattr("_fastgui_window", self_);
        Ok(())
    }

    /// Add `panel` as a real OS window (separate from this window's surface), initially placed
    /// at `(x, y)` relative to this window's inner origin with the given size. Draggable via its
    /// title bar anywhere on screen — including onto another monitor — and resizable via
    /// edge/corner drag. If this window's content is a `DockArea`, dropping the floater onto a
    /// docked region (or this window's outer edge) re-docks it. A docked panel can't be dragged
    /// *out* into a new floater. Survives later `set_content` calls until re-docked.
    fn add_floating_panel(
        self_: &Bound<'_, Self>,
        panel: &Bound<'_, PyAny>,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> PyResult<()> {
        let panel_ref = panel
            .cast::<widgets::Panel>()
            .map_err(|_| PyValueError::new_err("add_floating_panel expects a Panel"))?;
        panel_ref.borrow().set_floating();
        let region_id = panel_ref.borrow().region_id();
        let title = panel_ref.borrow().title();
        let dispatch = self_.borrow().dispatch.for_floating(region_id);
        widgets::bind_dispatch(panel, &dispatch);
        if let Some(content) = self_
            .borrow()
            .content
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|c| c.clone_ref(self_.py()))
        {
            bind_panel_to_dock_rearrange(&panel_ref, content.bind(self_.py()));
        }
        let mut described = panel_ref.borrow().describe_window_content()?;
        described.force_fill();

        {
            let window = self_.borrow();
            let mut floating = window.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
            floating.push((panel.clone().unbind(), x, y, width, height));
        }

        let closure_dispatch = dispatch.clone();
        dispatch
            .send(Command::AddFloatingPanel {
                region_id,
                title,
                x,
                y,
                width,
                height,
                build: Box::new(move |tree| {
                    let root = tree.root();
                    widgets::attach(tree, root, described, &closure_dispatch);
                }),
            })
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))
    }

    /// Look up a still-floating panel by its stable region id without removing it. Used by
    /// `DockArea._on_rearrange` to decide whether a dragged id that's missing from the dock tree
    /// is a re-dockable floater, *before* committing the tree mutation.
    #[pyo3(name = "_peek_floating_panel")]
    fn peek_floating_panel(self_: &Bound<'_, Self>, region_id: u64) -> Option<Py<PyAny>> {
        let py = self_.py();
        let window = self_.borrow();
        let floating = window.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
        floating.iter().find_map(|(panel, ..)| {
            let bound = panel.bind(py).cast::<widgets::Panel>().ok()?;
            (bound.borrow().region_id() == region_id).then(|| panel.clone_ref(py))
        })
    }

    /// Remove a floating panel from this window's bookkeeping, clear its floating flag, and
    /// close its OS window. Called by `DockArea._on_rearrange` once a re-dock drop has a valid
    /// target.
    #[pyo3(name = "_take_floating_panel")]
    fn take_floating_panel(self_: &Bound<'_, Self>, region_id: u64) -> Option<Py<PyAny>> {
        let py = self_.py();
        let window = self_.borrow();
        let mut floating = window.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
        let index = floating.iter().position(|(panel, ..)| {
            panel
                .bind(py)
                .cast::<widgets::Panel>()
                .ok()
                .is_some_and(|bound| bound.borrow().region_id() == region_id)
        })?;
        let (panel, ..) = floating.remove(index);
        if let Ok(bound) = panel.bind(py).cast::<widgets::Panel>() {
            bound.borrow().clear_floating();
        }
        let _ = window.dispatch.send(Command::RemoveFloatingPanel { region_id });
        Some(panel)
    }

    /// Open the window and block the calling thread, rendering until it is closed. Can only
    /// be called once per `Window`.
    fn run(&self, py: Python<'_>) -> PyResult<()> {
        let receiver = self
            .receiver
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Window.run() can only be called once"))?;

        let title = self.title.clone();
        let (width, height) = (self.width, self.height);
        let clear_color_readback = self.clear_color.clone();
        let initial_color = clear_color_readback.get();

        py.detach(|| {
            let handles = RenderThreadHandles {
                commands: receiver,
                clear_color: clear_color_readback,
                waker: self.dispatch.waker.clone(),
            };
            backend::run(&title, width, height, initial_color, handles)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
    }
}

/// If `content` is a `DockArea` (exposes `_on_rearrange`), bind that handler onto `panel` so
/// dragging the floating title bar can re-dock. Silently no-ops for any other content widget.
fn bind_panel_to_dock_rearrange(panel: &Bound<'_, widgets::Panel>, content: &Bound<'_, PyAny>) {
    if let Ok(handler) = content.getattr("_on_rearrange") {
        if handler.is_callable() {
            panel.borrow().set_rearrange_handler(handler.unbind());
        }
    }
}

fn bind_floating_panels_to_content(window: &Bound<'_, Window>, content: &Bound<'_, PyAny>) {
    let py = window.py();
    let panels: Vec<Py<PyAny>> = {
        let borrowed = window.borrow();
        let floating = borrowed.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
        floating.iter().map(|(panel, ..)| panel.clone_ref(py)).collect()
    };
    for panel in panels {
        if let Ok(bound) = panel.bind(py).cast::<widgets::Panel>() {
            bind_panel_to_dock_rearrange(&bound, content);
        }
    }
}

#[pymodule]
fn _fastgui(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Window>()?;
    m.add_class::<Viewport>()?;
    m.add_class::<CudaSurface>()?;
    widgets::register(m)?;
    Ok(())
}
