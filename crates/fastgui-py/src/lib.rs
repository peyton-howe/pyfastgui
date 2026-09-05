mod widgets;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use std::sync::atomic::AtomicU64;

use fastgui_core::{
    command_channel, oneshot_channel, CommandReceiver, CpuFrame, FrameSlot, PixelFormat, Readback,
};
use fastgui_interop_cuda::{CudaContext, CudaStream, ExternalMemory, ExternalSemaphore};
use fastgui_render_vk::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.06, 0.07, 0.09, 1.0];

static NEXT_VIEWPORT_ID: AtomicU64 = AtomicU64::new(1);
fn next_viewport_id() -> u64 {
    NEXT_VIEWPORT_ID.fetch_add(1, Ordering::Relaxed)
}

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
}

impl Viewport {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe_widget(&self) -> widgets::DescribedWidget {
        widgets::described_viewport(self.viewport_id, self.frame_slot.clone())
    }
}

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
    /// Every `Panel` ever passed to `add_floating_panel`, with its current `(x, y, width,
    /// height)` — re-described and re-attached on every `set_content` (which otherwise resets
    /// the whole tree and would silently drop them, e.g. every time a `DockArea` rearrange
    /// calls `set_content` again internally).
    floating_panels: Mutex<Vec<(Py<PyAny>, f32, f32, f32, f32)>>,
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
            dispatch: CommandDispatch { sender, waker },
            receiver: Mutex::new(Some(receiver)),
            clear_color: Readback::new(DEFAULT_CLEAR_COLOR),
            floating_panels: Mutex::new(Vec::new()),
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
        let mut described = widgets::describe(widget)?;
        described.force_fill();

        // Re-describe every floating panel too — `tree.reset()` below wipes the whole tree, and
        // this is the only place that happens, so it's the only place that can put them back.
        // Needed for drag-to-rearrange: a `DockArea` rearrange calls this method again on its
        // own (see `Window`'s doc comment on `floating_panels`), and floating panels must
        // survive that exactly like they'd survive any other `set_content` call.
        let floating_described = {
            let window = self_.borrow();
            let floating = window.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
            floating
                .iter()
                .map(|(panel, x, y, width, height)| {
                    Python::attach(|py| {
                        let panel = panel.bind(py).cast::<widgets::Panel>().map_err(|_| {
                            PyRuntimeError::new_err("internal error: a non-Panel ended up in floating_panels")
                        })?;
                        panel.borrow().describe_floating(*x, *y, *width, *height)
                    })
                })
                .collect::<PyResult<Vec<_>>>()?
        };

        let dispatch = self_.borrow().dispatch.clone();
        let closure_dispatch = dispatch.clone();
        dispatch
            .send(Command::MutateWidgetTree(Box::new(move |tree| {
                tree.reset();
                let root = tree.root();
                widgets::attach(tree, root, described, &closure_dispatch);
                for floating in floating_described {
                    widgets::attach(tree, root, floating, &closure_dispatch);
                }
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

    /// Add `panel` as an always-on-top floating region — an approximation of a real floating
    /// window (a second native OS window per floating panel) simulated *within* this single
    /// window instead, at `(x, y, width, height)` in the same physical-pixel coordinates as
    /// everything else. Chosen deliberately over real multi-window support: that would need
    /// reworking `fastgui-render-vk`'s one-`Window`/one-swapchain-per-`App` assumption, a much
    /// bigger and riskier change than this session had appetite for (see ROADMAP.md's M6
    /// status) — the real trade-off here is losing "drag onto a second monitor" and "separate
    /// taskbar entry", not anything about how it renders or drags.
    ///
    /// Draggable via its title bar (moves the panel directly; unlike a docked `Panel`'s title
    /// bar, this never goes through drop-zone/rearrange logic — see `WidgetKind::PanelTitleBar`'s
    /// doc comment). Not resizable, and not re-dockable back into a `DockArea`, in this pass.
    /// Persists across future `set_content` calls (including ones a `DockArea` rearrange
    /// triggers internally) — see `Window`'s doc comment on `floating_panels`.
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
        widgets::bind_dispatch(panel, &self_.borrow().dispatch);
        let described = panel_ref.borrow().describe_floating(x, y, width, height)?;

        {
            let window = self_.borrow();
            let mut floating = window.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
            floating.push((panel.clone().unbind(), x, y, width, height));
        }

        let dispatch = self_.borrow().dispatch.clone();
        let closure_dispatch = dispatch.clone();
        dispatch
            .send(Command::MutateWidgetTree(Box::new(move |tree| {
                let root = tree.root();
                widgets::attach(tree, root, described, &closure_dispatch);
            })))
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))
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
            fastgui_render_vk::run(&title, width, height, initial_color, handles)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
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
