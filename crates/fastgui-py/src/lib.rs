mod backend;
mod charts;
mod cuda;
mod dialogs;
mod d3d11_import;
mod dmabuf;
mod image_file;
mod interaction;
mod metal_import;
mod plots;
mod tier4;
mod theme;
mod widgets;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use std::sync::atomic::AtomicU64;

use backend::{Command, CommandDispatch, EventWaker, RenderThreadHandles};
use fastgui_core::{command_channel, CommandReceiver, CpuFrame, FrameSlot, ImportStatus, ImportedLayer, PixelFormat, Readback};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyAny;

const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.06, 0.07, 0.09, 1.0];

static NEXT_VIEWPORT_ID: AtomicU64 = AtomicU64::new(1);
/// Also used for `Image` layers: both key the renderer's per-window layer/texture map.
pub(crate) fn next_viewport_id() -> u64 {
    NEXT_VIEWPORT_ID.fetch_add(1, Ordering::Relaxed)
}

/// A GPU-displayed image. CPU pixels go through `submit_frame`. Zero-copy paths are
/// `submit_metal` (macOS), `submit_dmabuf` (Linux), `submit_d3d11` (Windows), and
/// `create_cuda_surface` / `submit_cuda`.
#[pyclass]
pub(crate) struct Viewport {
    viewport_id: u64,
    frame_slot: FrameSlot<CpuFrame>,
    imported: FrameSlot<ImportedLayer>,
    /// What the render thread did with the last `submit_d3d11`.
    d3d11_status: Arc<ImportStatus>,
    // Set by `Window.set_viewport` / `set_content` as soon as this viewport is attached to a
    // window, so `submit_frame` can wake the idle event loop and the CUDA paths can reach the
    // right render thread.
    dispatch: Mutex<Option<CommandDispatch>>,
    fit: fastgui_core::widget::LayerFit,
    cuda: Mutex<cuda::CudaState>,
    /// The viewport's node in its window once shown, for live `set_file_drop` changes.
    id: widgets::IdCell,
    sender: widgets::SenderCell,
    interaction: interaction::Interaction,
}

#[pymethods]
impl Viewport {
    /// `fit` is `"stretch"` (fill the widget, historical) or `"contain"` (letterbox).
    #[new]
    #[pyo3(signature = (fit="stretch"))]
    fn new(fit: &str) -> PyResult<Self> {
        let fit = match fit.trim().to_ascii_lowercase().as_str() {
            "stretch" | "" => fastgui_core::widget::LayerFit::Stretch,
            "contain" | "fit" => fastgui_core::widget::LayerFit::Contain,
            _ => return Err(PyValueError::new_err("fit must be 'stretch' or 'contain'")),
        };
        Ok(Self {
            viewport_id: next_viewport_id(),
            frame_slot: FrameSlot::new(),
            imported: FrameSlot::new(),
            d3d11_status: Arc::default(),
            dispatch: Mutex::new(None),
            fit,
            cuda: Mutex::new(cuda::CudaState::default()),
            id: std::sync::Arc::new(Mutex::new(None)),
            sender: std::sync::Arc::new(Mutex::new(None)),
            interaction: interaction::Interaction::new(true),
        })
    }

    /// Submit a `(height, width, 3-or-4)` uint8 array (or any other object implementing the
    /// buffer protocol) as the viewport's next frame. Three-channel (RGB) input is expanded
    /// to RGBA with alpha=255. Safe to call from any thread, as often as you like — only the
    /// most recent unconsumed frame is ever displayed.
    ///
    /// Copies the buffer into an owned `Vec<u8>` today (`PyBuffer::to_vec`). Fine for demos and
    /// moderate rates; a camera/high-FPS path that is already packed RGBA will want a later
    /// zero-copy or fewer-copy upload when the array is C-contiguous RGBA8.
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

        // A 0-pixel frame becomes a 0-extent VkImage and a 0-byte allocation: validation
        // errors, then a crash mapping the null memory.
        if shape[0] == 0 || shape[1] == 0 {
            return Err(PyValueError::new_err("frame must be at least 1x1 pixels"));
        }
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        if width > fastgui_core::MAX_CPU_FRAME_EXTENT || height > fastgui_core::MAX_CPU_FRAME_EXTENT {
            return Err(PyValueError::new_err(format!(
                "frame edge must be <= {} pixels (got {}x{})",
                fastgui_core::MAX_CPU_FRAME_EXTENT,
                width,
                height,
            )));
        }
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

        self.push_cpu_frame(CpuFrame { width, height, format: PixelFormat::Rgba8, data: rgba });
        Ok(())
    }

    /// Sample an existing `MTLTexture` in place. `texture` is an integer pointer or any object
    /// with `mtl_ptr` (an mtlpy `Texture`). fastgui retains it. The texture must be 2D, on the
    /// system default Metal device, with `ShaderRead` (`readable=True`) and a float or unorm
    /// format such as `rgba8Unorm`.
    ///
    /// `wait_event` / `signal_event` are `MTLSharedEvent` pointers (or objects with `mtl_ptr`),
    /// or omitted. fastgui encodes the wait at the start of the command buffer that first draws
    /// this submission and the signal at the end of that same buffer, once. A submission that is
    /// replaced before it is drawn signals immediately so the producer is not left waiting.
    ///
    /// macOS only. Clears any pending CPU frame.
    #[pyo3(signature = (texture, *, wait_event=None, wait_value=0, signal_event=None, signal_value=0))]
    fn submit_metal(
        &self,
        texture: &Bound<'_, PyAny>,
        wait_event: Option<&Bound<'_, PyAny>>,
        wait_value: u64,
        signal_event: Option<&Bound<'_, PyAny>>,
        signal_value: u64,
    ) -> PyResult<()> {
        let layer = metal_import::prepare(texture, wait_event, wait_value, signal_event, signal_value)?;
        self.imported.submit(layer);
        self.frame_slot.clear();
        self.wake();
        Ok(())
    }

    /// Sample a linear RGBA8 dma-buf. `fd` is duplicated; the caller keeps and closes theirs.
    /// `stride` 0 means tightly packed (`width * 4`). `modifier` must be 0 (no DRM modifier).
    /// The producer must finish writing before this call. Linux only. Clears any pending CPU frame.
    #[pyo3(signature = (fd, width, height, *, stride=0, modifier=0))]
    fn submit_dmabuf(&self, fd: i32, width: u32, height: u32, stride: u32, modifier: u64) -> PyResult<()> {
        let layer = dmabuf::prepare(fd, width, height, stride, modifier)?;
        self.imported.submit(layer);
        self.frame_slot.clear();
        self.wake();
        Ok(())
    }

    /// Sample a shared Direct3D 11 texture in place (Windows, Vulkan). `handle` is an NT handle
    /// from `IDXGIResource1::CreateSharedHandle` on a 2D `R8G8B8A8_UNORM` or `B8G8R8A8_UNORM`
    /// texture created with `D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`. It is
    /// duplicated; the caller keeps and closes theirs. The D3D device must be on the window's GPU:
    /// create it on the adapter whose LUID is `fastgui.adapter_luid()`.
    ///
    /// fastgui reads the texture's real size and format from D3D, so `width` and `height` must match
    /// it, and `format` (`"rgba8"` or `"bgra8"`) is detected when left `None` and checked when given.
    /// A mismatch or a handle fastgui can't open is reported in `d3d11_status` and the previous frame
    /// stays.
    ///
    /// Sync: pass `fence`, an NT handle from `ID3D11Fence`/`ID3D12Fence` `CreateSharedHandle` (a
    /// shared fence), to wait for `wait_value` before sampling (up to 100 ms on the render thread;
    /// if it is late the submission is dropped and the previous frame stays) and to have
    /// `signal_value` signaled once the frame that first draws this submission has finished sampling.
    /// `0` skips that half. Without a fence the producer must have finished (flush, then wait on an
    /// event query) before the call. Keyed mutexes are not used.
    ///
    /// Create the handle once per texture and resubmit it each frame: a duplicate of a texture
    /// already imported (up to four per viewport) is not imported again. Clears any pending CPU frame.
    /// `d3d11_status` says what the render thread did with the last submission.
    #[pyo3(signature = (handle, width, height, *, format=None, fence=None, wait_value=0, signal_value=0))]
    fn submit_d3d11(
        &self,
        handle: isize,
        width: u32,
        height: u32,
        format: Option<&str>,
        fence: Option<isize>,
        wait_value: u64,
        signal_value: u64,
    ) -> PyResult<()> {
        let options = d3d11_import::Options { format, fence, wait_value, signal_value };
        let layer = d3d11_import::prepare(handle, width, height, options, &self.d3d11_status)?;
        self.imported.submit(layer);
        self.frame_slot.clear();
        self.wake();
        Ok(())
    }

    /// Submit a CUDA array as the viewport's next frame: anything with
    /// `__cuda_array_interface__` (torch, cupy, numba, jax), shaped `(height, width, 4)` uint8
    /// with contiguous pixels (rows may be padded). Host arrays go to `submit_frame`.
    ///
    /// `stream` is the CUDA stream the array was written on (an int handle, `torch.cuda.Stream`,
    /// cupy stream, or `__cuda_stream__` object); by default the one the array reports. fastgui
    /// copies after the work already queued on it, and that stream waits for the copy before
    /// running anything enqueued later, so reusing the array right away is safe. Nothing waits
    /// for the display.
    ///
    /// Where the window's GPU can share memory with CUDA (Windows or Linux, Vulkan, NVIDIA) this
    /// is one GPU-to-GPU copy. Elsewhere — macOS, a window on a non-NVIDIA GPU, before
    /// `window.run()` — it falls back to a copy through host memory; `cuda_status` says which.
    ///
    /// Safe to call from any thread; only the newest frame is shown. **Unverified on NVIDIA
    /// hardware.**
    #[pyo3(signature = (array, stream = None))]
    fn submit_cuda(&self, py: Python<'_>, array: &Bound<'_, PyAny>, stream: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let Some(image) = cuda::parse_cuda_array(array)? else {
            return self.submit_frame(array);
        };
        let stream = stream.filter(|s| !s.is_none()).map(cuda::parse_stream).transpose()?;
        let dispatch = self.dispatch.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let submitted = cuda::submit(
            py,
            &self.cuda,
            dispatch.as_ref(),
            self.viewport_id,
            image.on_stream(stream),
            array.clone().unbind(),
            || {
                self.frame_slot.clear();
                self.imported.clear();
            },
        )?;
        if let cuda::Submitted::Host(frame) = submitted {
            self.push_cpu_frame(frame);
        }
        Ok(())
    }

    /// Call `on_drop(paths, x, y)` when files are dropped on this viewport from the OS (`paths`
    /// is a list of strings; `x, y` the point in the viewport's coordinates). `None` stops it.
    #[pyo3(signature = (on_drop))]
    fn set_file_drop(&self, on_drop: Option<Py<PyAny>>) -> PyResult<()> {
        self.interaction.set_file_drop(on_drop, &self.id, &self.sender)
    }

    /// What the render thread did with the last `submit_d3d11`: `"ok: ..."`, `"unavailable: ..."`
    /// when this GPU can't import it, or a message saying what to fix. `"unused"` before any.
    #[getter]
    fn d3d11_status(&self) -> String {
        self.d3d11_status.get()
    }

    /// How the last `submit_cuda` reached the screen: `"interop"` (GPU-to-GPU), `"host copy:
    /// <reason>"`, or `"unused"`.
    #[getter]
    fn cuda_status(&self, py: Python<'_>) -> String {
        self.cuda.lock_py_attached(py).unwrap_or_else(|p| p.into_inner()).status()
    }

    /// Allocate a `width`x`height` RGBA8 surface CUDA writes into directly (no copy on the CUDA
    /// side) and return it as a `CudaSurface`; write frames with `with surface.frame(stream) as
    /// f:`. Must be called after `window.set_viewport(viewport)`, once `window.run()` has started,
    /// from a thread other than the one `run()` blocks on. A widget callback runs on that thread
    /// and raises `RuntimeError` instead of waiting. Before `run()` it raises `RuntimeError`
    /// after a ~2s grace period. Also raises where interop isn't available (macOS, or a device
    /// without the export extensions); `submit_cuda` works everywhere. Linux exports opaque fds.
    ///
    /// **Unverified on NVIDIA hardware.**
    fn create_cuda_surface(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<cuda::CudaSurface> {
        let dispatch = self.dispatch.lock().unwrap_or_else(|p| p.into_inner()).clone();
        cuda::create_surface(py, &self.cuda, dispatch, self.viewport_id, width, height, || {
            self.frame_slot.clear();
            self.imported.clear();
        })
    }
}

impl Viewport {
    pub(crate) fn bind_dispatch(&self, dispatch: CommandDispatch) {
        *self.dispatch.lock().unwrap_or_else(|p| p.into_inner()) = Some(dispatch);
    }

    pub(crate) fn describe_widget(&self) -> widgets::DescribedWidget {
        widgets::described_viewport(
            self.viewport_id,
            self.frame_slot.clone(),
            self.imported.clone(),
            self.fit,
            self.id.clone(),
            self.sender.clone(),
            self.interaction.spec(),
        )
    }

    fn push_cpu_frame(&self, frame: CpuFrame) {
        self.imported.clear();
        self.frame_slot.submit(frame);
        self.wake();
    }

    fn wake(&self) {
        if let Some(dispatch) = self.dispatch.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            dispatch.waker.wake();
        }
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
    #[pyo3(signature = (title="fastgui", width=1280, height=720))]
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
        // Floating panels follow the content: a `DockArea`'s handlers let them re-dock and
        // close, anything else clears them. Floaters whose handlers changed get their own
        // trees rebuilt so their title bars carry the new `on_drop`/`on_close`.
        rebind_floating_panels(self_, Some(widget))?;

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

    /// Make `theme` current (like `fg.set_theme`) and rebuild this window's content so it shows
    /// right away. Widgets keep their text, list selection and the like; scroll positions and
    /// keyboard focus reset, as for any `set_content`.
    fn set_theme(self_: &Bound<'_, Self>, theme: theme::Theme) -> PyResult<()> {
        theme::install(theme);
        let content = self_.borrow().content.lock().unwrap_or_else(|p| p.into_inner()).as_ref().map(|c| c.clone_ref(self_.py()));
        match content {
            Some(content) => Window::set_content(self_, content.bind(self_.py())),
            None => Ok(()),
        }
    }

    /// Open `popup` (a `Popup` or object with `as_popup()`, e.g. `Dialog`) in this window with
    /// its top-left at `(x, y)` (window coordinates, flipped near the far edges), or centered
    /// when `x`/`y` are omitted (dialogs).
    #[pyo3(signature = (popup, x=None, y=None))]
    fn show_popup(&self, popup: &Bound<'_, PyAny>, x: Option<f32>, y: Option<f32>) -> PyResult<()> {
        let anchor = match (x, y) {
            (Some(x), Some(y)) => fastgui_core::widget::PopupAnchor::Point(x, y),
            (None, None) => fastgui_core::widget::PopupAnchor::Center,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("pass both x and y, or neither")),
        };
        if let Ok(p) = popup.cast::<widgets::Popup>() {
            return p.borrow().open_in(&self.dispatch, anchor);
        }
        let as_popup = popup
            .call_method0("as_popup")
            .map_err(|_| PyValueError::new_err("show_popup expects a Popup or an object with as_popup()"))?;
        let p = as_popup
            .cast::<widgets::Popup>()
            .map_err(|_| PyValueError::new_err("as_popup() must return a Popup"))?;
        p.borrow().open_in(&self.dispatch, anchor)
    }

    /// Add `panel` as a real OS window (separate from this window's surface), initially placed
    /// at `(x, y)` relative to this window's inner origin with the given size. Draggable via its
    /// title bar anywhere on screen — including onto another monitor — and resizable via
    /// edge/corner drag. If this window's content is a `DockArea`, dropping the floater onto a
    /// docked region (or this window's outer edge) re-docks it. Survives later `set_content`
    /// calls until re-docked or closed. Order doesn't matter: a later `set_content(dock)`
    /// gives an already-floating panel that dock's re-dock and close handling.
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
        let content = self_
            .borrow()
            .content
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|c| c.clone_ref(self_.py()));
        bind_panel_to_dock_handlers(self_, &panel_ref, content.as_ref().map(|c| c.bind(self_.py())));
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

/// Point floating `panel`'s rearrange/close handlers at `content`'s when it's a `DockArea`
/// (callable `_on_rearrange`/`_on_close`). Otherwise clear rearrange, so a floater never
/// re-docks into a dock the window no longer shows, and fall back to `window`'s own
/// `_take_floating_panel` for close, so every floater stays closeable (its × and Alt+F4).
/// Returns whether either handler changed.
fn bind_panel_to_dock_handlers(
    window: &Bound<'_, Window>,
    panel: &Bound<'_, widgets::Panel>,
    content: Option<&Bound<'_, PyAny>>,
) -> bool {
    let callable = |obj: Option<&Bound<'_, PyAny>>, name: &str| {
        obj.and_then(|o| o.getattr(name).ok())
            .filter(|h| h.is_callable())
            .map(Bound::unbind)
    };
    let rearrange = callable(content, "_on_rearrange");
    let close = callable(content, "_on_close").or_else(|| callable(Some(window.as_any()), "_take_floating_panel"));
    panel.borrow().replace_dock_handlers(panel.py(), rearrange, close)
}

/// Rebind every floating panel to `content` (see `bind_panel_to_dock_handlers`), and rebuild
/// the OS-window tree of each one whose handlers changed. Floater title bars are built once,
/// at `add_floating_panel`, so without the rebuild a floater added before
/// `set_content(dock)` would never get that dock's `on_drop`/`on_close`.
fn rebind_floating_panels(window: &Bound<'_, Window>, content: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let py = window.py();
    let panels: Vec<Py<PyAny>> = {
        let borrowed = window.borrow();
        let floating = borrowed.floating_panels.lock().unwrap_or_else(|p| p.into_inner());
        floating.iter().map(|(panel, ..)| panel.clone_ref(py)).collect()
    };
    for panel in panels {
        let Ok(bound) = panel.bind(py).cast::<widgets::Panel>() else { continue };
        if !bind_panel_to_dock_handlers(window, bound, content) {
            continue;
        }
        let region_id = bound.borrow().region_id();
        let described = bound.borrow().describe_window_content()?;
        let dispatch = window.borrow().dispatch.for_floating(region_id);
        let closure_dispatch = dispatch.clone();
        dispatch
            .send(Command::MutateFloatingTree {
                region_id,
                mutation: Box::new(move |tree| {
                    tree.reset();
                    let root = tree.root();
                    widgets::attach(tree, root, described, &closure_dispatch);
                }),
            })
            .map_err(|_| PyRuntimeError::new_err("window has already closed"))?;
    }
    Ok(())
}

/// The adapter LUID (`low | high << 32`) of the GPU windows render on, so a Direct3D device can be
/// created on the same adapter for `submit_d3d11`. `None` off Windows or when it can't be found.
#[pyfunction]
fn adapter_luid() -> Option<u64> {
    d3d11_import::adapter_luid()
}

#[pymodule]
fn _fastgui(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adapter_luid, m)?)?;
    m.add_class::<Window>()?;
    m.add_class::<Viewport>()?;
    m.add_class::<cuda::CudaSurface>()?;
    m.add_class::<cuda::CudaFrame>()?;
    widgets::register(m)?;
    plots::register(m)?;
    charts::register(m)?;
    tier4::register(m)?;
    theme::register(m)?;
    dialogs::register(m)?;
    Ok(())
}
