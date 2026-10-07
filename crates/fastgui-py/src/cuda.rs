//! CUDA arrays into a `Viewport`: `Viewport.submit_cuda`, `Viewport.create_cuda_surface`,
//! `CudaSurface` and `CudaFrame`.
//!
//! Two paths, picked per viewport:
//!
//! - **Interop** (Windows/Vulkan with an NVIDIA GPU driving the window): the render thread
//!   exports a CUDA layer (`fastgui_app::cuda_handles`) and this module imports it. A frame is a
//!   GPU-to-GPU copy into a free slot on fastgui's own CUDA stream, ordered after the caller's
//!   stream with events in both directions, then published from a host callback once it is done.
//!   The caller's stream never waits on the display, only (briefly) on the copy itself.
//! - **Host copy** (everywhere else: macOS, Linux until fd export lands, a window on a non-NVIDIA
//!   GPU, before `run()`): copy device→host on a stream ordered after the caller's, then the same
//!   path as `submit_frame`.
//!
//! **Unverified on NVIDIA hardware** — see `fastgui_interop_cuda`'s crate docs.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use fastgui_app::{AcquireError, AcquiredSlot, CudaExportHandles, CudaLayerShared};
use fastgui_core::{oneshot_channel, CpuFrame, PixelFormat};
use fastgui_interop_cuda::{
    device_by_uuid, pointer_device, CudaError, ExternalMemory, ExternalSemaphore, PrimaryContext, RawStream,
    Stream,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyAny, PyDict};

use crate::backend::{Command, CommandDispatch, EventWaker};

/// More link creations than this within `CHURN_WINDOW` and `submit_cuda` host-copies instead.
const CHURN_LIMIT: usize = 3;
const CHURN_WINDOW: Duration = Duration::from_secs(2);

/// How long a producer waits for a free slot when it is ahead of its own GPU work by more than
/// the slack the slots give it.
const ACQUIRE_TIMEOUT: Duration = Duration::from_secs(5);

fn cuda_err(err: CudaError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// ---------------------------------------------------------------------------------------------
// Input parsing
// ---------------------------------------------------------------------------------------------

/// A `(height, width, 4)` uint8 CUDA array whose pixels are contiguous (rows may be padded).
pub(crate) struct DeviceImage {
    ptr: u64,
    width: u32,
    height: u32,
    row_stride: usize,
    /// Stream the array's producer wrote it on; `None` when no synchronization is needed.
    stream: Option<RawStream>,
}

impl DeviceImage {
    /// Use `stream` (when given) instead of the stream the array reported.
    pub(crate) fn on_stream(mut self, stream: Option<RawStream>) -> Self {
        if stream.is_some() {
            self.stream = stream;
        }
        self
    }

    fn row_bytes(&self) -> usize {
        self.width as usize * 4
    }
}

const SHAPE_HELP: &str = "expected a (height, width, 4) uint8 CUDA array with contiguous RGBA \
     pixels (e.g. torch: t.permute(1, 2, 0).contiguous(); add an alpha channel to RGB data)";

/// `obj.__cuda_array_interface__` as a [`DeviceImage`], or `None` when `obj` has no such
/// attribute (a host array, say).
pub(crate) fn parse_cuda_array(obj: &Bound<'_, PyAny>) -> PyResult<Option<DeviceImage>> {
    let Ok(cai) = obj.getattr("__cuda_array_interface__") else { return Ok(None) };
    let cai = cai
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("__cuda_array_interface__ must be a dict"))?;
    let get = |key: &str| cai.get_item(key);

    let shape: Vec<usize> = get("shape")?
        .ok_or_else(|| PyValueError::new_err("__cuda_array_interface__ has no 'shape'"))?
        .extract()?;
    let typestr: String = get("typestr")?
        .ok_or_else(|| PyValueError::new_err("__cuda_array_interface__ has no 'typestr'"))?
        .extract()?;
    if shape.len() != 3 || shape[2] != 4 || !matches!(typestr.as_str(), "|u1" | "<u1" | ">u1" | "=u1") {
        return Err(PyValueError::new_err(format!("{SHAPE_HELP}; got shape {shape:?}, dtype {typestr}")));
    }
    if let Some(mask) = get("mask")? {
        if !mask.is_none() {
            return Err(PyValueError::new_err("masked CUDA arrays are not supported"));
        }
    }
    let (height, width) = (shape[0], shape[1]);
    if height == 0 || width == 0 {
        return Err(PyValueError::new_err("frame must be at least 1x1 pixels"));
    }
    let max = fastgui_core::MAX_CPU_FRAME_EXTENT as usize;
    if width > max || height > max {
        return Err(PyValueError::new_err(format!(
            "frame edge must be <= {max} pixels (got {width}x{height})"
        )));
    }
    let (ptr, _readonly): (u64, bool) = get("data")?
        .ok_or_else(|| PyValueError::new_err("__cuda_array_interface__ has no 'data'"))?
        .extract()?;
    let row_stride = match get("strides")? {
        Some(strides) if !strides.is_none() => {
            let strides: Vec<isize> = strides.extract()?;
            let row_bytes = (width * 4) as isize;
            if strides.len() != 3 || strides[2] != 1 || strides[1] != 4 || strides[0] < row_bytes {
                return Err(PyValueError::new_err(format!("{SHAPE_HELP}; got strides {strides:?}")));
            }
            strides[0] as usize
        }
        _ => width * 4,
    };
    let stream = cai_stream(cai.as_any())?;
    Ok(Some(DeviceImage { ptr, width: width as u32, height: height as u32, row_stride, stream }))
}

/// The producer stream a `__cuda_array_interface__` dict declares. v3: `stream` is the stream
/// the data was written on (`None`: no sync needed). Older producers don't say; assume the legacy
/// default stream, which orders after everything that isn't non-blocking.
pub(crate) fn cai_stream(cai: &Bound<'_, PyAny>) -> PyResult<Option<RawStream>> {
    let stream = match cai.get_item("stream") {
        Ok(stream) => stream,
        Err(err) if err.is_instance_of::<PyKeyError>(cai.py()) => return Ok(Some(RawStream(1))),
        Err(err) => return Err(err),
    };
    if stream.is_none() {
        Ok(None)
    } else {
        Ok(Some(RawStream(stream.extract()?)))
    }
}

/// A stream argument: an int handle, a `torch.cuda.Stream` (`.cuda_stream`), a `cupy` stream
/// (`.ptr`), or anything with `__cuda_stream__()` (the CUDA stream protocol).
pub(crate) fn parse_stream(obj: &Bound<'_, PyAny>) -> PyResult<RawStream> {
    if let Ok(handle) = obj.extract::<usize>() {
        return Ok(RawStream(handle));
    }
    if let Ok(protocol) = obj.call_method0("__cuda_stream__") {
        let (_version, handle): (usize, usize) = protocol.extract()?;
        return Ok(RawStream(handle));
    }
    for attr in ["cuda_stream", "ptr", "handle"] {
        if let Ok(handle) = obj.getattr(attr).and_then(|h| h.extract::<usize>()) {
            return Ok(RawStream(handle));
        }
    }
    Err(PyTypeError::new_err(
        "stream must be an int CUstream handle, a torch.cuda.Stream, a cupy stream, or an object \
         with __cuda_stream__()",
    ))
}

// ---------------------------------------------------------------------------------------------
// Interop link
// ---------------------------------------------------------------------------------------------

/// The CUDA side of one render-thread CUDA layer.
pub(crate) struct CudaLink {
    // Field order is drop order: the stream synchronizes on drop, so its pending copies and
    // signals finish before the imported memory and semaphores go away.
    stream: Stream,
    ready: ExternalSemaphore,
    release: ExternalSemaphore,
    memory: ExternalMemory,
    shared: Arc<CudaLayerShared>,
    width: u32,
    height: u32,
    slot_stride: u64,
    /// Last `ready` value enqueued. Locked across "bump and enqueue the signal" so values reach
    /// the stream in increasing order.
    next_ready: Mutex<u64>,
    waker: EventWaker,
}

enum LinkError {
    /// The render thread dropped the layer; make a new one.
    Closed,
    Failed(PyErr),
}

impl From<CudaError> for LinkError {
    fn from(err: CudaError) -> Self {
        Self::Failed(cuda_err(err))
    }
}

impl CudaLink {
    fn import(handles: CudaExportHandles, waker: EventWaker) -> Result<Self, String> {
        let ordinal = device_by_uuid(&handles.device_uuid)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| "the window is drawn by a GPU CUDA doesn't drive (not the NVIDIA one)".to_owned())?;
        let ctx = PrimaryContext::new(ordinal).map_err(|e| e.to_string())?;
        let memory = ExternalMemory::import_win32(
            &ctx,
            handles.memory_win32_handle,
            handles.memory_size,
            handles.memory_dedicated,
        )
        .map_err(|e| e.to_string())?;
        let ready = ExternalSemaphore::import_win32_timeline(&ctx, handles.ready_win32_handle)
            .map_err(|e| e.to_string())?;
        let release = ExternalSemaphore::import_win32_timeline(&ctx, handles.release_win32_handle)
            .map_err(|e| e.to_string())?;
        let stream = Stream::new(&ctx).map_err(|e| e.to_string())?;
        Ok(Self {
            stream,
            ready,
            release,
            memory,
            shared: handles.shared.clone(),
            width: handles.width,
            height: handles.height,
            slot_stride: handles.slot_stride,
            next_ready: Mutex::new(0),
            waker,
        })
        // `handles` drops here, closing the exported NT handles: the imports hold their own refs.
    }

    fn slot_ptr(&self, index: usize) -> u64 {
        self.memory.device_ptr() + index as u64 * self.slot_stride
    }

    fn acquire(&self) -> Result<AcquiredSlot, LinkError> {
        self.shared.acquire(ACQUIRE_TIMEOUT).map_err(|err| match err {
            AcquireError::Closed => LinkError::Closed,
            AcquireError::Timeout => LinkError::Failed(PyRuntimeError::new_err(format!(
                "no free CUDA frame slot after {}s: frames are being submitted faster than the GPU \
                 finishes them",
                ACQUIRE_TIMEOUT.as_secs()
            ))),
        })
    }

    /// Copy `image` into a slot and publish it. Blocking only while every slot is in flight.
    fn submit(&self, image: &DeviceImage, keepalive: Py<PyAny>) -> Result<(), LinkError> {
        let slot = self.acquire()?;
        let enqueued = (|| -> Result<(), CudaError> {
            if let Some(producer) = image.stream {
                self.stream.wait_for(producer)?;
            }
            if slot.wait_release > 0 {
                self.release.wait(&self.stream, slot.wait_release)?;
            }
            let row_bytes = image.row_bytes();
            // SAFETY: `keepalive` holds the source array until the completion callback, the slot
            // is ours until then, and the caller's stream waits for the copy (below) before it can
            // overwrite the source.
            unsafe {
                self.stream.copy_2d_to_device(
                    image.ptr,
                    image.row_stride,
                    self.slot_ptr(slot.index),
                    row_bytes,
                    row_bytes,
                    image.height as usize,
                )?;
            }
            if let Some(producer) = image.stream {
                self.stream.make_wait(producer)?;
            }
            Ok(())
        })();
        let publish = enqueued.is_ok();
        self.finish(slot, publish, Some(keepalive))?;
        enqueued.map_err(LinkError::from)
    }

    /// Signal `ready` (when publishing) and hand the slot back once everything enqueued on the
    /// stream so far is done: published to the render thread, or freed.
    fn finish(&self, slot: AcquiredSlot, publish: bool, keepalive: Option<Py<PyAny>>) -> Result<(), CudaError> {
        let mut next_ready = lock(&self.next_ready);
        let mut value = None;
        let mut signal_error = None;
        if publish {
            let candidate = *next_ready + 1;
            // On failure the slot is still handed back below, just not published.
            match self.ready.signal(&self.stream, candidate) {
                Ok(()) => {
                    *next_ready = candidate;
                    value = Some(candidate);
                }
                Err(err) => signal_error = Some(err),
            }
        }
        let shared = self.shared.clone();
        let waker = self.waker.clone();
        let index = slot.index;
        let done = move || {
            match value {
                Some(ready) => {
                    shared.publish(index, ready);
                    waker.wake();
                }
                None => shared.abandon(index),
            }
            // Dropped without the GIL: PyO3 defers the decref until a thread next holds it.
            drop(keepalive);
        };
        let shared = self.shared.clone();
        if let Err(err) = self.stream.on_complete(Box::new(done)) {
            // Couldn't enqueue the callback: wait it out here so the slot isn't lost.
            let _ = self.stream.synchronize();
            shared.abandon(index);
            return Err(err);
        }
        signal_error.map_or(Ok(()), Err)
    }
}

/// Ask the render thread for a CUDA layer of `width`×`height` for `viewport_id` and import it.
fn create_link(
    py: Python<'_>,
    dispatch: &CommandDispatch,
    viewport_id: u64,
    width: u32,
    height: u32,
) -> Result<CudaLink, CreateError> {
    if cfg!(target_os = "macos") {
        return Err(CreateError::Unsupported(
            "CUDA interop is not supported on macOS: there is no CUDA<->Metal path".into(),
        ));
    }
    if dispatch.floating_region.is_some() {
        return Err(CreateError::Unsupported(
            "CUDA interop only reaches viewports in the main window, not floating panels".into(),
        ));
    }
    // No driver, no point having the render thread build and export a layer.
    fastgui_interop_cuda::ensure_loaded().map_err(|err| CreateError::Unsupported(err.to_string()))?;
    if !dispatch.waker.is_bound() {
        return Err(CreateError::NotRunning);
    }
    let (respond, response) = oneshot_channel();
    dispatch
        .send(Command::CreateCudaSurface { viewport_id, width, height, respond })
        .map_err(|_| CreateError::WindowClosed)?;
    // Creating the layer takes milliseconds; this only guards against a render thread that is
    // shutting down and never gets to the command.
    const RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
    let handles = py
        .detach(|| response.recv_timeout(RESPONSE_TIMEOUT))
        .map_err(|err| if err.is_timeout() { CreateError::Unsupported("timed out waiting for the render thread".into()) } else { CreateError::WindowClosed })?
        .map_err(CreateError::Unsupported)?;
    let waker = dispatch.waker.clone();
    py.detach(|| CudaLink::import(handles, waker)).map_err(CreateError::Unsupported)
}

enum CreateError {
    /// `window.run()` hasn't started yet.
    NotRunning,
    WindowClosed,
    /// Interop can't work for this viewport (with the reason).
    Unsupported(String),
}

// ---------------------------------------------------------------------------------------------
// Per-viewport state and the two paths
// ---------------------------------------------------------------------------------------------

/// Which path `submit_cuda` took last, for `Viewport.cuda_status`.
#[derive(Clone, Default)]
enum Path {
    #[default]
    Unused,
    Interop,
    HostCopy(String),
}

#[derive(Default)]
pub(crate) struct CudaState {
    link: Option<Arc<CudaLink>>,
    /// When links were last created, to notice a window that keeps dropping them (a viewport
    /// bound to a window but not in its layout loses its layer every frame).
    created_at: Vec<Instant>,
    /// Why interop can't be used for this viewport; once set, `submit_cuda` host-copies.
    unavailable: Option<String>,
    /// Stream for host copies, on the device that owns the arrays.
    host_stream: Option<Stream>,
    path: Path,
}

impl CudaState {
    pub(crate) fn status(&self) -> String {
        match &self.path {
            Path::Unused => "unused".into(),
            Path::Interop => "interop".into(),
            Path::HostCopy(reason) => format!("host copy: {reason}"),
        }
    }
}

/// What a `submit_cuda` call produced: shown via the interop layer already, or a CPU frame for
/// the caller to submit like `submit_frame` does.
pub(crate) enum Submitted {
    Interop,
    Host(CpuFrame),
}

/// `Viewport.submit_cuda` for a CUDA array.
pub(crate) fn submit(
    py: Python<'_>,
    state: &Mutex<CudaState>,
    dispatch: Option<&CommandDispatch>,
    viewport_id: u64,
    image: DeviceImage,
    keepalive: Py<PyAny>,
    clear_pending_cpu_frame: impl Fn(),
) -> PyResult<Submitted> {
    // A thread holding this lock detaches to wait on the GPU and must be able to re-attach, so
    // wait for the lock detached too.
    let mut state = state.lock_py_attached(py).unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut attempts = 0;
    let host_reason = loop {
        if let Some(reason) = &state.unavailable {
            break reason.clone();
        }
        let Some(dispatch) = dispatch else {
            break "viewport isn't in a window yet".to_owned();
        };
        let link = match &state.link {
            Some(link) if (link.width, link.height) == (image.width, image.height) && !link.shared.is_closed() => {
                link.clone()
            }
            _ => {
                state.link = None;
                attempts += 1;
                let now = Instant::now();
                state.created_at.retain(|t| now.duration_since(*t) < CHURN_WINDOW);
                if attempts > 2 || state.created_at.len() >= CHURN_LIMIT {
                    break "the window keeps dropping the CUDA layer (is the viewport in its layout?)".to_owned();
                }
                state.created_at.push(now);
                // A CPU frame still queued would replace the new CUDA layer when it uploads.
                clear_pending_cpu_frame();
                match create_link(py, dispatch, viewport_id, image.width, image.height) {
                    Ok(link) => {
                        let link = Arc::new(link);
                        state.link = Some(link.clone());
                        link
                    }
                    Err(CreateError::NotRunning) => break "window.run() hasn't started yet".to_owned(),
                    Err(CreateError::WindowClosed) => return Err(PyRuntimeError::new_err("window has already closed")),
                    Err(CreateError::Unsupported(reason)) => {
                        state.unavailable = Some(reason.clone());
                        break reason;
                    }
                }
            }
        };
        let keep = keepalive.clone_ref(py);
        match py.detach(|| link.submit(&image, keep)) {
            Ok(()) => {
                state.path = Path::Interop;
                return Ok(Submitted::Interop);
            }
            // Replaced or dropped by the render thread since we last looked: make a new one.
            Err(LinkError::Closed) => state.link = None,
            Err(LinkError::Failed(err)) => return Err(err),
        }
    };
    let frame = host_copy(py, &mut state, &image)?;
    state.path = Path::HostCopy(host_reason);
    Ok(Submitted::Host(frame))
}

/// Copy `image` to host memory, ordered after its producer's stream.
fn host_copy(py: Python<'_>, state: &mut CudaState, image: &DeviceImage) -> PyResult<CpuFrame> {
    py.detach(|| -> Result<CpuFrame, CudaError> {
        let ordinal = pointer_device(image.ptr)?;
        if state.host_stream.as_ref().is_none_or(|s| s.context().device() != ordinal) {
            state.host_stream = Some(Stream::new(&PrimaryContext::new(ordinal)?)?);
        }
        let stream = state.host_stream.as_ref().expect("set above");
        if let Some(producer) = image.stream {
            stream.wait_for(producer)?;
        }
        let row_bytes = image.row_bytes();
        let mut data = vec![0u8; row_bytes * image.height as usize];
        // SAFETY: the caller holds the array for the duration of this call, and we synchronize
        // before `data` is read or dropped.
        unsafe {
            stream.copy_2d_to_host(image.ptr, image.row_stride, data.as_mut_ptr(), row_bytes, row_bytes, image.height as usize)?;
        }
        stream.synchronize()?;
        Ok(CpuFrame { width: image.width, height: image.height, format: PixelFormat::Rgba8, data })
    })
    .map_err(|err| PyRuntimeError::new_err(format!("can't read the CUDA array: {err}")))
}

/// `Viewport.create_cuda_surface`: a fresh layer of the given size, or an error saying why not.
pub(crate) fn create_surface(
    py: Python<'_>,
    state: &Mutex<CudaState>,
    dispatch: Option<CommandDispatch>,
    viewport_id: u64,
    width: u32,
    height: u32,
    clear_pending_cpu_frame: impl Fn(),
) -> PyResult<CudaSurface> {
    if width == 0 || height == 0 || width > fastgui_core::MAX_CPU_FRAME_EXTENT || height > fastgui_core::MAX_CPU_FRAME_EXTENT {
        return Err(PyValueError::new_err(format!(
            "surface edges must be 1..={} pixels (got {width}x{height})",
            fastgui_core::MAX_CPU_FRAME_EXTENT
        )));
    }
    let dispatch = dispatch.ok_or_else(|| {
        PyRuntimeError::new_err(
            "call window.set_viewport(viewport) or window.set_content(...) before viewport.create_cuda_surface()",
        )
    })?;
    // The render thread only exists once `window.run()` has started, and `run()` blocks, so a
    // call before it would wait forever, usually on the very thread about to call `run()`. A
    // background thread started just before `run()` is fine, so allow a short startup grace.
    const STARTUP_GRACE: Duration = Duration::from_secs(2);
    let started = py.detach(|| {
        let deadline = std::time::Instant::now() + STARTUP_GRACE;
        while !dispatch.waker.is_bound() {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        true
    });
    if !started {
        return Err(PyRuntimeError::new_err(
            "the render thread isn't running yet: viewport.create_cuda_surface() needs a running \
             window. window.run() starts the render thread and blocks, so call this from a widget \
             callback or another thread once run() is going (or use submit_cuda, which works before \
             run())",
        ));
    }
    let mut state = state.lock_py_attached(py).unwrap_or_else(|poisoned| poisoned.into_inner());
    state.link = None;
    clear_pending_cpu_frame();
    let link = match create_link(py, &dispatch, viewport_id, width, height) {
        Ok(link) => Arc::new(link),
        Err(CreateError::NotRunning) => unreachable!("waited for the render thread above"),
        Err(CreateError::WindowClosed) => return Err(PyRuntimeError::new_err("window has already closed")),
        Err(CreateError::Unsupported(reason)) => {
            return Err(PyRuntimeError::new_err(format!(
                "can't create a CUDA surface: {reason}. Viewport.submit_cuda() still works (it \
                 falls back to a host copy)"
            )))
        }
    };
    state.link = Some(link.clone());
    state.unavailable = None;
    state.path = Path::Interop;
    Ok(CudaSurface { link })
}

// ---------------------------------------------------------------------------------------------
// Zero-copy surface
// ---------------------------------------------------------------------------------------------

/// A CUDA-writable surface behind a `Viewport`, from `Viewport.create_cuda_surface`. Write frames
/// straight into it with `with surface.frame(stream) as f:` — `f` exposes
/// `__cuda_array_interface__`, so `torch.as_tensor(f, device="cuda")` or `cupy.asarray(f)` wrap
/// it without a copy.
///
/// **Unverified on NVIDIA hardware.**
#[pyclass]
pub(crate) struct CudaSurface {
    link: Arc<CudaLink>,
}

#[pymethods]
impl CudaSurface {
    #[getter]
    fn width(&self) -> u32 {
        self.link.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.link.height
    }

    /// True once the window dropped this surface (closed, viewport removed, resized by another
    /// `create_cuda_surface`, or replaced by `submit_frame`). `frame()` raises from then on.
    #[getter]
    fn closed(&self) -> bool {
        self.link.shared.is_closed()
    }

    /// A frame to write on `stream` (a CUstream int, `torch.cuda.Stream`, cupy stream, or
    /// `__cuda_stream__` object; default: the legacy default stream). Use it as a context
    /// manager: entering waits (on `stream`, GPU-side) until the slot is free, leaving publishes
    /// what was written on `stream` by then. An exception inside the block drops the frame.
    #[pyo3(signature = (stream = None))]
    fn frame(&self, stream: Option<&Bound<'_, PyAny>>) -> PyResult<CudaFrame> {
        let stream = stream.filter(|s| !s.is_none()).map(parse_stream).transpose()?.unwrap_or(RawStream(0));
        Ok(CudaFrame { link: self.link.clone(), stream, slot: Mutex::new(FrameSlot::Unentered) })
    }
}

#[derive(Clone, Copy)]
enum FrameSlot {
    Unentered,
    Open(AcquiredSlot),
    Done,
}

/// One frame of a `CudaSurface`; see `CudaSurface.frame`.
#[pyclass]
pub(crate) struct CudaFrame {
    link: Arc<CudaLink>,
    stream: RawStream,
    slot: Mutex<FrameSlot>,
}

#[pymethods]
impl CudaFrame {
    fn __enter__<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let this = slf.borrow();
        if !matches!(*lock(&this.slot), FrameSlot::Unentered) {
            return Err(PyRuntimeError::new_err("a CudaFrame can only be entered once"));
        }
        let link = this.link.clone();
        let user = this.stream;
        let slot = py
            .detach(|| -> Result<AcquiredSlot, LinkError> {
                let slot = link.acquire()?;
                let ordered = (|| -> Result<(), CudaError> {
                    if slot.wait_release > 0 {
                        link.release.wait(&link.stream, slot.wait_release)?;
                    }
                    link.stream.make_wait(user)
                })();
                if let Err(err) = ordered {
                    let _ = link.finish(slot, false, None);
                    return Err(err.into());
                }
                Ok(slot)
            })
            .map_err(|err| match err {
                LinkError::Closed => PyRuntimeError::new_err("this CudaSurface was closed by the window"),
                LinkError::Failed(err) => err,
            })?;
        *lock(&this.slot) = FrameSlot::Open(slot);
        Ok(slf.clone())
    }

    #[pyo3(signature = (exc_type, _exc_value, _traceback))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let publish = exc_type.is_none();
        self.close(py, publish).map_err(cuda_err)?;
        Ok(false)
    }

    /// `(height, width, 4)` uint8, packed rows, valid inside the `with` block.
    #[getter]
    fn __cuda_array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let FrameSlot::Open(slot) = *lock(&self.slot) else {
            return Err(PyRuntimeError::new_err("use the CudaFrame inside its `with` block"));
        };
        let link = &self.link;
        let cai = PyDict::new(py);
        cai.set_item("shape", (link.height as usize, link.width as usize, 4usize))?;
        cai.set_item("typestr", "|u1")?;
        cai.set_item("data", (link.slot_ptr(slot.index), false))?;
        cai.set_item("strides", py.None())?;
        cai.set_item("version", 3)?;
        // Writes must go on (or after) the stream the slot was made ready on. 0 is not a valid
        // value here; the legacy default stream is 1.
        cai.set_item("stream", if self.stream.0 == 0 { 1 } else { self.stream.0 })?;
        Ok(cai)
    }
}

impl CudaFrame {
    fn close(&self, py: Python<'_>, publish: bool) -> Result<(), CudaError> {
        let slot = {
            let mut state = lock(&self.slot);
            let FrameSlot::Open(slot) = *state else { return Ok(()) };
            *state = FrameSlot::Done;
            slot
        };
        let link = &self.link;
        let user = self.stream;
        py.detach(|| match link.stream.wait_for(user) {
            Ok(()) => link.finish(slot, publish, None),
            Err(err) => {
                let _ = link.finish(slot, false, None);
                Err(err)
            }
        })
    }
}

impl Drop for CudaFrame {
    fn drop(&mut self) {
        // Entered but never exited (the `with` was bypassed): give the slot back unpublished.
        if let FrameSlot::Open(slot) = *lock(&self.slot) {
            if self.link.stream.wait_for(self.stream).is_ok() {
                let _ = self.link.finish(slot, false, None);
            } else {
                let _ = self.link.stream.synchronize();
                self.link.shared.abandon(slot.index);
            }
        }
    }
}
