# fast-gui — Status & Roadmap

Lightweight, GPU-native, no-GIL-safe Python GUI toolkit. This doc is a snapshot for picking
work back up in a fresh session (or handing to a sub-agent) with no prior conversation
context. Project root: `c:\sandbox\fast-gui` (Windows, not yet a git repo).

Read this whole file before starting M6 — especially **"How this project has been built"**
near the bottom; the verification discipline described there caught real bugs that
`cargo check` alone missed, twice.

## Environment (Windows dev machine)

- Rust via rustup (`stable-x86_64-pc-windows-msvc`). `cargo`/`rustc` are at
  `%USERPROFILE%\.cargo\bin`, not reliably on PATH in fresh shells — prepend it explicitly:
  `$env:Path = "$env:USERPROFILE\.cargo\bin;" + [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`
- Visual Studio Build Tools (MSVC 14.51) installed — needed for the `-msvc` Rust target.
- Vulkan SDK 1.4.357.0 at `C:\VulkanSDK\1.4.357.0` (`Bin\glslc.exe` used to precompile
  shaders to `.spv`, checked into `crates/fastgui-render-vk/shaders/`).
- GPU on this machine: **AMD Radeon 840M integrated only — no NVIDIA GPU.** CUDA-dependent
  code (M3) has never run against real CUDA hardware; see the M3 section below.
- Two Python venvs at the project root, both with `numpy` and `maturin` pip-installed:
  - `.venv` — regular GIL-enabled CPython 3.14.3
  - `env` — free-threaded CPython 3.14.3 (`Py_GIL_DISABLED`), user-installed
- Build/iterate loop: from the project root, with the target venv activated —
  ```powershell
  $env:VIRTUAL_ENV = "c:\sandbox\fast-gui\.venv"          # or ...\env for free-threaded
  $env:Path = "c:\sandbox\fast-gui\.venv\Scripts;" + $env:Path
  maturin develop
  ```
  Rebuild for **both** venvs before calling a change done — several real bugs only showed up
  (or needed re-verifying) on one build and not the other.
- **Gotcha**: a still-running `python.exe` from a previous test locks the built `.pyd`, and
  `maturin develop` fails with a file-in-use error. `taskkill //F //IM python.exe` before
  rebuilding if a prior run wasn't cleanly closed.
- **Gotcha**: PowerShell tool calls in this harness do **not** preserve variables or
  `Add-Type` definitions across separate calls — each call is a fresh session. Any C# P/Invoke
  helper type (window capture, click simulation) must be redefined in the same call that uses
  it.
- **Verification methodology** (see also the dedicated section below): launch an example via
  `Start-Process`, find the *actual* window by `MainWindowTitle` across all `python*`
  processes (the launched `python.exe` spawns a child `python.exe` — the window belongs to the
  child, not the PID `Start-Process` returns), capture via the Win32 `PrintWindow` API (not
  `SetForegroundWindow` + `CopyFromScreen` — a background process can't reliably steal focus
  from whatever you were last interacting with, e.g. an IDE, so that approach intermittently
  screenshots the wrong window). Close via `PostMessage(hwnd, WM_CLOSE, ...)` and confirm the
  process actually exits with empty stderr.
- Bash tool's `/tmp` maps to `C:\Users\...\AppData\Local\Temp` — use `cygpath -w` to get a
  Windows path before passing to the `Read` tool.
- Don't write files outside the project folder without checking first (raised explicitly by
  the user mid-project) — use the project directory or the session's scratchpad only.

## Crate layout

```
Cargo.toml (workspace), rust-toolchain.toml
crates/
  fastgui-core/         cross-thread primitives + widget tree
    src/queue.rs           command_channel (unbounded MPSC), oneshot_channel
    src/readback.rs        Readback<T> (Mutex-cached state for sync getters)
    src/frame.rs            FrameSlot<T> (latest-wins mailbox), CpuFrame, PixelFormat
    src/widget.rs           WidgetTree (taffy-backed), WidgetKind, WidgetId, Color, hit_test
  fastgui-render/        Renderer trait (backend-agnostic; fastgui-render-vk implements it)
  fastgui-render-vk/     Vulkan backend (Windows + Linux)
    src/renderer.rs         VulkanRenderer: instance/device/swapchain, dynamic rendering
    src/app.rs               winit ApplicationHandler: command draining, input, render loop
    src/command.rs           Command enum (SetClearColor/SetViewport/CreateCudaSurface/
                              MutateWidgetTree)
    src/pipeline.rs          ViewportPipeline: fullscreen-triangle shaders + descriptor set
    src/texture.rs           ViewportTexture: host-visible LINEAR CPU-upload texture
    src/cuda_texture.rs      CudaSharedTexture: exportable image + timeline semaphore (M3)
    shaders/                 viewport.vert/.frag + precompiled .spv
    examples/resize_probe.rs  standalone winit+Vulkan resize-latency diagnostic (no fastgui
                              code) — run with `cargo run --example resize_probe -p
                              fastgui-render-vk` if live resize ever regresses; see M6 status's
                              "Tenth round" for what it found and why it's kept
  fastgui-render-mtl/    Metal backend (macOS), via objc2-metal/objc2-quartz-core — M5, see its
                          status section below for what's verified vs. not
    src/renderer.rs         MetalRenderer: CAMetalLayer/device/pipeline setup, chrome + Viewport
                              texture upload, render
    src/app.rs               winit ApplicationHandler — mirrors fastgui-render-vk::app closely
                              (same widget/drag logic), no CUDA, no resize debounce
    src/pipeline.rs          ViewportPipeline: MSL fullscreen-triangle + sampling shaders
    src/texture.rs           ViewportTexture: MTLTexture + replaceRegion CPU upload
  fastgui-chrome/        cosmic-text + tiny-skia widget rasterizer (ChromeRenderer)
  fastgui-interop-cuda/  CUDA driver API FFI, dynamically loads nvcuda.dll. UNVERIFIED — see M3.
  fastgui-py/            PyO3 bindings
    src/backend.rs           cfg-picks fastgui-render-mtl (macOS) vs fastgui-render-vk
                              (else) so the rest of this crate names Command/CommandDispatch/
                              EventWaker/RenderThreadHandles/run without a cfg at each call site
    src/lib.rs               Window, Viewport, CudaSurface (macOS: stub that always errors —
                              see M5's status section)
    src/widgets.rs            Box/Label/Button/Slider/Splitter/Panel, DescribedWidget
                              tree-building (see M6 status for Splitter/Panel)
python/
  fastgui/__init__.py, __init__.pyi   DockArea (pure Python, see M6 status) lives here
  examples/
    basic_window.py        M0
    threaded_mutation.py   M1
    live_camera_feed.py    M2
    cuda_viewport.py       M3 (unverified — see caveat)
    widgets_demo.py        M4
    dock_layout.py          M6 (rendering verified; drag NOT interactively verified — see M6 status)
pyproject.toml            maturin backend config — minimal so far, no abi3/free-threaded
                           wheel matrix yet (that's M6 work)
README.md, docs/ARCHITECTURE.md   added in 6C — read these instead of re-deriving crate
                           layout/threading model from scratch each session; this doc stays
                           the place for milestone history and in-progress reasoning, those
                           are the stable reference
```

No `.gitignore` and no git repo yet — worth setting up in M6 (see below).

## Resolved dependency versions (worth knowing before re-researching)

`ash 0.38.0+1.3.281`, `ash-window 0.13.0`, `winit 0.30.13`, `taffy 0.12.2`,
`cosmic-text 0.14.2` (0.19.0 is latest but its API differs — 0.14.2's `Buffer::set_size`/
`set_text` take `&mut FontSystem` explicitly; later versions may not — verify against
whatever actually resolves before assuming), `tiny-skia 0.11.4`, `pyo3 0.29.1`,
`libloading 0.8`, `crossbeam-channel 0.5`.

PyO3 0.29.1 specifics that cost time to discover: `Python::attach`/`.detach()` (not
`with_gil`/`allow_threads`), `Py<T>` doesn't implement `Clone` — use `.clone_ref(py)`,
`Bound::cast` (not `.downcast`).

## Milestones completed (M0–M4)

**M0 — Vulkan window bring-up.** Cargo workspace, winit event loop, `ash`-based Vulkan
instance/device/swapchain, PyO3 `Window` class with `.run()`.

**M1 — Cross-thread command-queue architecture.** `fastgui_core::command_channel` (unbounded
MPSC) + `Readback<T>`. `Window.set_clear_color()` is the proof case: callable from any
thread, applied by the render thread once per frame. Verified under **both** regular and
free-threaded Python with a real concurrent stress test (8 threads hammering
`set_clear_color`, confirmed via screenshots that the color actually cycles).

**M2 — Viewport widget + CPU texture upload.** `FrameSlot<T>` (latest-wins mailbox — different
semantics from the command queue, which never drops). Bumped to Vulkan 1.3 +
`VK_KHR_dynamic_rendering` (no render pass/framebuffer boilerplate). `ViewportTexture`:
host-visible `LINEAR` `R8G8B8A8_UNORM` image, uploaded via CPU memcpy respecting row pitch.
Fullscreen-triangle pipeline (no vertex buffer, `gl_VertexIndex` trick). `Viewport.submit_frame
(numpy_array)` via the Python buffer protocol, RGB auto-expanded to RGBA. Verified with
`live_camera_feed.py` (animated numpy gradient, confirmed animating across captured frames).

**M3 — CUDA↔Vulkan external memory/timeline-semaphore interop.** `fastgui-interop-cuda`: raw
CUDA driver API FFI with struct layouts transcribed from NVIDIA's own `cuda-python`
bindings-generator source (not from memory), dynamically loading `nvcuda.dll`. Graceful
failure on this NVIDIA-less machine confirmed clean (RuntimeError, no crash) under both
venvs. `CudaSharedTexture`: device-local `LINEAR` image exported via
`VK_KHR_external_memory_win32` + a `VK_KHR_timeline_semaphore` exported via
`VK_KHR_external_semaphore_win32`. `Viewport.create_cuda_surface(w, h)` is the one command
that legitimately needs a synchronous reply (a real device pointer), via `oneshot_channel`.
**Found and fixed two real bugs using validation layers on real (AMD) hardware**: a missing
`VkMemoryDedicatedAllocateInfo` for the exported memory, and a missing `timelineSemaphore`
device feature enable. After fixing both, the **Vulkan-side export path runs with zero
validation errors on real hardware.** The CUDA-side import (`cuImportExternalMemory` etc.)
**remains completely unverified** — no NVIDIA GPU has ever been available to test it. **This
is the single highest-risk unverified area of the whole project.** If anyone picks up work
near CUDA interop, prioritize getting time on real NVIDIA hardware before relying on it.

**M4 — Widget tree, layout, text rendering, input.** `fastgui_core::widget::WidgetTree` wraps
`taffy::TaffyTree`; `WidgetKind` enum (`Container`/`Label`/`Button`/`Slider`); absolute-rect
tracking + `hit_test()`. `fastgui-chrome::ChromeRenderer` rasterizes the *whole* tree into one
window-sized RGBA8 buffer via `cosmic-text` (shaping) + `tiny-skia` (rasterization) — reuses
M2's `ViewportTexture` upload path directly, no new Vulkan code needed.
`Command::MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>)` is closure-based
specifically so the `Command` enum doesn't need a new variant per widget property. Python
widgets (`Box`/`Label`/`Button`/`Slider`) are pure descriptors until `Window.set_content()`
walks them into the real tree; each holds `IdCell`/`SenderCell`
(`Arc<Mutex<Option<...>>>`) populated once attached, enabling later `label.set_text(...)`.
Mouse input (hit-test, `Button` click, `Slider` drag) dispatches to Python callbacks wrapped
via `Python::attach` (callbacks fire from the render thread, which normally runs detached from
the GIL).

Three real bugs found by actually running things, not by `cargo check`:
1. **Deadlock**: `set_content()` originally blocked synchronously waiting for the render
   thread to process it — but the render thread *is* the calling thread until `.run()`
   executes, so calling `set_content()` before `run()` (the normal pattern!) hung forever.
   Fixed by making it fire-and-forget, like every other mutator.
2. **Color-space bug**: the swapchain used an `_SRGB` surface format, so the GPU silently
   gamma-re-encoded colors already meant as final display bytes, washing out midtones. Fixed
   by switching to `_UNORM` — confirmed via before/after screenshots. This also improves M2's
   color accuracy retroactively.
3. `taffy::Style` contains a raw pointer (for `calc()` expressions) and isn't `Send`, breaking
   the cross-thread command closure. Fixed by carrying plain `Send`-safe params across the
   thread boundary and building the real `Style` on the render thread.

Verified via simulated mouse clicks/drags (Win32 `SetCursorPos`/`mouse_event` automation), not
just static screenshots: clicking the button 3 times produced "Count: 3"; dragging the slider
produced "Step: 0.96" with the thumb visibly moved.

Known simplifications carried forward from M4 (fine for now, worth knowing about):
- Text sizing during layout uses a rough heuristic (`measure_text` in
  `fastgui-core/src/widget.rs`), not real `cosmic-text` shaping — would need upgrading for
  tight layouts or non-Latin scripts.
- No aspect-ratio preservation for `Viewport` (texture always stretches to fill the window).
- Chrome and `Viewport` content are mutually exclusive per `Window` (whichever was set most
  recently wins) — no compositing layer between them yet.
- No text wrapping, no scrolling, no keyboard input/focus handling.
- Single-buffered CPU-uploaded textures (M2/M4) can occasionally show one torn frame under
  fast updates — documented in code, non-fatal.

## M5 — Metal backend (macOS)

### Status as of 2026-09-05

Landed: `fastgui-render-mtl` is a full parallel implementation of `fastgui-render-vk`'s
non-CUDA functionality — `MetalRenderer` (device/`CAMetalLayer`/pipeline setup, chrome +
per-`Viewport` texture upload, render), `ViewportPipeline` (MSL fullscreen-triangle vertex
shader + sampling fragment shader, compiled from an embedded source string, same trick
`viewport.vert`/`.frag` use in GLSL), `ViewportTexture` (`MTLTexture` + `replaceRegion` CPU
upload), and its own `app.rs`/`command.rs`/`error.rs` mirroring the Vulkan backend's shapes
closely enough that `fastgui-py` picks whichever backend applies via a tiny `cfg`-gated
`backend` shim module (`crates/fastgui-py/src/backend.rs`) instead of the (largely unused)
`fastgui_render::Renderer` trait.

**Binding choice: `objc2-metal`/`objc2-quartz-core`, not `metal-rs`.** Both were on the table;
`objc2-metal` won because it was already resolved in `Cargo.lock` (pulled in transitively by
`winit`'s own macOS backend, which uses `objc2`/`objc2-app-kit` for `NSWindow`/`NSView`) —
using it keeps the whole stack on one Objective-C runtime instead of mixing in `metal-rs`'s
older `objc`-crate-based one, and it's the same kind of direct, 1:1-with-Apple's-headers
binding style this project already chose for Vulkan (`ash`) over a higher-level wrapper.

**Architecture notes:**
- `CAMetalLayer` is attached directly to winit's own `NSView` (`view.setWantsLayer(true)` +
  `view.setLayer(metal_layer.as_super())`) — layer-*hosting*, not layer-*backed*, which means
  (unlike a plain auto-backed view) AppKit does **not** keep the layer's `frame`/`contentsScale`
  in sync with the view on its own. `MetalRenderer::resize()` does that explicitly every call:
  recomputes the layer's point-space `frame` from the view's `backingScaleFactor` and sets
  `drawableSize` in physical pixels.
- No resize-debounce logic (unlike `fastgui-render-vk::app`'s `RESIZE_DEBOUNCE`, a ~2s-stall
  DWM/swapchain-recreation workaround specific to Windows): `CAMetalLayer::setDrawableSize` is
  cheap (no GPU object recreation), so `app.rs` applies a resize immediately on every
  `WindowEvent::Resized`.
- No CUDA interop path at all (`fastgui_render_mtl::Command` has no `CreateCudaSurface`
  variant) — Apple hasn't shipped an NVIDIA GPU since ~2019, so there's no zero-copy
  CUDA↔Metal surface to build. `fastgui-py`'s `Viewport.create_cuda_surface` is
  `#[cfg(target_os = "macos")]`-overridden to return a clear `RuntimeError` instead of ever
  reaching the render thread; `Viewport.submit_frame()` (CPU copy) works unchanged.
- `ViewportTexture` uses `MTLStorageModeShared` unconditionally — correct and fast on Apple
  Silicon's unified memory (this project's primary target), but leaves a real optimization on
  the table for a discrete-GPU Intel Mac (`Managed` storage + explicit `didModifyRange` sync),
  mirroring the Vulkan backend's own carried-forward single-buffered-upload simplification.

**Verification — real but partial.** `cargo check --workspace` is clean (including
`fastgui-render-vk` still compiling fine on macOS, now simply unused there — `fastgui-py`
selects the backend via `target.'cfg(target_os = "macos")'.dependencies` in its `Cargo.toml`).
`maturin develop` builds and installs a real extension module; both `basic_window.py`
(clear-color-only path) and `widgets_demo.py` (the full path: MSL shader compile → render
pipeline state → texture upload → sampler → `drawPrimitives` → present, via
`fastgui-chrome`'s text/widget rasterization) launch, sit at 0% CPU under `ControlFlow::Wait`
(not busy-looping or erroring in a retry loop) for 20+ seconds, and shut down cleanly on
`SIGTERM` with no crash report. **This session's sandbox had no Screen Recording/Accessibility
permission** (`screencapture`/`osascript`/`Quartz` all refused), so none of this was confirmed
by an actual screenshot or by interactively clicking the button/dragging the slider — only by
process behavior and the absence of errors/crashes. Whoever picks this up next on a normal
interactive desktop session should: screenshot `widgets_demo.py` to confirm the button/label/
slider actually render (not just that nothing crashed), confirm click/drag input work (same
gap M4/M6 flagged for other sessions' sandboxes), and try `dock_layout.py`/`live_camera_feed.py`
for the per-`Viewport`-widget-rect draw path (only chrome's full-window draw has been exercised
so far).

## M6 — remaining work (this session's scope, per explicit user choice)

The user chose **full dockable/floating/tabbed panels** (not just a resizable splitter) when
asked to scope this, understanding it's comparable in size to all of M4 by itself. Combined
with wheel packaging, docs, and polish, **M6 is realistically its own multi-session
milestone** — don't try to rush the docking system.

### Status as of 2026-08-11 (session 1 of M6)

Landed, in `crates/` and `python/`:

- **`Splitter`** (new `WidgetKind::Splitter` in `fastgui-core::widget`): a draggable divider
  between two sibling panes. The bar stores `first`/`second` `WidgetId`s (wired up by
  `fastgui-py::widgets::attach`'s special-case splitter handling — see its doc comment); drag
  updates a `ratio` and reassigns both panes' `flex_grow` via the new
  `WidgetTree::set_flex_grow` (`fastgui-render-vk::app::update_dragged_splitter`, modeled
  directly on the existing `Slider` drag code). Python: `fg.Splitter(first, second, direction,
  ratio, bar_color, thickness)`.
- **`Panel`**: a titled container (fixed-height title bar + flex-grow content pane). Pure
  composition of existing `Container`/`Label` widget kinds, no new `WidgetKind`. Python:
  `fg.Panel(title, content, ...)`.
- **`DockArea`**: pure-Python (`python/fastgui/__init__.py`), *not* a Rust type — composes
  `Panel`s onto a `Splitter` tree one `add_panel(panel, region, size)` call at a time. Works via
  a small new hook in `fastgui-py::widgets::describe()`: any Python object exposing a
  `_fastgui_widget` property is resolved through it, so composite widgets don't need a matching
  Rust type. This is **step 1 (splitter) + step 2 (static DockArea) from the plan below** —
  panels resize via drag (once interactively verified — see caveat), but there is deliberately
  no tabs, no drag-to-rearrange, and no floating yet (steps 3–5, untouched).
- New example: `python/examples/dock_layout.py` — a 3-panel layout (center placeholder, right
  "Controls", bottom "Log"), each region a `Panel` inside a `DockArea`.
- Rendering/layout verified by screenshot on both `.venv` and `env` (free-threaded): title bars,
  panel backgrounds, and splitter bars all render at the geometrically correct positions for the
  configured ratios (960×600 window, row bar measured by pixel-scanning the screenshot at
  x=738–743 for a 0.72/0.28 split, matching the flex-grow math).
- One real bug found and fixed the same way M0–M4's bugs were found: `Panel`'s title `Label`
  was invisible. Cause: the title bar's default `align-items: stretch` squeezed the label to the
  bar's padded interior height (12px) before cosmic-text ever got a chance to lay out a
  14pt line — fixed by adding an `align_items` override to `StyleParams`, set to `Center` for
  `Panel` title bars. A second bug in the same area: short titles (e.g. "Log") got clipped
  to "Lo" — `measure_text`'s heuristic (see M4 known-simplifications) underestimates short
  strings' real width badly enough to matter at this scale; fixed by giving the title label
  `flex_grow=1` so it gets the bar's full remaining width instead of a tight intrinsic box.

**Caveat — could not interactively verify drag-to-resize this session.** This project's
discipline is to verify by actually running and interacting with things, not just screenshotting
static state (see "How this project has been built" below). I attempted that: simulated
mouse-drag on the splitter bar via `SetCursorPos`+`mouse_event` (the method M4's button-click/
slider-drag verification used, per that section above), then via raw `SendMessage`/`PostMessage`
straight to the window's message queue (bypassing focus/z-order entirely), then via `SendInput`.
**None of the three registered any click at all — not even on the untouched, previously-verified
M4 `widgets_demo.py`'s `Button`** (triple-clicked it directly, `Count:` stayed at `0` across all
attempts). `GetForegroundWindow()` during this session showed the fast-gui window was never
actually foreground (something else — the editor — held it), and `PostMessage`-injected
`WM_LBUTTONDOWN`/`WM_LBUTTONUP` were accepted (returned `TRUE`) but produced no state change
either. This points to a session/sandbox input-delivery limitation specific to *this*
conversation's environment, not a regression in the splitter code — the drag logic is a
near-literal copy of the already-interactively-verified `Slider` drag path (same
hit-test → store dragging id → cursor-move recomputes → `mutate_kind` pattern). **Still,
this is unverified, not verified** — whoever picks up M6 next should re-run
`dock_layout.py` on a normal interactive desktop session and confirm the bars actually drag
before trusting this further, and ideally figure out why input injection didn't reach any window
this session (worth a sentence in a future update either way).

**Update, same session, after user testing interactively:** the user confirmed my "could not
verify" caveat above was masking a real environment limitation, not that the feature was
broken — they *could* click/drag normally on their own machine, and reported two real bugs from
that live session:
1. No resize cursor over a `Splitter` bar (Windows only shows this automatically for its own
   native window-border resize regions; a client-area splitter bar needs it set explicitly).
2. Dragging a splitter — and separately, resizing the OS window itself — didn't visually update
   until the mouse was released, only snapping to the final state at that point.

Root cause for (2): this app's render loop is purely `ControlFlow::Poll` +
`window.request_redraw()` called at the end of handling `RedrawRequested`, i.e. "draw a frame,
then ask winit for another one, repeat." That self-scheduling chain doesn't get a turn during
either (a) `CursorMoved` while a button is held (nothing was forcing a repaint synchronously —
it relied on the *next* `RedrawRequested` happening to come around) or, worse, (b) an OS window
border drag, which Windows runs through its own modal loop (`WM_ENTERSIZEMOVE`) that only pumps
messages winit forwards to us directly — our independent "next redraw" scheduling doesn't run at
all until that modal loop exits on mouse-up. Fixed by adding `App::render_now` (extracted from
the old `RedrawRequested` body) and calling it *synchronously* from both `Resized` and from
`CursorMoved` while dragging, instead of only ever from `RedrawRequested`. Also added
`App::update_cursor_icon`, called on every `CursorMoved` (and on drag-release), which sets
`CursorIcon::ColResize`/`RowResize` while hovering or dragging a `Splitter`, `Default`
otherwise. Both fixes are in `crates/fastgui-render-vk/src/app.rs`; rebuilt and screenshot-
verified on both venvs (renders fine, no crash) but the *interactive* fix itself — does it now
actually track the cursor live, does the resize icon show — still needs the user's own
confirmation, since this session still can't drive real mouse input itself.

**Second round of user feedback, same session:** cursor icon confirmed working. Live-update
during drag now happens, but felt "really slow" / laggy while actively moving the mouse. Cause:
calling `render_now` (full chrome re-rasterization + texture upload + present) synchronously on
*every* `CursorMoved` serializes input handling behind however long a full render takes — a
fast-moving mouse can queue many `CursorMoved` messages between two `Poll` iterations, and
rendering all of them one-by-one is much more expensive than just updating a flex ratio.
Fixed by only calling `render_now` synchronously from `Resized` (still needed there — that's the
OS modal-loop case, a different problem); the drag path now just calls `window.request_redraw()`
after applying the state update, letting winit coalesce however many drag events land between
paints instead of rendering once per pixel of mouse movement. Rebuilt and running; **not yet
re-confirmed smooth by the user** — next step if picking this up.

**Third round of user feedback, same session:** worse, not better — dragging now updates live,
but freezes for multi-second stretches, cursor visibly stuck on the resize icon, no way to
interrupt it until the freeze ends. This is a real hang/severe-stall symptom, not just
"throttled," and the likely explanation is that `render_now` (rasterize + GPU submit/present) is
individually expensive enough (plausibly 15–30ms+ on this machine's AMD 840M integrated GPU,
unverified — see below) that calling it synchronously up to the drag throttle's cap, every time,
for the *entire duration* of a multi-second drag, adds up to real multi-second total blocking
time — and because it's synchronous/reentrant inside `CursorMoved` dispatch, nothing (including
cursor-icon updates) can happen while it's draining. Rather than guess a fourth architecture
change without data, added **temporary instrumentation**: `App::render_now`
(`crates/fastgui-render-vk/src/app.rs`) now times `upload_active_content` (chrome
rasterize+upload) and `render_frame` (GPU submit+present) separately and `eprintln!`s both any
time a frame takes ≥5ms — silent otherwise, so normal operation doesn't spam stderr. Also dropped
the drag-render cap from 120fps to 30fps as a stopgap (less total render volume per drag
regardless of per-frame cost). **Next step for whoever continues this: run `dock_layout.py` from
a terminal (so stderr is visible) while dragging a splitter, and read off which of the two stages
is actually slow** — that tells you whether the fix belongs in `fastgui-chrome`'s rasterizer
(e.g. cosmic-text is re-shaping more than it needs to, or tiny-skia's full-window redraw is the
cost) or in `renderer.rs`'s Vulkan submit/present path (e.g. a fence wait that's blocking longer
than one vsync interval for some reason). Don't keep iterating on `app.rs`'s scheduling logic
further without that data — three attempts already went in blind and each one either didn't fix
it or made it worse.

**Fourth round:** user ran it from a terminal as asked — **zero perf lines printed**, yet the
freeze was identical. This rules out `render_now` (rasterize+upload+present) itself being slow;
every single call across the whole drag session stayed under 5ms. Widened instrumentation
(`crates/fastgui-render-vk/src/app.rs`) to cover the *entire* `CursorMoved` handler (not just
`render_now`) and to log `CursorMoved` events/sec once a second, to distinguish two remaining
possibilities: (a) something else *inside* the handler (state mutation, hit-testing, cursor-icon
update) is the real cost, which the wider per-handler timer would now catch, or (b) our own
process's work is all fast and the felt freeze is actually downstream of it — e.g. in display
compositing — in which case `vkQueuePresent` can return quickly on our end (nothing to time)
while the frame backs up before it's actually flipped to screen, which would be invisible to any
timer inside this process. (b) would be notable given this whole session has been running inside
a sandboxed/remoted environment where synthetic input delivery already didn't behave normally
(see the input-automation caveat above) — a virtualized display/GPU path wouldn't be shocking
either. Rebuilt; **next step is the same as last time — run from a terminal, drag, and report
what (if anything) gets printed**, plus this time note roughly how many events/sec are logged
during the drag.

**Sixth round — a real, separate bug, now fixed.** Startup got better once system load eased, but
the user specifically flagged that live *OS window-border* resize (even on pristine `basic_window
.py`, M0, no widget/splitter code at all) was still awful. This one had a concrete, findable cause,
unrelated to the earlier system-load red herring: `WindowEvent::Resized`'s handler called
`renderer.resize(...)` — a full swapchain recreation (`device_wait_idle` + destroy/recreate) —
**unconditionally on every single `WM_SIZE`**, completely unthrottled (the drag-render throttle
only ever applied to `CursorMoved`). Windows sends `WM_SIZE` about as often as `WM_MOUSEMOVE`
during a live border drag, and swapchain recreation is a much heavier operation than a normal
frame render. Fixed properly in two parts (`crates/fastgui-render-vk/src/app.rs`):
1. `Resized` now only does cheap `width`/`height` bookkeeping; `render_now` itself checks
   `last_applied_size` and only calls `VulkanRenderer::resize` when the size actually changed
   since it last did — so a burst of `WM_SIZE` collapses to however many *throttled* render ticks
   actually ran, not one recreation per pixel of drag movement, and a final call is always
   guaranteed to catch it up (no risk of the swapchain staying permanently stale).
2. Split the single render throttle into two: `DRAG_RENDER_INTERVAL` (60fps, for `CursorMoved` —
   cheap, no resource recreation) and a much coarser `RESIZE_RENDER_INTERVAL` (200ms) for
   `Resized`. This split was necessary, not cosmetic: an automated stress test (rapid synthetic
   `WM_SIZE` bursts via `SetWindowPos`) showed that even at a 60fps throttle, `dock_layout.py`
   (real widget content, so `Resized` also re-rasterizes chrome and recreates the CPU-upload
   content texture, not just the swapchain) could take **tens of seconds** to drain a
   half-second burst of resize events — each individual recreation on this machine apparently
   costs enough that a 16ms throttle interval barely reduces total work (the gate is already
   satisfied again the instant the previous slow recreation finishes). `basic_window.py` (M0, no
   widget content, nothing for `Resized` to do beyond the swapchain) was unaffected by this,
   which is why it looked fine in isolation earlier and is a useful data point for anyone
   revisiting this: *the cost scales with active content, not just window size*.
   **Verified via an automated stress test** (not just eyeballed): 8 synthetic `WM_SIZE` events
   ~25ms apart against a running `dock_layout.py`, confirmed `Get-Process`'s `Responding` stayed
   `True` throughout (no hang) and the final screenshot showed correct layout reflow at the new
   size, empty stderr.
3. **Superseded by the eighth/ninth rounds below** — this throttled-recreation approach turned
   out not to be enough; kept here for the reasoning trail, but see the final state described
   there (recreation deferred entirely until the drag settles, not just throttled).

**Seventh round.** User confirmed the automated-stress-test-passing fix above did *not* fix the
felt experience on a real drag — specifically clarified via `AskUserQuestion` that it's the OS
window-border resize (not the splitter) that still freezes for multi-second stretches, on plain
`basic_window.py` (M0, no widget content — rules out chrome rasterization as a factor for this
particular report). Tried one more targeted, technically-motivated fix: `recreate_swapchain` and
`set_viewport_frame`'s recreation path both called `device_wait_idle()` — waits for *every*
submission on the whole device queue, not just this app's own. On a shared integrated GPU, that
includes DWM's own live-resize-preview compositing work while the user is actively dragging a
window border, which a synthetic `SetWindowPos`-driven test doesn't trigger (no real interactive
drag gesture) — plausible explanation for why automated tests kept passing while the real
experience didn't. Replaced both with `wait_for_fences(&self.in_flight, ...)` — scoped to only
this app's own outstanding GPU work, equally correct for the resource-lifetime guarantee actually
needed (nothing about to be destroyed can still be referenced once our own frames-in-flight are
done), without waiting on unrelated device-wide work.

**This one is unverified even by me** — my own automated re-test of it (rapid synthetic resize
against `basic_window.py`) itself took long enough to blow through a 30s tool timeout before
finishing on its own (no crash, no error). Combined with everything else observed this session —
synthetic input not registering at all early on, the user independently confirming pristine
`basic_window.py` *startup* (not even resize) was slow under heavy system load, and now this same
kind of highly-variable timing recurring for the identical operation across runs — the working
theory going into the next round is that **this is resource contention on this specific machine
(multiple VS Code windows, Discord, several concurrent Claude Code agent processes, all sharing
one AMD 840M integrated GPU), not a remaining defect in fastgui's render loop.** Asked the user to
close background load and retest as a diagnostic before any further Vulkan-level changes — if
resize is smooth with a quiet system, there's nothing left to fix here; if it's still bad, the
next real lead is something GPU/driver-specific (e.g. present-mode choice, or this driver's
swapchain-recreation cost specifically), not event-loop scheduling, and is worth investigating
with that framing rather than continuing to guess architecture changes. **Do not keep iterating
on `app.rs`/`renderer.rs` scheduling logic without new evidence** — by this point four
substantively different, individually-reasoned fixes have gone in (drag throttle, resize/drag
throttle split, resize-skips-recreation-until-changed, device-wide-wait → fence-scoped-wait), and
the felt experience hasn't measurably changed according to the user, which is itself evidence the
bottleneck is elsewhere.

**Fifth round — resolved, and it wasn't a code bug.** Still zero perf output, and the user
reported the *pristine, untouched* `basic_window.py` (M0 — no widgets, no splitter, no drag
handling, nothing touched this session) was **also** slow to open and use right now. That
conclusively rules out anything in `crates/fastgui-render-vk/src/app.rs` — the slowness predates
`Window.run()` even getting to `resumed()`. Root cause: general system load on the user's machine
at the time (multiple VS Code windows, Discord, Task Manager, and several concurrent Claude Code
agent processes all competing for the same AMD 840M integrated GPU/CPU), not this project's code.
Removed all the temporary timing instrumentation (`eprintln!`s, the `cursor_moved_*` counters) —
it did its job (ruled out a real bug in five rounds instead of an unbounded number), and there's
no reason to ship debug scaffolding once the question it existed to answer is settled. Kept the
two genuine fixes from this whole saga: `update_cursor_icon` (confirmed working by the user) and
the synchronous-but-`DRAG_RENDER_INTERVAL`-throttled render in `CursorMoved` while dragging (bumped
back up to a plain 60fps cap now that we know per-call cost is a non-issue — the earlier 30fps
cap was a blind mitigation for a problem that turned out to not be about render cost at all).
**Still not independently re-confirmed smooth by the user under normal system load** — reasonable
next step for anyone picking this up, but the mechanism (why `request_redraw()` alone doesn't
work — `WM_PAINT`-class messages losing priority to `WM_MOUSEMOVE`) is verified against winit's
actual source, not guessed, so there's no open question left to chase here, just a "confirm it
feels right" check.

**Eighth round.** User pasted real Vulkan validation-layer output (this build runs with
validation enabled): repeated `vkDestroySemaphore`/`vkDestroySwapchainKHR` "currently in use by
VkQueue" errors — genuine undefined behavior, not a red herring. Root cause: the Sixth round's
`wait_for_fences(&self.in_flight, ...)` "optimization" (replacing `device_wait_idle()` to avoid
contending with DWM) was actually *wrong*, not just unhelpful — fences only track when a
submitted command buffer's execution finishes, not when the subsequent `vkQueuePresentKHR` (a
separate operation on the same queue, using the semaphore/swapchain we were about to destroy) has
completed. Fixed by using `queue_wait_idle(self.queue)` instead (`recreate_swapchain` and
`set_viewport_frame` in `crates/fastgui-render-vk/src/renderer.rs`) — the spec-correct primitive
that actually covers present completion, at the same blocking cost as `device_wait_idle` for this
single-queue renderer. **Confirmed fixed**: rebuilt, reran the same stress test, stderr empty, no
validation errors. **But the resize was still just as slow** — my own re-test of the identical
burst against `basic_window.py` took over 45s to finish. So the fence-based change was a real bug
worth finding and fixing, but it was never what caused the reported slowness.

Tried the next concrete, testable lead: switched `recreate_swapchain`'s present-mode selection
from "`MAILBOX` if available, else `FIFO`" to unconditional `FIFO` (removing the extra
swapchain-image/present-replacement bookkeeping `MAILBOX` needs), on the theory that present-mode
choice affects recreation cost on some integrated-GPU drivers, not just present latency. Measured
directly this time (not just eyeballed): **55.9 seconds** for a single swapchain recreation to
settle after a 10-event resize burst, correct synchronization, `FIFO`, no other system load (user
had already closed background apps per the Seventh round's ask, which itself ruled out simple
contention). So: not present mode either.

**Ninth round — root-caused for real, and it's outside the app.** Given recreation cost alone is
~56 seconds regardless of synchronization primitive or present mode, throttling how *often* it
happens was never going to be enough — even one recreation triggered mid-drag freezes the whole
app for the better part of a minute. Changed strategy entirely:
`WindowEvent::Resized`'s handler (`crates/fastgui-render-vk/src/app.rs`) no longer touches the
renderer *at all* — purely `self.width = ...; self.height = ...;` bookkeeping. The swapchain is
left at its old size for the *entire* duration of a drag (the window shows a stretched/stale
image — a real visual compromise), and only gets caught up once via `render_now`'s existing
`last_applied_size` check, whenever the self-perpetuating `RedrawRequested` loop next gets a turn
— which happens naturally once `WM_SIZE` stops flooding the queue (drag paused or ended). Removed
`RESIZE_RENDER_INTERVAL`/`last_resize_render` entirely (dead code once nothing calls `render_now`
synchronously from `Resized` anymore).

Verified this achieves its actual goal — **the app itself stays responsive throughout a live
resize burst** (`Get-Process`'s `Responding` stayed `True` for all 10 events, checked after each
one, not just at the end) — but posting those 10 synthetic `SetWindowPos` calls still took
**56.8 seconds** to return, even though our own message handling for every one of them is now two
field assignments. That was the deciding data point: `SetWindowPos` blocks the *calling* process
until the target window has processed certain resize-related messages, so if it's slow even when
our own handler does nothing, the bottleneck isn't in fastgui's event handling at all. Confirmed
by controlled comparison: the **identical** `SetWindowPos` burst against a freshly-launched
**Notepad** window (no Vulkan surface, otherwise unrelated) completed in **538ms**. Same OS, same
DWM, same machine, same exact automation — 100x difference, and the only variable is "does this
window have a live Vulkan swapchain attached."

**Conclusion: this is a DWM ↔ Vulkan-surface interaction issue specific to this machine's
graphics stack (driver, or possibly a virtualized/remoted display path — this session has shown
other anomalies consistent with that, e.g. synthetic mouse input never registering earlier in the
session), not a bug in fastgui's rendering code.** Every avenue actually inside the app's control
has been tried and measured, not guessed: correct Vulkan synchronization (fixed a real bug along
the way), present-mode choice, event-loop scheduling, and fully deferring recreation until the
drag settles. None of them touch the ~56-second figure, and the Notepad comparison rules out
anything left in `app.rs`/`renderer.rs` from being the cause. **Recommendation for anyone picking
this up:** don't keep iterating on this project's code for this specific symptom — instead check
GPU driver version/update, whether this session is running through any form of remote/virtualized
display, and whether a minimal non-fastgui Vulkan+winit sample shows the same live-resize
behavior on this machine (would confirm it's Vulkan-on-this-machine in general, not something
specific to this project's renderer setup).

**Tenth round — actually solved.** User pushed back on accepting "this is unfixable" and asked to
keep digging, and asked what Notepad uses (`dcomp.dll`/`d3d11.dll`/`dxgi.dll` — no Vulkan at all;
telling, since DirectComposition is literally the tech DWM's own compositor is built on, so it's
a first-class citizen in a way a Vulkan swapchain isn't, but not proof by itself). Checked whether
this machine is a VM (my leading theory at the time) — it isn't: real Lenovo laptop (model 83JR),
real AMD BIOS, real AMD Radeon 840M driver (32.0.22042.31003), ruling out virtualization as the
explanation.

Built `crates/fastgui-render-vk/examples/resize_probe.rs` — a standalone, minimal winit+Vulkan
program with **none** of fastgui's own code, to determine whether this is general to Vulkan on
this machine or specific to fastgui. This was the actual breakthrough tool; kept in the repo as a
permanent diagnostic (`cargo run --example resize_probe -p fastgui-render-vk`). Findings, in
order:

1. First version: bare swapchain create/destroy on resize, `ControlFlow::Wait`, no render loop.
   Identical 10-event `SetWindowPos` burst: **756ms**. Individual `recreate_swapchain` calls
   measured **20-40ms**, `queue_wait_idle` in **microseconds**. This flatly refuted "Vulkan
   swapchain recreation is just slow on this machine/driver" — it isn't, at all, in isolation.
2. Extended the probe to match fastgui's actual architecture: self-perpetuating
   `ControlFlow::Poll` render loop (`request_redraw()` called at the end of every
   `RedrawRequested`) plus a real `acquire_next_image`→`queue_submit`→`queue_present` cycle each
   frame (previously the probe didn't present anything at all). Same burst: **18.7 seconds**.
   Confirmed the differentiator was the continuous render loop, not swapchain recreation itself.
3. Tried capping the render loop to 60fps (instead of fully uncapped): **18.6s** — no
   improvement. Rate wasn't the variable.
4. Tried skipping `render_frame` entirely whenever a resize was pending (never present and be
   mid-resize at the same time): **18.5s** — still no improvement. Presence of the render loop
   wasn't the variable either.
5. Added per-event timestamped logging (`Resized` events and periodic `RedrawRequested` counts).
   This was the actual answer: before any resize, `RedrawRequested` fires **~30,000 times/sec**
   (an expected tight spin loop). The instant a resize starts, that rate collapses to **roughly 1
   event per 2 seconds**, and each subsequent `Resized` event arrives ~2 seconds after the
   previous one *even though `SetWindowPos` calls were issued 20ms apart* — i.e. `SetWindowPos`
   itself (a synchronous OS call) was blocking for ~2s each time. 10 recreations × ~2s ≈ the
   18.5s observed. Our own Vulkan API calls (measured via `Instant`, printed) stayed at
   20-40ms throughout — the ~2s cost is **entirely external to our process**, overwhelmingly
   likely DWM redoing expensive bookkeeping every single time a swapchain is created/destroyed,
   confirmed to only manifest once an active render loop (submit+present) exists alongside it.
6. **The fix**: debounce, not throttle. Changed the probe (then ported to
   `crates/fastgui-render-vk/src/app.rs`) so `Resized` *only* updates `width`/`height`
   bookkeeping and records `last_resize_event = Instant::now()`; the swapchain only actually gets
   recreated once `RESIZE_DEBOUNCE` (150ms) has passed *without* a new `Resized` — i.e. once per
   resize **gesture**, not once per `WM_SIZE`. During an active drag the window shows a
   stretched/stale image (a real, accepted visual compromise) but the app stays fully responsive;
   the ~2s external cost gets paid exactly once, after the drag settles, instead of once per
   pixel of movement.

**Verified on the actual app, not just the probe**: the identical 10-event `SetWindowPos` burst
against real `basic_window.py` (M0) dropped from confirmed **56+ seconds** to **801ms**; against
`dock_layout.py` (M6, real widget content) to **807ms**. Both rebuilt and reverified clean on
`.venv` and the free-threaded `env`, correct final layout, empty stderr.

**Eleventh round — the remaining ~2s settle pause, eliminated too.** User correctly pointed out
the debounce fix only made the *drag itself* smooth — the one deferred recreation, paid once
after the drag settles, still cost ~2s (confirmed: the "801ms" figure above only measured how
fast the burst could be *posted*, not how long the window took to actually catch up to the final
size afterward — a real gap in the round-ten verification, caught by the user asking a sharper
question rather than accepting "better" as "done"). Extended `resize_probe.rs` with two scripted,
no-drag-required experiments (run automatically 3s and 6s after startup): recreate at the *same*
size (a) with the normal `old_swapchain` "smooth handoff" hint set in `VkSwapchainCreateInfoKHR`,
and (b) without it (destroy the old swapchain first, then create fresh with `old_swapchain:
null`). Both recreations measured identically fast internally (~45ms), but only (a) — the
handoff-hint version — caused `RedrawRequested` to stall for ~2.5s afterward; (b) caused no
measurable stall at all, `RedrawRequested` kept ticking at its normal rate straight through.
**That isolated it precisely**: requesting the graceful old→new swapchain handoff is specifically
what triggers the expensive external (almost certainly DWM-side) synchronization — not swapchain
recreation in general.

Applied to the real renderer (`crates/fastgui-render-vk/src/renderer.rs`'s `recreate_swapchain`):
destroy the old swapchain first, then create the new one with `old_swapchain: null`, instead of
the usual keep-old-alive-as-a-hint-then-destroy-after pattern. Trade-off: a brief window with no
valid swapchain (a theoretical one-frame flicker) instead of multi-second blocking — acceptable
given resize is already debounced to happen once per gesture. Also reverted the present-mode
change from the eighth round back to preferring `MAILBOX` (never actually validated as helping;
superseded by finding the real cause).

**Verified end-to-end this time** — measuring the *full* cycle (first resize event to the window
actually showing the correct final size, not just how fast events could be posted):
`basic_window.py` settled in **855ms** total; `dock_layout.py` in **774ms**; free-threaded `env`
build of `dock_layout.py` in **768ms**. All three: correct final layout/size, empty stderr. No
perceptible pause at all anymore, during the drag or after releasing it. **This is the actual
resolution** — the eleven-round trail above (including the four dead ends) is preserved as-is
since it's a genuinely useful case study in how to isolate an externally-imposed cost using a
minimal reproduction, but nothing further should need to change here absent a regression.

### 6A status update: tabs, drag-to-rearrange, and floating panels all landed

Second M6 session. User asked to finish the rest of 6A (steps 3-5). Floating-panel approach was
an explicit decision point per the plan below — asked the user directly rather than assume: chose
**simulated overlay within the single window**, not real second OS windows per floating panel
(the other option), specifically to avoid reworking `fastgui-render-vk`'s one-`Window`/
one-swapchain-per-`App` assumption — a much bigger, riskier change than this session had appetite
for, especially right after the M6-status resize investigation above demonstrated how fragile
that swapchain/window layer already is on this machine.

**Tabs (6A.3).** New self-contained `WidgetKind::TabBar` (bakes its own header-segment + text
rendering directly, like `Slider` bakes its track+thumb — deliberately *not* composed from child
`Label` nodes, since that's exactly the pattern that caused `Panel`'s title-bar text-clipping bug
earlier in M6; baking it in sidesteps the whole class of bug). `content_ids` on the `TabBar`
point at sibling content-wrapper nodes, one per tab; clicking a header segment
(`fastgui-render-vk::app::handle_tab_click`) sets `active` and flips the clicked wrapper to
`Display::Flex`/every other one to `Display::None` via a new `WidgetTree::set_display` (taffy
does support `Display::None` — confirmed before relying on it, not assumed). Python:
`fg.Tabs(panels: list[Panel])` — deliberately reuses `Panel` (pulling just `.title`/`.content` via
a new `Panel::title_and_content`, *not* attaching each member's own title bar, which would show
it twice) rather than inventing a separate `TabItem` type. `Tabs` composes into `DockArea`/
`Splitter` exactly like `Panel` does — no special-casing needed there, confirmed by passing one
straight to `dock.add_panel(...)` in `python/examples/tabs_demo.py`.

**Drag-to-rearrange (6A.4).** Scoped deliberately, not silently: only a standalone `Panel`'s
title bar is a drag *source*; any region (`Panel` or `Tabs`) is a valid drop *target*. A `Panel`
already inside a `Tabs` group isn't itself draggable (no per-tab title bar to grab — `Tabs`
renders one combined header, not individual grabbable ones), and dropping onto an existing
`Tabs`' center isn't supported (only forming a *new* two-tab group by dropping one `Panel` onto
another `Panel`). Both gaps are documented in `DockArea`'s own docstring, not just here.

Mechanism: every `Panel`/`Tabs` gets a stable `region_id` (`fastgui-py`'s `NEXT_REGION_ID`,
assigned once at construction — deliberately *not* reusing the widget-tree node id, which gets
reassigned on every rebuild) — exposed as `.id`. `Panel`'s title bar became its own
`WidgetKind::PanelTitleBar` (self-contained rendering, same reasoning as `TabBar` above).
`Container` gained `region_id: Option<u64>` (`Some` only on a `Panel`'s/`Tabs`' own outer
container) and a new `WidgetTree::find_region_at` walks the tree for the one containing a point —
deliberately *not* the existing `hit_test` (topmost-wins z-order), since regions never overlap
(each occupies a distinct rect via the `Splitter` tree) so any match already *is* the match, no
z-order tie-break needed. A new `DropZone` enum (`Center`/`Left`/`Right`/`Top`/`Bottom`) with a
single `DropZone::classify(rect, x, y)` shared by both the live hover-preview and the actual
drop-commit, so they can never disagree about what a given cursor position means.

On mouse-down on a `PanelTitleBar`, `app.rs` starts tracking `dragging_panel_title`; every
`CursorMoved` recomputes `hover_region` (which region, which zone) and forces a re-rasterize with
a translucent drop-indicator overlay (`ChromeRenderer::rasterize` gained a `drop_indicator`
parameter, drawn as the last step so it's always on top — half the target region for an edge
zone, matching what actually happens on drop, not just "near this edge"). On release, if
`hover_region` is `Some`, the dragged panel's `on_drop` callback fires with
`(dragged_region_id, target_region_id, zone)` — the *same* callback on every `PanelTitleBar` a
`DockArea` built (`Panel.set_rearrange_handler`, called by `DockArea.add_panel`); a standalone
`Panel` never has one bound, so dragging it is a harmless no-op. Deliberately never mutates the
widget tree directly on the Rust side — the callback is Python's `DockArea._on_rearrange`, which
does the actual tree surgery, then re-attaches via `Window.set_content` again.

This needed a real rethink of `DockArea` itself: it used to nest live `Splitter`/`Panel` Rust
objects directly, which Python can't introspect (no getters on `Splitter.first`/`.second`) — no
way to find-and-move a node in that. Rewritten to keep its own plain-dict tree
(`{"kind": "leaf", "widget": ...}` / `{"kind": "split", "direction", "ratio", "first", "second"}`)
as the actual source of truth, with `_fastgui_widget` building a fresh `Splitter`/`Panel` tree
from it on every read (including after a rearrange) rather than mutating live Rust objects.
`_extract`/`_find_leaf`/`_replace` are the tree-surgery primitives; `_on_rearrange` combines them
into "detach dragged leaf (collapsing its old parent split if needed), then either wrap the
target in a new split or merge into a new `Tabs`". Also needed a route for `DockArea` to trigger
a *second* `set_content` call on its own after a rearrange (nothing in the existing architecture
gave a composite widget a way back to "its" window) — `Window.set_content` now does
`widget.setattr("_fastgui_window", self)` (silently no-ops on plain pyclass widgets that don't
support arbitrary attributes), and `_on_rearrange` calls `window.set_content(self)` again once
it's done mutating.

**Verification**: this session's synthetic-input limitation (established earlier in M6 — neither
`SendInput`/`mouse_event`/`PostMessage` register a click, confirmed even against the pristine M4
`Button`) means the actual drag *gesture* — mouse-down on a title bar, drag, see the overlay,
release — could not be driven end-to-end by this session, same as tab-click in the paragraph
above. But the part that was actually most at risk of a logic bug — the tree-surgery
(`_extract`/`_find_leaf`/`_replace`/`_on_rearrange`) — needed no GUI at all to test directly: a
standalone script (not part of the repo, scratch-only) called `dock._on_rearrange(...)` directly
with synthetic region ids and asserted on the resulting tree shape. All cases passed: dragging a
panel onto another's edge produces the correct split (including correctly collapsing the
dragged-from split), dragging onto a center produces a new `Tabs`, no leaves lost or duplicated
in either case, `_fastgui_widget` builds without raising after each mutation, dropping onto self
is a no-op, and an unknown dragged id is safely ignored. That leaves the Rust-side input/hit-test
wiring (mirrors the already-interactively-proven `Splitter`/`Slider` drag pattern closely) and the
Rust→Python callback dispatch as the only genuinely untested links — reasonable confidence, not
full confidence; **worth the user confirming an actual drag once, when convenient.**

**Floating panels (6A.5).** `Window.add_floating_panel(panel, x, y, width, height)` — deliberately
a `Window`-level method, not a `DockArea` one, since a floating panel is conceptually independent
of any dock tree (matches the roadmap's own framing: "an always-on-top overlay region simulated
within the existing single window"). Needed taffy's `Position::Absolute` support (confirmed it
exists and behaves as needed before relying on it, not assumed) — `StyleParams` gained an
`absolute: Option<(f32, f32)>` field; when set, `to_style()` emits `Position::Absolute` with
`inset.left`/`inset.top` at that `(x, y)` instead of normal flex flow. Taffy's default
`Position::Relative` on every other node (including root) already makes the tree root the
"closest positioned ancestor" for absolute children without needing anything explicit, so
attaching a floating panel as a child of root positions it in plain window-relative physical
pixels — exactly what was needed, confirmed by the first screenshot landing at the exact
requested `(x, y, width, height)`.

Dragging one moves it directly rather than going through the drop-zone machinery: `PanelTitleBar`
gained `floating: bool` and `container_id: Option<WidgetId>` (its own parent `Panel` container's
id — needed because *the container*, not the title-bar leaf itself, is what has the absolute
position to update). `container_id` is backfilled generically in `attach`'s plain child-attaching
branch (not special-cased per caller): any `PanelTitleBar` child automatically learns its
just-created parent's id, since that's always true regardless of whether the parent `Panel` is
docked or floating. A new `WidgetTree::set_position` updates `inset` live, mirroring
`set_flex_grow`/`set_display`'s existing pattern; dragging is a new `dragging_floating_panel`
state in `app.rs`, structurally the same shape as `dragging_panel_title` but simpler (no
hover-region/zone computation, no drop commit — just track cursor-minus-grab-offset and call
`set_position` every move).

Floating panels needed to survive a `DockArea` rearrange calling `set_content` again internally
(which resets the *entire* tree) — `Window` now remembers every panel ever passed to
`add_floating_panel` (with its live `x`/`y`/`width`/`height`) and `set_content` re-describes and
re-attaches all of them after rebuilding the main content, every time, not just the first.
Explicit scope cuts, not oversights: not resizable (no resize-handle hit-testing implemented),
and not re-dockable back into a `DockArea` (dragging one always just moves it, never checks
`hover_region`) — both are reasonable, contained follow-ups if ever wanted, not required for this
to be a genuinely useful feature as-is.

**Verification**: static rendering confirmed correct on both venvs (`python/examples/
floating_panel_demo.py` — exact requested position/size/colors, layered correctly over the
docked content underneath, empty stderr). The actual drag gesture has the same untested-by-this-
session status as the other two features above, for the same reason (no working synthetic input
this session) — **also worth confirming manually.**

All three new examples (`tabs_demo.py`, `dock_rearrange_demo.py`, `floating_panel_demo.py`) plus
a full regression pass of the pre-existing ones (`dock_layout.py`, the M0-M4 examples) were
rebuilt and screenshot-verified clean on both `.venv` and the free-threaded `env` after every
change in this session, not just once at the end.

**Follow-up, same session: whole-`DockArea` edge drops.** User tried the drag-to-rearrange and
confirmed the core mechanism works, but couldn't move a panel to become a new row/column spanning
the *entire* dock area (e.g. a full-width strip above every other panel) — asked a clarifying
question first (real bug in vertical-zone detection, vs. a missing feature) rather than guess,
since the fix differs completely between those two; confirmed it was the latter. The original
design only ever supported "split relative to whichever specific panel the cursor happens to be
over" (`DropZone::classify` + `find_region_at`, both scoped to one region's own rect) — there was
no way to target "the whole layout" at all, so dropping on one panel's top edge only ever added a
strip the width of *that* panel, never the full window.

Fixed by adding a second, higher-priority check in `update_panel_drag_hover`
(`crates/fastgui-render-vk/src/app.rs`): before falling back to per-panel `find_region_at`, check
whether the cursor is within `OUTER_EDGE_MARGIN` (24px) of the *window's* own edge (not a panel's
— using `self.width`/`self.height` directly), and if so report a reserved `ROOT_REGION_ID` (`0`,
never assigned to a real panel — `fastgui-py`'s id counter starts at 1) instead of a specific
region, with the zone computed against the whole window rect the same way
`DropZone::classify` computes it for a single region. `DockArea._on_rearrange`
(`python/fastgui/__init__.py`) special-cases `target_id == 0`: instead of `_find_leaf` +
`_replace` on one leaf, it wraps the *entire* extracted remainder tree in a new split, with the
newly-placed panel getting a thin default 0.25 share (matching `add_panel`'s own `size` default —
an edge strip should default to thin, not an even 50/50 split) rather than the 0.5 used for a
per-panel split.

**Verified with unit tests before rebuilding**, extending the same direct-tree-manipulation
script used for the original rearrange logic (no GUI needed for this part either): dragging a
panel onto the window's top edge, when only one of two side-by-side panels was directly under it,
correctly produces a full-width top row with *both* other panels underneath (not just the one
that happened to be there) — checked the resulting tree shape directly (top-level split, dragged
panel on the correct side, the *other* side containing both remaining panels, correct 0.25/0.75
ratio depending on which edge), not just that it didn't crash. Also checked the symmetric bottom
case. All pass, on both venvs. Screenshot regression pass (`dock_rearrange_demo.py`) still clean.
Same caveat as before: the tree logic is directly verified, but the actual cursor-near-window-edge
detection during a live drag is not — **worth confirming together with the other two "please
drag this for real" items above.**

**Follow-up, same session: ungrouping `Tabs`.** User asked whether a merged `Tabs` group could be
pulled back apart — it couldn't (this was one of the two explicitly-scoped-out gaps from the
original 6A.3/6A.4 pass). Implemented by making a `TabBar`'s individual tab segments drag sources
too, not just click targets, reusing the *exact same* `dragging_panel_title`/`hover_region`/
`handle_panel_drop` machinery `PanelTitleBar` already drives (see `fastgui-render-vk::app`) rather
than inventing a parallel mechanism: `WidgetKind::TabBar` gained `panel_ids: Vec<u64>` and
`on_drop: Vec<Option<PanelDropCallback>>`, parallel arrays to `titles` (one entry per tab) — each
tab's `panel_id` is its member `Panel`'s own stable region id (set once at construction, untouched
by ever being merged into a group) and `on_drop` is that same `Panel`'s existing
`rearrange_handler` (already bound by `DockArea.add_panel`, previously just never read once a
panel became tabbed). `handle_mouse_press`'s `TabBar` arm now looks up which segment was pressed
and seeds `dragging_panel_title` with that tab's `panel_id` (still also calling
`handle_tab_click` — a plain click is just a drag that never lands on a valid `hover_region`, the
same "no separate threshold" design `PanelTitleBar` already used). `handle_panel_drop` was
generalized to pull its callback from either a `PanelTitleBar` or, if `panel_ids` contains the
dragged id, the matching `TabBar` slot.

Python-side, `_extract` (`python/fastgui/__init__.py`) now looks *inside* a `Tabs` leaf for a
member `Panel` matching the dragged id (previously only matched a leaf's own `.id`, which for a
`Tabs` leaf is the *group's* id, not any member's): removes that panel from the group's member
list and, if exactly one member remains, collapses the group back into a plain `Panel` leaf rather
than leaving a pointless one-tab group around. `Tabs` gained `panels`/`active` getters
(`fastgui-py::widgets::Tabs`) so Python can read a live group's membership without Rust needing to
expose tree-surgery primitives itself. Growing an *existing* group past 2 members by dropping onto
its center is still unsupported (unchanged, separate gap — `_find_leaf`/`_on_rearrange`'s
center-zone branch still only handles merging two standalone panels into a *new* group).

Verified with extended unit tests (no GUI): merge two panels into a group, confirm both panels
still resolve via a `Tabs`-aware presence check, drag one back out onto a third panel's edge,
confirm the group fully dissolved back into a plain `Panel` leaf (not a 1-tab group), confirm the
dragged-out panel landed correctly and nothing was lost — plus the click-onto-self and
unknown-dragged-id no-op cases repeated with the target living inside a group. All pass, on both
venvs. Screenshot-verified the merged-group rendering path itself still renders both tab segments
and their labels correctly (a throwaway script building a group via `_on_rearrange` directly, the
same code path a real center-drop takes) and reran the full existing-example regression pass
(`tabs_demo.py`, `dock_rearrange_demo.py`) clean on both venvs — the new per-tab `on_drop`/
`panel_ids` wiring didn't disturb ordinary tab-bar rendering or click-to-select. Same
can't-simulate-a-mouse-drag caveat as the rest of 6A's interactive gestures: the tree/render logic
is directly verified, the actual drag-a-tab-out gesture itself isn't — **a fourth thing worth
confirming interactively** alongside tab-click, floating-panel drag, and root-edge drop.

**Build gotcha hit while rebuilding for this fix, worth remembering:** `maturin develop` resolves
"the current virtualenv" by convention (finds a `.venv`-named folder near the manifest) rather
than by which `Scripts\maturin.exe` you actually invoke — running `env\Scripts\maturin.exe
develop` silently built and installed into `.venv` *again* instead of the free-threaded `env`
(no error, just wrong target — `env`'s copy went stale). Always set `VIRTUAL_ENV` explicitly when
targeting `env`, e.g. `VIRTUAL_ENV=<path>\env env\Scripts\maturin.exe develop ...` — confirmed via
maturin's own "Found CPython 3.14t at ...\env\Scripts\python.exe" banner line, which silently said
`.venv` (no `t` suffix) both times before this was caught.

Viewport-inside-a-dock-panel, idle `ControlFlow::Wait`, `.gitignore`, and headless tests
(`fastgui-core` + `python/tests/test_dock_area.py`) landed after 6A. Still not started: wheel
packaging (6B), polish pass (6D). Docs (6C) are done — see its own section below. **6A itself (splitter, static dock, tabs — including ungrouping,
drag-to-rearrange — including whole-area edge drops, floating panels) is now feature-complete**
modulo the "confirm interactively when convenient" items above and the remaining explicitly-
scoped-out gaps (dropping onto an existing `Tabs`' center, and floating-panel resize/re-dock)
noted in `DockArea`'s and `Window.add_floating_panel`'s own docstrings.

### 6A. Docking/panel system (full scope)

Build bottom-up; each step is independently useful and testable, so verify as you go rather
than building all of it before running anything:

1. **Splitter primitive first.** A draggable divider between two child regions (row or
   column), resizing both sides as it's dragged. This is the foundation everything else sits
   on, and is fully useful on its own (e.g. a viewport next to a control sidebar) even before
   any docking logic exists.
2. **Static `DockArea`.** A fixed split-tree of titled `Panel`s (title bar rendered, no
   drag-to-rearrange yet, no tabs, no floating). Data structure suggestion — a binary
   split-tree, the standard approach (egui_dock, Dear ImGui's docking branch, Qt's dock
   system all model it similarly):
   ```rust
   enum DockNode {
       Split { direction: Row | Column, ratio: f32, first: Box<DockNode>, second: Box<DockNode> },
       Tabs { panels: Vec<PanelId>, active: usize },
   }
   ```
   Panel content is either a widget subtree (reuse M4's `WidgetTree`/`WidgetKind`) or a
   `Viewport` — this likely means generalizing `WidgetKind`/`ActiveContent` further so a dock
   leaf can point at either.
3. **Tab groups.** Multiple panels sharing one region; click a tab to switch which is active.
4. **Drag-to-rearrange.** Drag a panel's title bar; hit-test drop zones (edges of the
   `DockArea` or of another panel, with a visual drop-indicator overlay); reflow the
   `DockNode` tree on drop. Needs distinguishing "click" from "drag started" on the title bar.
5. **Floating panels — decide the approach explicitly before starting.** Either (a) a second
   real winit window per floating panel, which means revisiting `app.rs`'s current
   one-`Window`-per-`App` assumption and `VulkanRenderer`'s 1:1 assumption with a single
   surface/swapchain (real, non-trivial architecture change), or (b) an always-on-top overlay
   region simulated within the existing single window (much less plumbing, but a real UX
   compromise vs. true OS-level floating windows). This is the highest-risk, highest-effort
   part of the whole docking system — if time-constrained, it's reasonable to land steps 1–4
   solidly and treat floating panels as a explicit follow-up, flagging that clearly rather
   than rushing it.

Suggested Python API shape to build toward:
```python
dock = fg.DockArea()
dock.add_panel(fg.Panel(title="Viewport", content=viewport_widget), region="center")
dock.add_panel(fg.Panel(title="Controls", content=controls_box), region="right", size=0.25)
window.set_content(dock)
```

### 6B. Wheel packaging (abi3 + free-threaded matrix)

Concrete and fully achievable/verifiable on this machine (at least the Windows leg):

- Configure `fastgui-py/Cargo.toml`'s `pyo3` dependency with an `abi3-pyXY` feature for the
  GIL-enabled build (one wheel covers all supported CPython minors via the stable ABI); the
  free-threaded build stays non-abi3 per-minor-version (`cp313t`, `cp314t`, ...) since `abi3t`
  needs Python 3.15+ (check current 3.15 release status when doing this work — it was still
  in beta as of this project's early sessions).
- Update `pyproject.toml`'s `[tool.maturin]` section accordingly.
- Write `.github/workflows/wheels.yml` (or equivalent): build matrix over OS × {abi3
  GIL-enabled, cp313t, cp314t}. Can only locally verify the Windows leg, but still write
  correct config for macOS/Linux (e.g. via `maturin-action`).
- Locally verify: `maturin build --release` (not `develop`) produces an installable wheel that
  works without dev-mode symlinking, for both the abi3 and free-threaded configurations.
- Add a `.gitignore` (`target/`, `.venv/`, `env/`, `*.pyd`, `__pycache__/`, etc.) — currently
  missing entirely, and there's no git repo yet either. Whether to `git init` is worth
  confirming with the user rather than assuming.

### 6C. Docs — done

- `README.md` at the project root: what fast-gui is and why (GPU-native, free-threaded-safe
  design), a runnable quickstart, install instructions (including the `maturin develop`
  target-venv gotcha from the ungroup follow-up above), an example table (one row per script
  under `python/examples/`, pulled from each script's own docstring rather than re-describing
  them from scratch so the two can't drift silently), a widget API summary pointing at
  `__init__.pyi` for full signatures, and a known-limitations section (CUDA path unverified —
  see M3, macOS not implemented — see M5, the remaining `DockArea`/floating-panel gaps, no
  automated tests yet, mutually-exclusive `Viewport`/chrome).
- `docs/ARCHITECTURE.md`: crate-by-crate tour, then the threading model section this item was
  really about — `command_channel` (unbounded, ordered, fire-and-forget mutations) vs.
  `FrameSlot` (latest-wins mailbox, drops superseded frames on purpose) vs. `oneshot_channel`
  (the one synchronous-reply case, `create_cuda_surface`) vs. `Readback` (mutex-cached last
  value for sync getters) — what each guarantees and why a single primitive doesn't cover all
  four cases. Also covers the widget-tree describe/attach split, why some widgets
  (`Slider`/`TabBar`/`PanelTitleBar`) bake their own rendering instead of composing children,
  and a short "adding a new widget" checklist for future contributors.
- Did not expand Rust module-level doc comments beyond what already existed — the existing
  comments (many written specifically to explain *why*, not just *what*, during M0–M6) turned
  out to be the right level of detail already; `docs/ARCHITECTURE.md` is the higher-level
  complement to those, not a replacement.

### 6D. Polish pass

- Review the "known simplification" comments accumulated across M2–M4 (listed above) and
  decide which are worth addressing now vs. documenting as deliberate future work.
- Headless tests now exist (`cargo test -p fastgui-core`, `python -m unittest` for `DockArea`
  tree surgery). Still no GPU/window integration suite — screenshot + interactive checks remain
  the way to verify rendering.
- Sanity-pass error messages across all `PyErr` sites for consistency.
- Note clearly (in the README) that `.venv`/`env` are local and machine-specific — a fresh
  session or a different machine needs to redo the M0 toolchain setup (Rust, MSVC, Vulkan
  SDK, both venvs) before any of this builds.

## How this project has been built (read before continuing)

- **No per-milestone re-planning ceremony** — implement directly using the architecture
  established in the original full plan (Rust + PyO3, native Vulkan/Metal backends, `taffy`
  for layout, `cosmic-text` + `tiny-skia` for chrome). Use judgment on the many small
  decisions within a milestone; don't stop to ask about each one.
- **Verify everything by actually running it**, not just `cargo check`. Every milestone in
  this project shipped with at least one bug that only surfaced when the example was actually
  launched, screenshotted, and (for M4) driven with simulated input — type-checking alone
  would have missed all of them (the M4 deadlock, the sRGB color bug, both M3 Vulkan
  validation errors). Continue this discipline for M6: build the docking system, launch it,
  screenshot it, drag things, confirm panels actually reflow — don't declare a step done from
  a clean compile.
- **Rebuild and check both venvs** (`.venv` and `env`) when a change touches anything under
  `fastgui-py`/`fastgui-render-vk`/`fastgui-core` — free-threaded-specific issues don't always
  show up under the regular GIL build.
