# Architecture

This is a tour of how the crates fit together and why the threading model looks the way it
does. For what's implemented vs. not, and the history behind specific decisions, see
[ROADMAP.md](../ROADMAP.md) — this doc is the stable "how it's built" reference; ROADMAP.md is
the living "what's done and why" log.

## Crate layout

```
crates/
  fastgui-core/          Cross-thread primitives + the widget tree. No rendering, no Python,
                          no platform code — pure logic, shared by every backend.
    src/queue.rs             command_channel (unbounded MPSC), oneshot_channel
    src/readback.rs          Readback<T> (mutex-cached last-committed value, for sync getters)
    src/frame.rs              FrameSlot<T> (latest-wins mailbox), CpuFrame, PixelFormat
    src/widget.rs             WidgetTree (wraps taffy::TaffyTree), WidgetKind (including
                              Viewport), WidgetId, Color, hit_test / find_region_at, DropZone

  fastgui-render/         Renderer trait — the interface a GPU backend implements. Backend-
                          agnostic. In practice `fastgui-py` cfg-picks `fastgui-render-vk` or
                          `fastgui-render-mtl` rather than dispatching through this trait.

  fastgui-render-vk/      The Vulkan backend (Windows + Linux). Owns the actual OS window and
                          event loop — this is the crate with a `fn main`-shaped entry point,
                          conceptually, even though it's a library.
    src/renderer.rs          VulkanRenderer: instance/device/swapchain setup, dynamic rendering
                              (VK_KHR_dynamic_rendering — no render pass/framebuffer objects)
    src/app.rs                winit ApplicationHandler: drains the command queue, handles
                              input (hit-testing, drag state machines), drives the render loop
    src/command.rs            Command enum: SetClearColor, SetViewport, CreateCudaSurface,
                              MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>)
    src/pipeline.rs           ViewportPipeline: fullscreen-triangle shaders + descriptor set
    src/texture.rs            ViewportTexture: host-visible LINEAR CPU-upload texture
    src/cuda_texture.rs       CudaSharedTexture: exportable image + timeline semaphore (CUDA
                              interop, see below)
    shaders/                  viewport.vert/.frag source + precompiled .spv (checked in, so a
                              plain build doesn't need the Vulkan SDK)

  fastgui-render-mtl/     The Metal backend (macOS). Same role as fastgui-render-vk: owns the
                          OS window and event loop. Selected automatically by fastgui-py on
                          macOS. No CUDA interop path — Viewport.create_cuda_surface raises
                          on this platform; CPU submit_frame is unchanged.
    src/renderer.rs          MetalRenderer: system MTLDevice, CAMetalLayer on winit's NSView,
                              chrome + per-Viewport MTLTexture upload via replaceRegion
    src/app.rs                winit ApplicationHandler mirroring fastgui-render-vk::app
                              (widget/drag/dock logic, layout in points, chrome at backing
                              scale)
    src/command.rs            Command enum: SetClearColor, MutateWidgetTree (no CreateCudaSurface)
    src/pipeline.rs           ViewportPipeline: MSL fullscreen-triangle + sampler
    src/texture.rs            ViewportTexture: StorageModeShared MTLTexture + replaceRegion

  fastgui-chrome/         Widget rasterizer: walks a WidgetTree and draws it into one RGBA8
                          buffer via cosmic-text (text shaping) + tiny-skia (2D rasterization).
                          Layout units are multiplied by a scale factor so HiDPI backends can
                          keep layout in points while the pixmap matches backing pixels. The
                          buffer is uploaded through the same CPU-texture path Viewport frames
                          use — chrome is, from the renderer's perspective, just another frame.

  fastgui-interop-cuda/   Raw CUDA driver API FFI (struct layouts transcribed from NVIDIA's own
                          cuda-python bindings generator), dynamically loading nvcuda.dll via
                          libloading so the crate builds and runs fine with no CUDA installed
                          at all — it only fails at the call site, gracefully, if invoked
                          without a working CUDA driver.

  fastgui-py/             PyO3 bindings — the only crate that knows about Python.
    src/backend.rs            cfg-picks fastgui-render-mtl on macOS, fastgui-render-vk elsewhere
    src/lib.rs                Window, Viewport, CudaSurface pyclasses (create_cuda_surface is a
                              RuntimeError stub on macOS)
    src/widgets.rs             Label/Button/Slider/Box/Splitter/Panel/Tabs pyclasses, the
                              DescribedWidget tree-building/attach machinery

python/
  fastgui/__init__.py, __init__.pyi   DockArea — a pure-Python composite widget (tree-surgery
                                      logic for drag-to-rearrange lives here, not in Rust)
  examples/                           one runnable script per feature, see README.md's table
```

## The threading model

The render thread owns the window, the `WidgetTree`, and all GPU state. It's the thread that
calls `winit`'s event loop — normally, on desktop platforms, this must be a specific OS thread
(the one that created the window), so **it can't be "whichever thread happens to call a
mutator."** Every other thread — including, under free-threaded Python, several Python threads
running genuinely in parallel — talks to it exclusively through message-passing, never through
shared mutable state guarded by a mutex that a caller blocks on. The event loop uses
`ControlFlow::Wait` (not `Poll`): a command send or `Viewport.submit_frame` wakes it through
an `EventLoopProxy` so idle windows don't spin.

Three different primitives exist because three different delivery guarantees are needed:

- **`command_channel`** (`fastgui-core::queue`, an unbounded `crossbeam-channel` MPSC): for
  mutations that must all be applied, in order, but where the caller doesn't want to wait —
  `label.set_text(...)`, `window.set_clear_color(...)`, a dragged panel's drop-rearrange
  callback. Unbounded specifically so a calling thread's send never blocks on the render
  thread's frame cadence. `Command::MutateWidgetTree(Box<dyn FnOnce(&mut WidgetTree) + Send>)`
  is closure-based so adding a new mutating operation never requires a new `Command` variant —
  the closure captures whatever it needs and the render thread just calls it.
- **`FrameSlot<T>`** (`fastgui-core::frame`, a `Mutex<Option<T>>`): for frame data, where the
  opposite guarantee is correct — a fast producer (a camera thread, a CUDA kernel's completion)
  should never queue up behind a slower consumer. Only the *latest* submitted frame matters;
  anything superseded before the render thread got to it is simply dropped.
- **`oneshot_channel`** (`fastgui-core::queue`, a bounded-1 `crossbeam-channel`): for the rare
  call that needs a synchronous reply from the render thread — currently only
  `Viewport.create_cuda_surface()`, which must hand back a real device pointer the caller can
  immediately write into. The calling thread sends a command carrying the sender half, then
  blocks on the receiver half via `Python::detach()` so it releases the GIL while waiting
  (letting other Python threads keep running, and avoiding a deadlock if the render thread
  itself needs the GIL to finish handling the command).
- **`Readback<T>`** (`fastgui-core::readback`, a `Mutex<T>`): for synchronous *getters* that
  shouldn't block on frame timing either — the render thread publishes its last-committed value
  after applying a command, and a getter just reads whatever's there.

None of these require the GIL to use correctly from the Python side — that's the point. A
free-threaded Python build can have several threads submitting frames, mutating widgets, and
reading state back concurrently, and correctness comes from these primitives' own guarantees,
not from Python's GIL serializing access.

## The widget tree

`fastgui-core::widget::WidgetTree` wraps a `taffy::TaffyTree` (flexbox layout) and adds a
side-table (`WidgetId -> WidgetKind`) for whatever taffy doesn't know about — colors, text,
callbacks, drag state. `WidgetId` is just `taffy::NodeId`.

Widgets are retained-mode on the render thread but **described, not built, on the Python side**:
`Label`, `Button`, `Panel`, etc. are plain descriptor objects until `Window.set_content(widget)`
walks them into a `DescribedWidget` tree and sends it across the command queue; `attach()` (on
the render thread) is what actually materializes each node into the real `WidgetTree`, wiring up
cross-references that can't be known until children exist (e.g. a `Splitter`'s bar needs both
its panes' ids; a `Panel`'s title bar needs its own freshly-created parent's id). Once attached,
a widget's `id`/`sender` cells are populated so later calls like `label.set_text(...)` know where
to send their `MutateWidgetTree` closure.

Some widgets are self-contained — `Slider`, `TabBar`, `PanelTitleBar` bake their own rendering
(track+thumb, tab segments, title bar background+text) directly in `fastgui-chrome` rather than
composing child `Label`/`Container` nodes. This sidesteps a real bug class hit early in the
docking work: `align-items: stretch` combined with taffy's intrinsic text measurement squeezed
and clipped title/tab text when it was built from composed children instead.

Mouse input is dispatched by each backend's `app` (`fastgui-render-vk` or
`fastgui-render-mtl`): hit-testing walks the tree for the
topmost widget under the cursor (`hit_test`) for normal clicks, or the topmost *drop region*
(`find_region_at`, which only considers `Container` nodes carrying a `region_id` — dock regions
never overlap, unlike arbitrary widgets, so this is a cheaper and more specific query) while a
panel or tab is being dragged. `DropZone::classify` turns a cursor position plus a target rect
into `Center`/`Left`/`Right`/`Top`/`Bottom` by fractional distance from each edge. None of this
logic lives in Python — Python only ever receives the *result* of a drag (`dragged_id,
target_id, zone`) via a callback, and decides how to restructure its own tree from that; see
`DockArea._on_rearrange` in `python/fastgui/__init__.py` for the tree-surgery side.

## Rendering pipeline

Each frame clears the surface, then optionally draws widget chrome (a full-window CPU raster
from `fastgui-chrome`) and then any in-tree `Viewport` widgets on top, each sampled into its
layout rect via a fullscreen-triangle pipeline (no vertex buffer). `Window.set_viewport` is just
`set_content` of a filling `Viewport` widget. Color surfaces are `_UNORM` / `BGRA8Unorm`, not
sRGB — an early bug had the GPU silently gamma-re-encoding colors that were already final
display bytes, washing out midtones.

On Vulkan, chrome is uploaded through a host-visible `LINEAR` `R8G8B8A8_UNORM` image; vertex
IDs come from `gl_VertexIndex`, and a smaller viewport/scissor places the triangle in the
widget's rect. On Metal the same picture is a `CAMetalLayer` drawable, `MTLTexture` +
`replaceRegion` for CPU bytes, and MSL `vertex_id`. Layout and hit-testing stay in points;
chrome rasterizes at the window's backing scale so Retina chrome matches the Python window size.

CUDA interop (`fastgui-interop-cuda`, `CudaSharedTexture`) is the one path that bypasses the CPU
upload entirely: a device-local image is exported via `VK_KHR_external_memory_win32` and
synchronized with a `VK_KHR_timeline_semaphore` exported via `VK_KHR_external_semaphore_win32`,
so a CUDA kernel can write directly into memory Vulkan will display next frame. The Vulkan-side
export path has been validated with zero validation errors on real (AMD) hardware; the CUDA-side
import has never run against a real NVIDIA GPU and should be treated as unverified — see
README's known limitations.

## Adding a new widget

Roughly, the checklist an existing widget's implementation demonstrates:

1. Add a `WidgetKind` variant in `fastgui-core::widget` (plus any callback type alias it needs).
2. Handle it in `fastgui-chrome`'s rasterizer if it draws itself (self-contained widgets) —
   otherwise it's just a layout container and composes existing children.
3. Add a pyclass in `fastgui-py::widgets` with a `describe()` that builds a `DescribedWidget`,
   and wire any special attach-time cross-referencing into `attach()` if needed.
4. Handle any new input behavior (click, drag) in both backends' `app` (`fastgui-render-vk`
   and `fastgui-render-mtl`) `handle_mouse_press` / cursor-move / release dispatch — they
   mirror each other.
5. Add the type stub in `python/fastgui/__init__.pyi`.
6. Verify by actually running an example — see ROADMAP.md's "How this project has been built"
   for why this step isn't optional.
