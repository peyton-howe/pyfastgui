# fastgui

Lightweight, GPU-native, no-GIL-safe Python GUI toolkit (`import fastgui`), written in Rust
(PyO3 bindings) with a native GPU renderer — Vulkan on Windows and Linux, Metal on macOS.
This repository is named `pyfastgui`. Designed from the ground up for CPython's free-threaded
(`Py_GIL_DISABLED`) build: every mutator is a message sent to a dedicated render thread over a
lock-free queue, so nothing about the API requires holding the GIL, and multiple threads can
safely drive the UI at once.

> **Status:** pre-alpha. Widget milestones through drag-and-drop are implemented. Wheel CI
> is green for Windows x64, macOS arm64, and manylinux x86_64 (`cp311-abi3` and `cp314t`);
> the package is not on PyPI yet. See [ROADMAP.md](ROADMAP.md) for history and what is left.
> Windows (Vulkan) and macOS (Metal) are the platforms exercised on a real window; Linux
> (Vulkan) has been tested on X11 with Mesa llvmpipe — see [Known limitations](#known-limitations).

## Why

Most Python GUI toolkits either wrap a heavyweight native toolkit (Qt, GTK) or render via a
CPU-bound abstraction that doesn't take advantage of the GPU already sitting in every machine.
fastgui instead:

- Renders everything — widget chrome and arbitrary GPU/CPU frame content (`Viewport`) — through
  a native GPU backend (Vulkan on Windows/Linux, Metal on macOS). Widget chrome is a display
  list of quads over a glyph atlas (text still shaped once on the CPU and cached); scrolling or
  resizing re-sends vertex data instead of a window-sized pixmap. Set `FASTGUI_CHROME=cpu` to
  fall back to the retained-pixmap path.
- Treats every widget mutation (`label.set_text(...)`, `slider.set_value(...)`, dragging a
  panel) as a command sent across a channel to the render thread, rather than requiring the
  caller to be "on the UI thread" — the free-threaded Python build can call into fastgui from
  any thread without contention.
- Takes CUDA arrays directly: `Viewport.submit_cuda(tensor)` accepts anything with
  `__cuda_array_interface__` (torch, cupy, numba, jax), ordered against the caller's stream.
  On Windows or Linux with Vulkan and NVIDIA it's one GPU-to-GPU copy (or none, via
  `create_cuda_surface`; Linux exports opaque fds). Elsewhere it falls back to a copy
  through host memory. Unverified on NVIDIA hardware; see below.

## Quickstart

```python
import fastgui as fg

def main() -> None:
    window = fg.Window(title="Hello, fastgui", width=800, height=500)

    counter = {"n": 0}
    label = fg.Label("Count: 0", font_size=20.0)

    def on_click() -> None:
        counter["n"] += 1
        label.set_text(f"Count: {counter['n']}")

    window.set_content(
        fg.Box(
            direction="column", gap=12.0, padding=16.0,
            children=[label, fg.Button("Click me", on_click=on_click)],
        )
    )
    window.run()

if __name__ == "__main__":
    main()
```

Run it (see [Install](#install) to set up a venv first):

```
python your_script.py
```

## Install

fastgui isn't on PyPI yet. Release wheels are built in CI
([`.github/workflows/wheels.yml`](.github/workflows/wheels.yml)) for Windows x64, macOS arm64
and manylinux x86_64. Each platform gets a `cp311-abi3` wheel for every GIL CPython 3.11+ and a
`cp314t` wheel for free-threaded 3.14. Once 0.0.1 is published:

```
pip install pyfastgui
```

The distribution is named `pyfastgui` because `fastgui` on PyPI is an unrelated project. You
still `import fastgui`.

Until then, build it from source with [maturin](https://www.maturin.rs/):

```
git clone <this repo>
cd pyfastgui
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin numpy
maturin develop --release
```

Requires CPython 3.11+ (3.11 is where the buffer protocol joined the stable ABI, which
`Viewport.submit_frame` and `Image` need). For the **free-threaded** build you need **CPython 3.14t**: the pinned
PyO3 (0.29) refuses to build against a 3.13t interpreter, so a `python3.13t` venv fails at
`maturin develop`. Regular (GIL) 3.13 is fine.

You'll also need the platform toolchain fastgui itself builds against:

- Rust (stable, via [rustup](https://rustup.rs/)) — `rust-toolchain.toml` pins the channel.
- On Windows: MSVC Build Tools, and the [Vulkan SDK](https://vulkan.lunarg.com/) (needed to
  recompile `crates/fastgui-render-vk/shaders/*.spv` if you touch the shaders; prebuilt `.spv`
  files are checked in so a plain build doesn't need the SDK).
- On macOS: Xcode Command Line Tools. Metal is part of the OS; there is no Vulkan SDK step.
  `fastgui-py` selects the Metal backend automatically on this platform.
- On Linux: a Vulkan-capable driver. Same shader note as Windows — prebuilt `.spv` is checked in.

`maturin develop` builds the Rust extension and installs it editable into whichever venv is
active — re-run it after any change under `crates/`. If you're testing against both a regular
and a free-threaded CPython build (as this project's own development does), keep two separate
venvs and rebuild into each; see the note in ROADMAP.md about `maturin develop` resolving its
target venv by naming convention (a folder literally named `.venv`), not by which venv's
`maturin` binary you invoke — set `VIRTUAL_ENV` explicitly when targeting a differently-named venv.

To build the same release wheels locally (into `dist/`):

```
maturin build --release --features abi3 --out dist      # cp311-abi3, any GIL CPython 3.11+
maturin build --release -i python3.14t --out dist       # cp314t, free-threaded
```

The `abi3` cargo feature is off by default, so `maturin develop` and the free-threaded build
stay version-specific.

## Examples

All under [`python/examples/`](python/examples/), runnable directly once installed:

| Example | Demonstrates |
|---|---|
| [`basic_window.py`](python/examples/basic_window.py) | Minimal window bring-up. |
| [`threaded_mutation.py`](python/examples/threaded_mutation.py) | Calling a `Window` mutator concurrently from several threads — no GIL, no crash. |
| [`live_camera_feed.py`](python/examples/live_camera_feed.py) | Streaming numpy frames into a `Viewport` from a background thread. |
| [`cuda_viewport.py`](python/examples/cuda_viewport.py) | cupy frames via `submit_cuda` (GPU-to-GPU where interop exists, host copy elsewhere), or `--surface` for zero-copy `create_cuda_surface`. **Unverified — no NVIDIA GPU has tested this path.** |
| [`widgets_demo.py`](python/examples/widgets_demo.py) | `Box` layout, `Label`, `Button`, `Slider`, click/drag input. |
| [`text_input_demo.py`](python/examples/text_input_demo.py) | `TextInput` fields, keyboard focus (Tab/Shift+Tab), selection, clipboard, undo, IME. |
| [`list_demo.py`](python/examples/list_demo.py) | A virtualized `ListView` of 1,000,000 rows with live filtering, keyboard selection and activation. |
| [`theme_demo.py`](python/examples/theme_demo.py) | `fg.Theme`: switch a running window between light, dark and a custom accent. |
| [`popup_demo.py`](python/examples/popup_demo.py) | `Popup` overlays: a menu anchored to a button, a modal dialog, a popup at a point — all drawn over live `Viewport` video. |
| [`scroll_demo.py`](python/examples/scroll_demo.py) | `ScrollArea`: wheel/trackpad and scrollbar scrolling, clipping, a live `Viewport` inside, Tab scrolling focus into view. |
| [`form_controls_demo.py`](python/examples/form_controls_demo.py) | M7 form controls: `Checkbox`, `Radio`, `Toggle`, `SpinBox`, `NumericScrub`, `ProgressBar`, `ComboBox`, `Grid`, `Image`, `TextArea`. |
| [`menus_demo.py`](python/examples/menus_demo.py) | M7 7C: `MenuBar` / submenus / check-radio-icon rows, context menus, hover tooltips, accelerators, `Toolbar`, `StatusBar`. |
| [`app_chrome_demo.py`](python/examples/app_chrome_demo.py) | M7 7C: `Dialog`, file/color pickers, `GroupBox`, `CollapsibleSection`, `StackedWidget`. |
| [`table_demo.py`](python/examples/table_demo.py) | M7 7D: virtualized `Table` of 1,000,000 numpy rows with sticky header and keyboard selection. |
| [`tree_demo.py`](python/examples/tree_demo.py) | M7 7D: `TreeView` expand/collapse, path selection, Left/Right navigation. |
| [`inspector_demo.py`](python/examples/inspector_demo.py) | M7 7D: `PropertyInspector` (label + 7B editors) driven by a tree selection. |
| [`plots_demo.py`](python/examples/plots_demo.py) | M7 7E: numpy-fed line, scatter, heatmap, histogram, bar, contour, 3D surface, 3D scatter, and mesh plots (CPU raster, GPU image layer). |
| [`tier4_demo.py`](python/examples/tier4_demo.py) | M7 7E: `ImageViewer`, `LogView`, `CodeEditor`, `CommandPalette` (Ctrl+K), `Observable`, `Gauge`, `Timeline`, `NodeGraph`. |
| [`interaction_demo.py`](python/examples/interaction_demo.py) | M7 7F: hover state layer, `set_enabled` on a control box, list reorder and tree reparent by drag-and-drop, a drop zone, and OS file drops on an `Image`. |
| [`dock_layout.py`](python/examples/dock_layout.py) | A `DockArea` of resizable, titled `Panel`s with a live `Viewport` in the center. |
| [`dock_rearrange_demo.py`](python/examples/dock_rearrange_demo.py) | Drag a panel's title bar to split or tab-merge regions, including dropping at the window's outer edge to span the whole dock area. |
| [`tabs_demo.py`](python/examples/tabs_demo.py) | Multiple `Panel`s sharing one `DockArea` region via `Tabs`, switched by clicking a header segment. |
| [`floating_panel_demo.py`](python/examples/floating_panel_demo.py) | A panel in its own OS window: move it anywhere (including another monitor), resize from its edges, re-dock by dropping it onto the dock, or tear a docked panel out. |

## Widget API

- **Layout**: `Box` (flexbox row/column via [`taffy`](https://github.com/DioxusLabs/taffy);
  `wrap=True` flows children onto more rows/columns),
  `Grid` (CSS Grid equal columns), `Splitter` (draggable divider between two panes),
  `ScrollArea` (scrollable, clipped viewport), `Popup` (menus, dropdowns, tooltips, modal
  dialogs — anchored overlays drawn above everything).
- **Menus / chrome**: `MenuBar`, `Menu`, `MenuItem` (`submenu=`, `checked=`, `radio_group=`,
  `icon=`), `MenuSeparator`, `Toolbar`, `StatusBar`, `Dialog`, `GroupBox`,
  `CollapsibleSection`, `StackedWidget` (Python composition); `open_file_dialog` /
  `save_file_dialog` (native via `rfd`), `pick_color` (in-app); `tooltip=` / `context_menu=`;
  keyboard accelerators and arrow-key menu navigation.
- **Data views**: `ListView`, `Table`, and `TreeView` (virtualized: only visible rows are
  laid out and drawn), `PropertyInspector`. `TreeView.move_node` moves a subtree without
  resetting expand state. `Table` cells are strings at the Python boundary.
- **Plots and tier 4**: `PlotLine`, `PlotScatter`, `PlotHeatmap`, `PlotHistogram`,
  `PlotBar`, `PlotContour`, `PlotSurface`, `PlotScatter3D`, `PlotMesh` (CPU raster into
  the `Image` layer; wheel zoom, drag pan). `ImageViewer`, `LogView`, `CodeEditor`,
  `CommandPalette`, `Gauge`, `Timeline`, `NodeGraph`, `Observable`.
- **Theming**: `Theme` (`dark()`, `light()`, `replace(...)`), `set_theme` / `get_theme`,
  `Window.set_theme`. Colors, font sizes and the font family default to the current theme;
  `font_size="small"|"body"|"large"` and `gap=`/`padding="small"|"medium"|"large"`
  name theme tokens.
- **Content**: `Label`, `Button`, `Slider`, `TextInput`, `TextArea`, `Checkbox`, `Radio`, `Toggle`,
  `SpinBox`, `NumericScrub`, `ProgressBar`, `ComboBox`, `Image`, `Viewport` (CPU frames,
  an `MTLTexture` via `submit_metal`, a linear dma-buf via `submit_dmabuf`, a shared D3D11
  texture via `submit_d3d11`, or CUDA).
- **Docking**: `Panel` (titled, draggable container), `Tabs` (several `Panel`s sharing one
  region), `DockArea` (a split-tree of panels/tabs with full drag-to-rearrange — split, tab-merge
  including growing an existing `Tabs` group, ungroup, whole-window edge drops, and re-docking
  a floating panel; see its docstring in
  [`python/fastgui/__init__.py`](python/fastgui/__init__.py) for the remaining gaps).
- **States and drag-and-drop**: every control and `Box` takes `enabled=` / `set_enabled()`
  (a disabled `Box` disables everything in it: dimmed, no clicks, focus, drags or shortcuts).
  Hovered controls, and the row under the cursor in lists, tables, trees and combo-box
  drop-downs, get the theme's `hover` state layer. `ListView`, `TreeView` and `Box` take
  `set_drag_source(tag, data)` / `set_drop_target(accept, on_drop)` (payload: a string tag plus
  bytes); `Viewport` and `Image` take `set_file_drop(on_drop)` for files dropped from the OS.
  `Image.load(path)` shows a PNG or JPEG; `TreeView.move_node(source, target, place)` moves a
  node keeping every node's expand state.
- **Window**: `Window(title, width, height)` — `set_content(widget)`, `set_clear_color(...)`,
  `add_floating_panel(panel, x, y, width, height)`, `.run()` (blocks, owns the render loop).

Full signatures and docstrings live in the type stub,
[`python/fastgui/__init__.pyi`](python/fastgui/__init__.pyi) — every constructor keyword and
its default is there.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the crate layout and threading model in
more depth. Short version: `fastgui-core` owns the cross-thread primitives (a command queue for
fire-and-forget mutations, a latest-wins mailbox for frame data, a retained-mode widget tree
over `taffy`); `fastgui-app` owns the shared winit loop (dock/float/ghost input, multi-window
lifecycle, command drain), generic over a `SurfaceBackend`; `fastgui-render-vk` (Windows/Linux)
and `fastgui-render-mtl` (macOS) implement `SurfaceBackend` and provide thin `run()` wrappers;
`fastgui-chrome` builds the chrome display list (GPU quads + atlas by default, or a retained
CPU pixmap with `FASTGUI_CHROME=cpu`); `fastgui-py` is the PyO3 layer and picks the backend with
`cfg(target_os = "macos")`.

## Frame upload

`submit_frame` copies a numpy array into a `Vec<u8>`. The render thread then writes those
bytes into its own texture: `replaceRegion` on Metal, a copy into a host-visible linear
image on Vulkan. That is the CPU path.

On macOS the fast way to start from numpy is mtlpy, not `submit_frame`. Create the texture
once with `empty_texture(..., readable=True, writable=False, private=True)` — shader-read
only, private storage, which is the layout mtlpy documents for GPU sampling. Each frame,
write into one reused `Buffer` and blit:

```python
stage.contents[:] = frame.reshape(-1)
texture.upload_from_buffer(stage)
viewport.submit_metal(texture)
```

`submit_metal` retains `Texture.mtl_ptr`. The viewport shader samples that object. When a
compute kernel has already written the texture, skip the blit; fastgui does not copy the
pixels.

Measured on this Mac (Apple GPU, 2026-10-19), median of 40 frames after warmup:

| | 1920×1080 | 3840×2160 |
|---|---:|---:|
| mtlpy reused buffer + `upload_from_buffer` | 0.38 ms | 0.88 ms |
| `submit_frame` host copy + Metal `replaceRegion` | 1.01 ms | 3.40 ms |
| `submit_metal` on a texture that is already resident | 0.14 ms | 0.23 ms |
| Sample that imported texture, no upload | 0.26 ms | 0.39 ms |

`python/examples/metal_import_bench.py` reprints the host-side rows.
`cargo run -p fastgui-render-mtl --release --example import_bench` reprints the GPU rows.

Measured on a Mali-G610 MC4 with `cargo run -p fastgui-render-vk --release --example upload_bench`,
median of 40 frames after warmup. Headless, so this is not a vsync present. Readback of the
sampled center pixel matched.

| | 1920×1080 | 3840×2160 |
|---|---:|---:|
| Host copy + linear upload, no GPU wait | 1.87 ms | 7.79 ms |
| That upload plus an offscreen sample | 3.25 ms | 11.46 ms |
| Sample an imported modifier-0 RGBA8 dma-buf | 1.09 ms | 2.00 ms |

Plain `VK_IMAGE_TILING_LINEAR` on this Mali is export-only. `submit_dmabuf` imports modifier
0 with `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` instead, which the driver marks exportable
and importable. The dma-buf row is that image, exported and imported in this process. The
center pixel matched. Non-zero modifiers are still not imported. Windows has not been
measured. The same command skips, and still exits 0, when the machine has no Vulkan 1.3
device. Windows has no dma-buf path; its zero-copy path is `submit_d3d11`.

Windows `submit_d3d11` imports a shared Direct3D 11 texture through `VK_KHR_external_memory_win32`,
and the viewport shader samples it with no CPU copy. The texture is 2D, `R8G8B8A8_UNORM` or
`B8G8R8A8_UNORM` (what DXGI desktop duplication and most video output produce), created with
`D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`. Pass the NT handle from
`IDXGIResource1::CreateSharedHandle`; fastgui duplicates it. Create the D3D device on the
adapter whose LUID is `fastgui.adapter_luid()`.

- **Checked against D3D.** Vulkan takes any size for an imported texture, so fastgui opens the
  handle on a D3D11 device on the window's adapter and reads the texture's real size, format,
  mip count and sample count. `width` and `height` must match, `format` (`"rgba8"`/`"bgra8"`) is
  detected when left `None`, and a mismatch, a handle that is not a texture, or a texture on a
  different GPU is refused instead of imported. `Viewport.d3d11_status` / `Image.d3d11_status`
  say what the render thread did with the last submission, and the previous frame stays.
- **Cached.** Create the handle once per texture and resubmit it. fastgui recognizes a duplicate
  of a texture it has already imported (`CompareObjectHandles`) and acquires the same image
  instead of importing again, so the producer's new pixels show up. Up to four textures per
  viewport stay imported, so double or triple buffering hits too. A new handle from another
  `CreateSharedHandle` is imported again.
- **Sync.** Without a fence the producer finishes (flush, then wait on an event query) before the
  call. With `fence=` (a shared `ID3D11Fence`/`ID3D12Fence` NT handle) fastgui waits for
  `wait_value` before sampling, for at most 100 ms on the render thread. A late fence drops the
  submission and keeps the previous frame, so a stalled producer cannot hang the window.
  `signal_value` is signaled once the frame that first draws the submission has finished
  sampling; a producer that reuses a texture waits for it. Keyed mutexes are not used.

`python/examples/d3d11_viewport.py` is a ctypes-only producer (no pywin32) that renders into a ring
of shared textures: `--bgra`, `--fence`, `--color R,G,B`, `--seconds N`. The Rust tests
(`cargo test -p fastgui-render-vk --release d3d11 -- --nocapture`) create real D3D11 textures on the
Vulkan device's adapter (matched by LUID) and check: pixels at the corners and middle for RGBA and
BGRA, the producer rewriting a cached texture, every wrong size/format/handle refused, and the
fence wait, timeout and signal-back. Measured on a Radeon 840M (2026-10-26), median of 40 frames,
headless (not a vsync present), `upload_bench` and the D3D11 test run back to back:

| | 1920×1080 | 3840×2160 |
|---|---:|---:|
| CPU upload + offscreen sample | 1.8 ms | 7.9 to 8.7 ms |
| `submit_d3d11`, first submit of a texture (import + sample) | 2.5 to 2.8 ms | 3.6 to 4.1 ms |
| `submit_d3d11`, texture already imported (acquire + sample) | 0.27 ms | 0.9 to 1.0 ms |

The first submit of a texture costs about 2 ms of import at any size and, with the D3D check
and fence import on top, is not faster than the CPU path at 1080p; the saving is the cached
row. In a real window the example shows the producer's color on screen (`--color 255,64,0`
reads back as exactly 255, 64, 0 for RGBA, BGRA, and BGRA with a fence, on 3.14 and 3.14t) with
zero validation messages.

## Known limitations

- **CUDA interop is unverified on NVIDIA hardware.** The Windows Vulkan half (exported slot
  buffer, timeline semaphores, the pickup copy) is tested headless with zero validation
  errors on real (AMD) hardware, and the CUDA struct layouts are checked against NVIDIA's
  own bindings. Linux exports the same objects as opaque fds when the device advertises
  `VK_KHR_external_memory_fd` and `VK_KHR_external_semaphore_fd`. The CUDA side has never
  run on an NVIDIA GPU. `submit_cuda` falls back to a host copy on macOS, on a non-NVIDIA
  window GPU, in floating panels, and before `run()`; `create_cuda_surface` raises there
  instead. `Viewport.cuda_status` says which path is in use.
- **Linux is less exercised than Windows and macOS.** It uses the same Vulkan backend as
  Windows. It has been tested on X11 (Xvfb + xfwm4) with Mesa llvmpipe (software Vulkan):
  every example, the docking/tab/splitter/floating interactions, and a free-threaded 3.14t
  build, all with zero validation errors. A headless upload bench has run on a Mali-G610
  (see [Frame upload](#frame-upload)). It hasn't been tested on Wayland, and no hardware
  GPU has driven a real window. Opaque-fd CUDA export is unverified on NVIDIA. On the
  Mali-G610, modifier-0 RGBA8 imports with DRM-modifier tiling; plain linear tiling is
  export-only. That dma-buf sample ran headless and the center pixel matched.
- **Wayland is untested.** winit picks Wayland when `WAYLAND_DISPLAY` is set; for 0.0.1 treat
  that as unsupported and run under X11 or XWayland (unset `WAYLAND_DISPLAY`).
- **Prebuilt wheels cover Windows x64, macOS arm64 and manylinux x86_64 only.** CI builds
  and smoke-tests those six wheels on `main`. They are not published. Intel Macs, Linux
  aarch64, and Windows ARM build from source.
- **Metal import is measured on this Mac. Vulkan CPU upload and a modifier-0 dma-buf sample are measured on a Mali-G610.**
  `Viewport.submit_metal` / `Image.submit_metal` retain an `MTLTexture` (an integer or any
  object with `mtl_ptr`) and the viewport shader samples it. The fast mtlpy texture is
  `empty_texture(..., readable=True, writable=False, private=True)` in a float format such
  as `rgba8Unorm`, on the system default device. For a hot loop, reuse one `Buffer`
  (`contents[:] = frame`, then `upload_from_buffer`) and call `submit_metal` with that
  texture. Optional `MTLSharedEvent` pointers wait and signal once, on the command buffer
  that first draws the submission. See [Frame upload](#frame-upload). `submit_dmabuf`
  (Linux) duplicates the caller's fd and samples a linear RGBA8 dma-buf (`modifier` must
  be 0; `stride` 0 means tightly packed). The producer finishes writing before the call.
  On a Mali-G610 MC4 the host copy plus linear upload was 1.87 ms at 1080p and 7.79 ms at
  4K. The offscreen sample of that upload was 3.25 ms and 11.46 ms. Plain linear tiling is
  export-only there, so import uses DRM-modifier tiling with modifier 0. Sampling that
  imported image was 1.09 ms and 2.00 ms, and the center pixel matched. Non-zero DRM
  modifiers and NV12 are not imported. `submit_d3d11` (Windows) samples a shared D3D11
  RGBA8 or BGRA8 texture, checks its size and format against D3D, keeps up to four imports per
  viewport, optionally syncs on a shared fence, and is verified on an AMD GPU in tests and in a
  real window (see [Frame upload](#frame-upload)); D3D12 resources, keyed mutexes, NV12, and
  textures from another GPU are not imported.
- **Free-threaded Python means 3.14t.** PyO3 0.29 doesn't build for 3.13t (see
  [Install](#install)). GIL builds need CPython 3.11+.
- **Text editing gaps:** `TextArea` has no soft wrap, the caret doesn't blink, and clicking a
  scrollbar track doesn't page. No bidi/RTL caret movement.
- **Drag-and-drop is main-window only.** Widget drags work inside the main window, not in
  floating panels or between windows. OS file drops land at the cursor on Windows; elsewhere
  they land where the cursor last moved in the window. GPU `Viewport`/`Image` layers aren't
  dimmed when disabled (they draw above chrome).
- **No accessibility support.** Screen readers can't see fastgui widgets yet.
- **`Viewport` and `Image` stretch by default**; pass `fit="contain"` to letterbox.
- **`Table`** has no column resize or sort, and **`DockArea`** layouts aren't saved or restored.
- **`Viewport.submit_frame` always copies** the numpy/buffer into an owned `Vec<u8>` before
  upload. That is the CPU path. Zero-copy is `submit_metal`, `submit_dmabuf`, `submit_d3d11`, or
  `create_cuda_surface`. From a numpy array, mtlpy's blit into a private texture is still
  a copy, and on this Mac it is faster than `submit_frame` plus `replaceRegion`. See
  [Frame upload](#frame-upload).
- **No GPU/window integration suite.** `cargo test --workspace` (works on every platform; the
  Metal crate compiles to empty off macOS) covers the core primitives, layout and hit-testing,
  and `python -m pytest python/tests` covers the Python API headlessly. The wheel smoke job
  runs that pytest suite from the installed wheel. Opening a window is still verified by
  running the examples.
- **`.venv`/`env` are local, machine-specific dev environments**, not checked in — a fresh
  clone needs its own `python -m venv` plus the Rust and platform GPU toolchain described in
  [Install](#install) before anything builds.

## Contributing / picking up development

This project has been built session-by-session with an AI pair-programmer, using
[ROADMAP.md](ROADMAP.md) as the persistent memory between sessions — it documents not just
what's done but *why*, including bugs found and fixed along the way and the reasoning behind
non-obvious design choices (e.g. how floating panels moved from in-window overlays to real OS
windows). Read it before making non-trivial changes.
