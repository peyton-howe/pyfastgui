# fast-gui

Lightweight, GPU-native, no-GIL-safe Python GUI toolkit, written in Rust (PyO3 bindings) with
a native Vulkan renderer. Designed from the ground up for CPython's free-threaded (`Py_GIL_DISABLED`)
build: every mutator is a message sent to a dedicated render thread over a lock-free queue, so
nothing about the API requires holding the GIL, and multiple threads can safely drive the UI at
once.

> **Status:** pre-alpha, actively developed, not yet published as a package. See
> [ROADMAP.md](ROADMAP.md) for the full milestone-by-milestone history and current work.
> Windows + Vulkan is the only platform verified so far — see [Known limitations](#known-limitations).

## Why

Most Python GUI toolkits either wrap a heavyweight native toolkit (Qt, GTK) or render via a
CPU-bound abstraction that doesn't take advantage of the GPU already sitting in every machine.
fast-gui instead:

- Renders everything — widget chrome and arbitrary GPU/CPU frame content (`Viewport`) — through
  one Vulkan swapchain, with widget chrome rasterized via `cosmic-text` + `tiny-skia` and
  uploaded as a single texture per frame.
- Treats every widget mutation (`label.set_text(...)`, `slider.set_value(...)`, dragging a
  panel) as a command sent across a channel to the render thread, rather than requiring the
  caller to be "on the UI thread" — the free-threaded Python build can call into fast-gui from
  any thread without contention.
- Supports zero-copy GPU interop: a CUDA kernel can write directly into a texture Vulkan
  displays next frame, with no CPU round-trip (see `Viewport.create_cuda_surface` — currently
  unverified on real hardware, see below).

## Quickstart

```python
import fastgui as fg

def main() -> None:
    window = fg.Window(title="Hello, fast-gui", width=800, height=500)

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

fast-gui isn't published yet — build it from source with [maturin](https://www.maturin.rs/):

```
git clone <this repo>
cd fast-gui
python -m venv .venv
.venv\Scripts\activate      # PowerShell: .venv\Scripts\Activate.ps1
pip install maturin numpy
maturin develop --release
```

You'll also need the platform toolchain fast-gui itself builds against:

- Rust (stable, via [rustup](https://rustup.rs/)) — `rust-toolchain.toml` pins the channel/target.
- On Windows: MSVC Build Tools, and the [Vulkan SDK](https://vulkan.lunarg.com/) (needed to
  recompile `crates/fastgui-render-vk/shaders/*.spv` if you touch the shaders; prebuilt `.spv`
  files are checked in so a plain build doesn't need the SDK).

`maturin develop` builds the Rust extension and installs it editable into whichever venv is
active — re-run it after any change under `crates/`. If you're testing against both a regular
and a free-threaded CPython build (as this project's own development does), keep two separate
venvs and rebuild into each; see the note in ROADMAP.md about `maturin develop` resolving its
target venv by naming convention (a folder literally named `.venv`), not by which venv's
`maturin` binary you invoke — set `VIRTUAL_ENV` explicitly when targeting a differently-named venv.

## Examples

All under [`python/examples/`](python/examples/), runnable directly once installed:

| Example | Demonstrates |
|---|---|
| [`basic_window.py`](python/examples/basic_window.py) | Minimal window bring-up. |
| [`threaded_mutation.py`](python/examples/threaded_mutation.py) | Calling a `Window` mutator concurrently from several threads — no GIL, no crash. |
| [`live_camera_feed.py`](python/examples/live_camera_feed.py) | Streaming numpy frames into a `Viewport` from a background thread. |
| [`cuda_viewport.py`](python/examples/cuda_viewport.py) | GPU-to-GPU CUDA→Vulkan interop, zero CPU copy. **Unverified — no NVIDIA GPU has tested this path, see below.** |
| [`widgets_demo.py`](python/examples/widgets_demo.py) | `Box` layout, `Label`, `Button`, `Slider`, click/drag input. |
| [`dock_layout.py`](python/examples/dock_layout.py) | A `DockArea` of resizable, titled `Panel`s with a live `Viewport` in the center. |
| [`dock_rearrange_demo.py`](python/examples/dock_rearrange_demo.py) | Drag a panel's title bar to split or tab-merge regions, including dropping at the window's outer edge to span the whole dock area. |
| [`tabs_demo.py`](python/examples/tabs_demo.py) | Multiple `Panel`s sharing one `DockArea` region via `Tabs`, switched by clicking a header segment. |
| [`floating_panel_demo.py`](python/examples/floating_panel_demo.py) | An always-on-top panel dragged freely over the rest of the window. |

## Widget API

- **Layout**: `Box` (flexbox row/column via [`taffy`](https://github.com/DioxusLabs/taffy)),
  `Splitter` (draggable divider between two panes).
- **Content**: `Label`, `Button`, `Slider`, `Viewport` (arbitrary CPU/GPU frame content).
- **Docking**: `Panel` (titled, draggable container), `Tabs` (several `Panel`s sharing one
  region), `DockArea` (a split-tree of panels/tabs with full drag-to-rearrange — split, tab-merge,
  ungroup, and whole-window edge drops; see its docstring in
  [`python/fastgui/__init__.py`](python/fastgui/__init__.py) for the couple of gestures not
  supported yet).
- **Window**: `Window(title, width, height)` — `set_content(widget)`, `set_clear_color(...)`,
  `add_floating_panel(panel, x, y, width, height)`, `.run()` (blocks, owns the render loop).

Full signatures and docstrings live in the type stub,
[`python/fastgui/__init__.pyi`](python/fastgui/__init__.pyi) — every constructor keyword and
its default is there.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the crate layout and threading model in
more depth. Short version: `fastgui-core` owns the cross-thread primitives (a command queue for
fire-and-forget mutations, a latest-wins mailbox for frame data, a retained-mode widget tree
over `taffy`) and is renderer-agnostic; `fastgui-render-vk` is the Vulkan backend and owns the
actual window/event loop; `fastgui-chrome` rasterizes widget chrome into a texture that
`fastgui-render-vk` uploads and displays like any other frame; `fastgui-py` is the PyO3 layer
tying it all to a Python API.

## Known limitations

- **CUDA interop is unverified on real hardware.** The Vulkan-side export path
  (`VK_KHR_external_memory_win32` + `VK_KHR_timeline_semaphore`) has been validated with zero
  Vulkan validation errors on real (AMD) hardware, but the CUDA-side import has never run
  against an actual NVIDIA GPU — this machine only has an integrated AMD GPU. Treat
  `Viewport.create_cuda_surface` as unverified until someone runs it on NVIDIA hardware.
- **No macOS support yet.** `fastgui-render-mtl` is an empty stub; Metal is planned (M5 in
  ROADMAP.md) but not started.
- **No text wrapping, scrolling, or keyboard input/focus handling** for any widget yet.
- **`DockArea` gaps**: dropping a panel onto an *existing* `Tabs` group's center (to grow it
  past 2 members) isn't supported yet — only forming a new 2-member group, or ungrouping one
  back out. Floating panels (`add_floating_panel`) can't be dragged into or out of a `DockArea`,
  and aren't resizable.
- **Automated tests are still thin.** `cargo test -p fastgui-core` covers `FrameSlot`, `DropZone`,
  and `WidgetTree` layout/hit-test; `python -m unittest tests.test_dock_area` (from `python/`)
  covers `DockArea` tree surgery. There is no GPU/window integration suite yet.
- **`.venv`/`env` are local, machine-specific dev environments**, not checked in — a fresh
  clone needs its own `python -m venv` plus the Rust/Vulkan toolchain described in
  [Install](#install) before anything builds.

## Contributing / picking up development

This project has been built session-by-session with an AI pair-programmer, using
[ROADMAP.md](ROADMAP.md) as the persistent memory between sessions — it documents not just
what's done but *why*, including bugs found and fixed along the way and the reasoning behind
non-obvious design choices (e.g. why floating panels are simulated within one window rather than
real OS windows). Read it before making non-trivial changes.
