# fastgui

Lightweight, GPU-native, no-GIL-safe Python GUI toolkit (`import fastgui`), written in Rust
(PyO3 bindings) with a native GPU renderer — Vulkan on Windows and Linux, Metal on macOS.
This repository is named `pyfastgui`. Designed from the ground up for CPython's free-threaded
(`Py_GIL_DISABLED`) build: every mutator is a message sent to a dedicated render thread over a
lock-free queue, so nothing about the API requires holding the GIL, and multiple threads can
safely drive the UI at once.

> **Status:** pre-alpha, actively developed, not yet published as a package. See
> [ROADMAP.md](ROADMAP.md) for the full milestone-by-milestone history and current work.
> Windows (Vulkan) and macOS (Metal) are the platforms exercised so far; Linux (Vulkan) has
> been tested on X11 with Mesa llvmpipe — see [Known limitations](#known-limitations).

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
- Supports zero-copy GPU interop on the Vulkan backend: a CUDA kernel can write directly into a
  texture displayed next frame, with no CPU round-trip (see `Viewport.create_cuda_surface` —
  currently unverified on real hardware, and not available on macOS; see below).

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

fastgui isn't published yet — build it from source with [maturin](https://www.maturin.rs/):

```
git clone <this repo>
cd pyfastgui
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin numpy
maturin develop --release
```

Requires CPython 3.10+. For the **free-threaded** build you need **CPython 3.14t**: the pinned
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

## Examples

All under [`python/examples/`](python/examples/), runnable directly once installed:

| Example | Demonstrates |
|---|---|
| [`basic_window.py`](python/examples/basic_window.py) | Minimal window bring-up. |
| [`threaded_mutation.py`](python/examples/threaded_mutation.py) | Calling a `Window` mutator concurrently from several threads — no GIL, no crash. |
| [`live_camera_feed.py`](python/examples/live_camera_feed.py) | Streaming numpy frames into a `Viewport` from a background thread. |
| [`cuda_viewport.py`](python/examples/cuda_viewport.py) | GPU-to-GPU CUDA→Vulkan interop, zero CPU copy. **Unverified — no NVIDIA GPU has tested this path. Not available on macOS.** |
| [`widgets_demo.py`](python/examples/widgets_demo.py) | `Box` layout, `Label`, `Button`, `Slider`, click/drag input. |
| [`dock_layout.py`](python/examples/dock_layout.py) | A `DockArea` of resizable, titled `Panel`s with a live `Viewport` in the center. |
| [`dock_rearrange_demo.py`](python/examples/dock_rearrange_demo.py) | Drag a panel's title bar to split or tab-merge regions, including dropping at the window's outer edge to span the whole dock area. |
| [`tabs_demo.py`](python/examples/tabs_demo.py) | Multiple `Panel`s sharing one `DockArea` region via `Tabs`, switched by clicking a header segment. |
| [`floating_panel_demo.py`](python/examples/floating_panel_demo.py) | A panel in its own OS window: move it anywhere (including another monitor), resize from its edges, re-dock by dropping it onto the dock, or tear a docked panel out. |

## Widget API

- **Layout**: `Box` (flexbox row/column via [`taffy`](https://github.com/DioxusLabs/taffy)),
  `Splitter` (draggable divider between two panes).
- **Content**: `Label`, `Button`, `Slider`, `Viewport` (arbitrary CPU/GPU frame content).
- **Docking**: `Panel` (titled, draggable container), `Tabs` (several `Panel`s sharing one
  region), `DockArea` (a split-tree of panels/tabs with full drag-to-rearrange — split, tab-merge
  including growing an existing `Tabs` group, ungroup, whole-window edge drops, and re-docking
  a floating panel; see its docstring in
  [`python/fastgui/__init__.py`](python/fastgui/__init__.py) for the remaining gaps).
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

## Known limitations

- **CUDA interop is unverified on real hardware, and is Vulkan-only.** The Vulkan-side export
  path (`VK_KHR_external_memory_win32` + `VK_KHR_timeline_semaphore`) has been validated with
  zero Vulkan validation errors on real (AMD) hardware, but the CUDA-side import has never run
  against an actual NVIDIA GPU. On macOS there is no CUDA↔Metal path;
  `Viewport.create_cuda_surface` raises `RuntimeError`. Treat the CUDA API as unverified until
  someone runs it on NVIDIA hardware.
- **Linux is less exercised than Windows and macOS.** It uses the same Vulkan backend as
  Windows. It has been tested on X11 (Xvfb + xfwm4) with Mesa llvmpipe (software Vulkan):
  every example, the docking/tab/splitter/floating interactions, and a free-threaded 3.14t
  build, all with zero validation errors. It hasn't been tested on Wayland or with a hardware
  GPU driver, and CUDA interop isn't implemented on Linux yet (`create_cuda_surface` raises
  `RuntimeError`).
- **Free-threaded Python means 3.14t.** PyO3 0.29 doesn't build for 3.13t (see
  [Install](#install)).
- **No text wrapping, scrolling, or keyboard input/focus handling** for any widget yet.
- **`DockArea` / floating**: floating panels are real OS windows (move, resize, tear out by
  dragging a docked panel outside the main window, re-dock by dropping onto the dock).
- **`Viewport.submit_frame` always copies** the numpy/buffer into an owned `Vec<u8>` before
  upload. Expected for the CPU path; a packed-RGBA camera feed will want a fewer-copy path later.
- **Automated tests are still thin.** `cargo test --workspace` (works on every platform; the
  Metal crate compiles to empty off macOS) covers `FrameSlot`, `DropZone` classification and
  preview rects, splitter ratio layout and hit slop, the × hit rect, and `WidgetTree`
  layout/hit-test; `python -m unittest tests.test_dock_area` (from `python/`)
  covers `DockArea` tree surgery. There is no GPU/window integration suite yet.
- **`.venv`/`env` are local, machine-specific dev environments**, not checked in — a fresh
  clone needs its own `python -m venv` plus the Rust and platform GPU toolchain described in
  [Install](#install) before anything builds.

## Contributing / picking up development

This project has been built session-by-session with an AI pair-programmer, using
[ROADMAP.md](ROADMAP.md) as the persistent memory between sessions — it documents not just
what's done but *why*, including bugs found and fixed along the way and the reasoning behind
non-obvious design choices (e.g. how floating panels moved from in-window overlays to real OS
windows). Read it before making non-trivial changes.
