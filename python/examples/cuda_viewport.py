"""CUDA arrays into a Viewport, two ways:

- default: `viewport.submit_cuda(array)` with any CUDA array (cupy here; torch works the same).
  One GPU-to-GPU copy where the window's GPU can share memory with CUDA (Windows, Vulkan,
  NVIDIA); a copy through host memory everywhere else. The label above the image shows which.
- `--surface`: `viewport.create_cuda_surface()` and `with surface.frame(stream) as f:`, writing
  straight into the memory fastgui displays (no CUDA-side copy). Interop only.

*** UNVERIFIED ON NVIDIA HARDWARE ***
Written on a machine with no NVIDIA GPU. The Vulkan half of the interop is tested (headless, with
validation layers); the CUDA half has never run against a real driver. If this misbehaves, the
bug is most likely in fastgui's interop plumbing, not in this script.

Requires an NVIDIA GPU, the CUDA driver, and cupy (`pip install cupy-cuda12x` or the build
matching your CUDA version).
"""

import sys
import threading
import time

try:
    import cupy as cp
except ImportError:
    sys.exit(
        "cuda_viewport.py needs cupy (and an NVIDIA GPU with the CUDA driver).\n"
        "Install the build matching your CUDA version, e.g. `pip install cupy-cuda12x`,\n"
        "or run live_camera_feed.py for the CPU submit_frame() path instead."
    )

import fastgui as fg

WIDTH, HEIGHT = 640, 480


def render(out: cp.ndarray, t: float, x: cp.ndarray, y: cp.ndarray) -> None:
    """Write an animated RGBA gradient into `out` ((HEIGHT, WIDTH, 4) uint8)."""
    out[..., 0] = ((0.5 + 0.5 * cp.sin(6.0 * x + t)) * 255).astype(cp.uint8)
    out[..., 1] = ((0.5 + 0.5 * cp.sin(6.0 * y + t * 1.3)) * 255).astype(cp.uint8)
    out[..., 2] = ((0.5 + 0.5 * cp.sin(6.0 * (x + y) + t * 0.7)) * 255).astype(cp.uint8)
    out[..., 3] = 255


def produce(status: fg.Label, viewport: fg.Viewport, use_surface: bool, stop: threading.Event) -> None:
    y, x = cp.mgrid[0:HEIGHT, 0:WIDTH].astype(cp.float32)
    x /= WIDTH
    y /= HEIGHT
    stream = cp.cuda.Stream(non_blocking=True)
    start = time.monotonic()

    if use_surface:
        try:
            surface = viewport.create_cuda_surface(WIDTH, HEIGHT)
        except RuntimeError as err:
            status.set_text(f"create_cuda_surface failed: {err}")
            return
        status.set_text("CUDA surface (zero-copy)")
        with stream:
            while not stop.is_set():
                with surface.frame(stream) as frame:
                    render(cp.asarray(frame), time.monotonic() - start, x, y)
                stop.wait(1 / 60)
        return

    frame = cp.empty((HEIGHT, WIDTH, 4), dtype=cp.uint8)
    with stream:
        while not stop.is_set():
            render(frame, time.monotonic() - start, x, y)
            # Reusing `frame` next iteration is safe: `stream` waits for fastgui's copy.
            viewport.submit_cuda(frame, stream=stream)
            status.set_text(f"submit_cuda: {viewport.cuda_status}")
            stop.wait(1 / 60)


def main() -> None:
    use_surface = "--surface" in sys.argv[1:]
    window = fg.Window(title="fastgui — CUDA viewport (unverified)", width=WIDTH, height=HEIGHT + 40)
    viewport = fg.Viewport()
    status = fg.Label("starting...")
    window.set_content(fg.Box(direction="column", padding=8.0, gap=8.0, children=[status, viewport]))

    # create_cuda_surface() needs the render thread, which run() starts (and run() blocks), so the
    # producer lives on a background thread. submit_cuda() works before run() too.
    stop = threading.Event()
    thread = threading.Thread(target=produce, args=(status, viewport, use_surface, stop), daemon=True)
    thread.start()
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
