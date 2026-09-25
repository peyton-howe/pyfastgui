"""GPU-to-GPU viewport demo: a CUDA kernel writes directly into the texture Vulkan displays,
with no CPU round-trip (contrast with live_camera_feed.py's submit_frame(), which copies a
numpy array through the CPU every frame).

*** UNVERIFIED — READ BEFORE RUNNING ***
This was written on a machine with no NVIDIA GPU. The CUDA<->Vulkan external-memory/timeline-
semaphore interop it exercises (fastgui_interop_cuda, and the matching Vulkan-side export in
fastgui-render-vk's cuda_texture module) has never been run against a real CUDA driver. Treat
this as a first draft to validate on real hardware, not as a working example. If it doesn't
work, the bug is almost certainly in the untested interop plumbing, not in this script.

Requires: an NVIDIA GPU, the CUDA driver, and cupy (`pip install cupy-cuda12x` or similar,
matching your installed CUDA version) — none of which this repo depends on otherwise.
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


def cuda_thread(surface: fg.CudaSurface, stop: threading.Event) -> None:
    # `pitch` is the real per-row byte stride Vulkan allocated (may be larger than width * 4
    # if the driver pads rows) -- respect it via explicit strides rather than assuming a
    # tightly packed buffer.
    mem = cp.cuda.UnownedMemory(surface.device_ptr, surface.pitch * surface.height, owner=surface)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    frame = cp.ndarray(
        (surface.height, surface.width, 4),
        dtype=cp.uint8,
        memptr=memptr,
        strides=(surface.pitch, 4, 1),
    )

    y, x = cp.mgrid[0:HEIGHT, 0:WIDTH].astype(cp.float32)
    x /= WIDTH
    y /= HEIGHT

    start = time.monotonic()
    while not stop.is_set():
        t = time.monotonic() - start
        r = 0.5 + 0.5 * cp.sin(6.0 * x + t)
        g = 0.5 + 0.5 * cp.sin(6.0 * y + t * 1.3)
        b = 0.5 + 0.5 * cp.sin(6.0 * (x + y) + t * 0.7)
        frame[..., 0] = (r * 255).astype(cp.uint8)
        frame[..., 1] = (g * 255).astype(cp.uint8)
        frame[..., 2] = (b * 255).astype(cp.uint8)
        frame[..., 3] = 255
        cp.cuda.Stream.null.synchronize()  # make sure our own writes are done before signalling
        surface.signal_ready()
        stop.wait(1 / 60)


def main() -> None:
    window = fg.Window(title="fastgui — M3 CUDA viewport (unverified)", width=WIDTH, height=HEIGHT)
    viewport = fg.Viewport()
    window.set_viewport(viewport)

    # create_cuda_surface() needs the render thread running, so it must come after
    # set_viewport() but the actual call happens once run() has started pumping commands --
    # do it from the same background thread that will drive the CUDA kernel. (Called on this
    # thread before run(), it raises RuntimeError after a short grace period.)
    stop = threading.Event()

    def start_cuda() -> None:
        try:
            surface = viewport.create_cuda_surface(WIDTH, HEIGHT)
        except RuntimeError as err:
            print(f"create_cuda_surface failed: {err}", file=sys.stderr)
            return
        cuda_thread(surface, stop)

    thread = threading.Thread(target=start_cuda, daemon=True)
    thread.start()

    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
