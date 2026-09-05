"""Synthetic "live camera feed" demo: a background thread generates animated RGB frames with
numpy and pushes them into a Viewport at ~30 FPS while the main thread owns the render loop.

Swap generate_frame() for something like `cv2.VideoCapture(0).read()` (BGR -> RGB) to drive
this from a real camera instead -- submit_frame() doesn't care where the pixels came from, it
just wants a (height, width, 3-or-4) uint8 array.
"""

import threading
import time

import numpy as np

import fastgui as fg

WIDTH, HEIGHT = 640, 480


def generate_frame(t: float) -> np.ndarray:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    x /= WIDTH
    y /= HEIGHT
    r = 0.5 + 0.5 * np.sin(6.0 * x + t)
    g = 0.5 + 0.5 * np.sin(6.0 * y + t * 1.3)
    b = 0.5 + 0.5 * np.sin(6.0 * (x + y) + t * 0.7)
    frame = np.stack([r, g, b], axis=-1)
    return (frame * 255).astype(np.uint8)


def camera_thread(viewport: fg.Viewport, stop: threading.Event) -> None:
    start = time.monotonic()
    while not stop.is_set():
        viewport.submit_frame(generate_frame(time.monotonic() - start))
        stop.wait(1 / 30)


def main() -> None:
    window = fg.Window(title="fast-gui — M2 live feed", width=WIDTH, height=HEIGHT)
    viewport = fg.Viewport()
    window.set_viewport(viewport)

    stop = threading.Event()
    thread = threading.Thread(target=camera_thread, args=(viewport, stop), daemon=True)
    thread.start()

    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
