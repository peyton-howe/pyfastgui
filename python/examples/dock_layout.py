"""M6 demo: a `DockArea` with three docked `Panel`s (a live `Viewport` in the center, a
right-side "Controls" panel, and a bottom "Log" panel), each resizable via draggable
`Splitter`s.
"""

import threading
import time

import numpy as np

import fastgui as fg

FEED_W, FEED_H = 640, 360


def generate_frame(t: float) -> np.ndarray:
    y, x = np.mgrid[0:FEED_H, 0:FEED_W].astype(np.float32)
    x /= FEED_W
    y /= FEED_H
    r = 0.5 + 0.5 * np.sin(6.0 * x + t)
    g = 0.5 + 0.5 * np.sin(6.0 * y + t * 1.3)
    b = 0.5 + 0.5 * np.sin(6.0 * (x + y) + t * 0.7)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def camera_thread(viewport: fg.Viewport, stop: threading.Event) -> None:
    start = time.monotonic()
    while not stop.is_set():
        viewport.submit_frame(generate_frame(time.monotonic() - start))
        stop.wait(1 / 30)


def main() -> None:
    window = fg.Window(title="fast-gui — M6 docking", width=960, height=600)

    count = 0
    count_label = fg.Label("Count: 0", font_size=22.0)

    def increment() -> None:
        nonlocal count
        count += 1
        count_label.set_text(f"Count: {count}")

    controls = fg.Box(
        direction="column",
        gap=12.0,
        padding=16.0,
        children=[
            count_label,
            fg.Button("Click me", on_click=increment, font_size=16.0),
            fg.Slider(value=0.5, min=0.0, max=1.0),
        ],
    )

    center = fg.Viewport()

    log = fg.Box(
        direction="column",
        padding=12.0,
        children=[fg.Label("Log panel", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0))],
    )

    dock = fg.DockArea()
    dock.add_panel(fg.Panel(title="Viewport", content=center), region="center")
    dock.add_panel(fg.Panel(title="Controls", content=controls), region="right", size=0.28)
    dock.add_panel(fg.Panel(title="Log", content=log), region="bottom", size=0.22)

    window.set_content(dock)

    stop = threading.Event()
    thread = threading.Thread(target=camera_thread, args=(center, stop), daemon=True)
    thread.start()
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
