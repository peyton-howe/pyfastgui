"""M7 demo: `ScrollArea`. A long list of rows (labels, text inputs, buttons, sliders) with a live
numpy `Viewport` in the middle, all clipped to the scroll area. Scroll with the wheel/trackpad,
drag the overlay scrollbar, or Tab through the fields (focus scrolls them into view).
"""

import threading
import time

import numpy as np

import fastgui as fg

FEED_W, FEED_H = 320, 180


def frame(t: float) -> np.ndarray:
    y, x = np.mgrid[0:FEED_H, 0:FEED_W].astype(np.float32)
    r = 0.5 + 0.5 * np.sin(x / 25.0 + t)
    g = 0.5 + 0.5 * np.sin(y / 20.0 + t * 1.3)
    b = 0.5 + 0.5 * np.sin((x + y) / 30.0 + t * 0.7)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def main() -> None:
    window = fg.Window(title="fastgui — M7 scroll area", width=520, height=460)
    status = fg.Label("Scroll the list below.", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))
    viewport = fg.Viewport()

    rows = []
    for i in range(40):
        if i == 12:
            rows.append(fg.Box(height=FEED_H, children=[viewport]))
        elif i % 5 == 0:
            rows.append(fg.TextInput(placeholder=f"Field {i}", on_submit=lambda text, i=i: status.set_text(f"Field {i}: {text!r}")))
        elif i % 5 == 2:
            rows.append(fg.Button(f"Button {i}", on_click=lambda i=i: status.set_text(f"Clicked button {i}")))
        elif i % 5 == 3:
            rows.append(fg.Slider(value=i / 40, on_change=lambda v, i=i: status.set_text(f"Slider {i}: {v:.2f}")))
        else:
            rows.append(fg.Label(f"Row {i}", font_size=16.0))

    scroll = fg.ScrollArea(
        fg.Box(direction="column", gap=10.0, padding=12.0, children=rows),
        background=(0.13, 0.14, 0.17, 1.0),
    )
    top = fg.Button("Back to top", on_click=lambda: scroll.scroll_to(0, 0))

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[fg.Box(direction="row", gap=12.0, children=[status, top]), scroll],
        )
    )

    stop = threading.Event()

    def feed() -> None:
        start = time.monotonic()
        while not stop.is_set():
            viewport.submit_frame(frame(time.monotonic() - start))
            stop.wait(1 / 30)

    threading.Thread(target=feed, daemon=True).start()
    try:
        window.run()
    finally:
        stop.set()


if __name__ == "__main__":
    main()
