"""M7 7E demo: image viewer, log, code editor, command palette, binding, gauge, timeline, nodes.

Plots stay on the CPU-raster / GPU image-layer path (see plots_demo.py). Ctrl+K opens the
command palette. The log and gauge update from a background thread.
"""

import math
import threading
import time

import numpy as np

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 tier 4", width=1100, height=780)
    status = fg.Label("Ctrl+K command palette", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    yy, xx = np.mgrid[0:48, 0:64]
    field = np.sin(xx / 7.0) * np.cos(yy / 5.0)
    viewer = fg.ImageViewer(field, colormap="viridis", on_readout=lambda text: status.set_text(text), height=220.0)

    log = fg.LogView(height=140.0)
    log.append("log ready — appends are safe from any thread")

    editor = fg.CodeEditor(
        "def hello(name):\n    # highlighted\n    return f'hi {name}'\n",
        height=120.0,
        flex_grow=0.0,
    )

    gauge = fg.Gauge(value=0.35, min=0.0, max=1.0, height=140.0, flex_grow=0.0)
    timeline = fg.Timeline(
        tracks=[[(0.5, 2.0), (3.0, 5.5, (0.35, 0.75, 1.0, 1.0))], [(1.0, 4.0)]],
        duration=8.0,
        time=1.5,
        height=100.0,
    )
    graph = fg.NodeGraph(
        nodes=[(16.0, 36.0, 150.0, 68.0, "SRC"), (250.0, 28.0, 150.0, 68.0, "MIX"), (470.0, 40.0, 150.0, 68.0, "OUT")],
        edges=[(0, 1), (1, 2)],
        pixel_width=680,
        pixel_height=140,
        height=130.0,
    )

    gain = fg.Observable(0.35)
    title = fg.Observable("clip")

    def reset_viewer() -> None:
        viewer.reset_view()
        status.set_text("view reset")

    def mark_log() -> None:
        log.append("palette: mark")

    palette = fg.CommandPalette(
        [
            ("Reset image view", reset_viewer),
            ("Mark log", mark_log),
            ("Gauge half", lambda: gauge.set_value(0.5)),
        ]
    )

    stop = threading.Event()

    def animate() -> None:
        t0 = time.monotonic()
        n = 0
        while not stop.is_set():
            t = time.monotonic() - t0
            gauge.set_value(0.5 + 0.5 * math.sin(t))
            timeline.set_time((t * 0.8) % 8.0)
            n += 1
            if n % 15 == 0:
                log.append(f"tick {n} t={t:.1f}")
            stop.wait(1 / 30)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    root = fg.Box(
        direction="column",
        gap=8.0,
        padding=12.0,
        flex_grow=1.0,
        background=(0.10, 0.11, 0.13, 1.0),
        accelerators=palette.accelerators(),
        children=[
            fg.Label("Image viewer (wheel / drag / double-click) and live readout", font_size="small"),
            viewer,
            fg.Box(
                direction="row",
                gap=8.0,
                flex_grow=0.0,
                children=[
                    fg.Box(direction="column", gap=4.0, flex_grow=1.0, children=[fg.Label("Gauge", font_size="small"), gauge]),
                    fg.Box(
                        direction="column",
                        gap=4.0,
                        flex_grow=1.0,
                        children=[
                            fg.Label("Bound gain", font_size="small"),
                            gain.slider(min=0.0, max=1.0),
                            gain.bind_label(fmt=lambda v: f"gain {float(v):.2f}"),
                            title.text_input(),
                        ],
                    ),
                ],
            ),
            fg.Label("Timeline", font_size="small"),
            timeline,
            fg.Label("Node graph — drag a node", font_size="small"),
            graph,
            fg.Box(
                direction="row",
                gap=8.0,
                flex_grow=0.0,
                children=[
                    fg.Box(direction="column", gap=4.0, flex_grow=1.0, children=[fg.Label("Code", font_size="small"), editor]),
                    fg.Box(direction="column", gap=4.0, flex_grow=1.0, children=[fg.Label("Log", font_size="small"), log]),
                ],
            ),
            status,
        ],
    )
    palette.bind(root)
    window.set_content(root)
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
