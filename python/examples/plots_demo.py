"""M7 7E demo: GPU plot widgets fed from numpy — line, scatter, and heatmap.

Each plot rasterizes into an RGBA frame and uploads through the same Image/Viewport GPU
layer path, so `set_data` is safe from a background thread.
"""

import threading
import time

import numpy as np

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 GPU plots", width=960, height=720)
    status = fg.Label("Streaming sine into PlotLine…", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    x = np.linspace(0.0, 4.0 * np.pi, 400)
    line = fg.PlotLine(x, np.sin(x), color=(0.35, 0.75, 1.0, 1.0), pixel_width=640, pixel_height=220)
    scatter = fg.PlotScatter(
        np.random.default_rng(0).normal(size=200),
        np.random.default_rng(1).normal(size=200),
        color=(1.0, 0.55, 0.25, 1.0),
        point_radius=2.5,
        pixel_width=640,
        pixel_height=220,
    )
    yy, xx = np.mgrid[0:64, 0:96]
    heat = fg.PlotHeatmap(
        np.sin(xx / 8.0) * np.cos(yy / 6.0),
        colormap="viridis",
        pixel_width=640,
        pixel_height=220,
    )

    stop = threading.Event()

    def animate() -> None:
        t0 = time.monotonic()
        while not stop.is_set():
            t = time.monotonic() - t0
            line.set_data(x, np.sin(x + t) * np.cos(0.25 * t))
            stop.wait(1 / 30)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Label("PlotLine (live numpy)", font_size="small"),
                line,
                fg.Label("PlotScatter", font_size="small"),
                scatter,
                fg.Label("PlotHeatmap (viridis)", font_size="small"),
                heat,
                status,
            ],
        )
    )
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
