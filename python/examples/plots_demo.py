"""M7 7E demo: GPU plot widgets fed from numpy — line, scatter, and heatmap.

Each plot rasterizes into an RGBA frame and uploads through the same Image/Viewport GPU
layer path, so `set_data` is safe from a background thread.
"""

import threading
import time

import numpy as np

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 GPU plots", width=1100, height=860)
    status = fg.Label("Streaming sine into PlotLine…", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    x = np.linspace(0.0, 4.0 * np.pi, 400)
    # Plots have no intrinsic size: inside a ScrollArea (unbounded height) give each one a height,
    # and in a row give them flex_grow so they share its width.
    line = fg.PlotLine(pixel_width=640, pixel_height=220, height=220, flex_grow=0, interactive=True)
    line.set_series(
        [
            (x, np.sin(x), (0.35, 0.75, 1.0, 1.0)),
            (x, np.cos(x), (0.95, 0.55, 0.35, 1.0)),
        ]
    )
    scatter = fg.PlotScatter(
        np.random.default_rng(0).normal(size=200),
        np.random.default_rng(1).normal(size=200),
        color=(1.0, 0.55, 0.25, 1.0),
        point_radius=2.5,
        pixel_width=640,
        pixel_height=220,
        height=220,
        flex_grow=0,
    )
    yy, xx = np.mgrid[0:64, 0:96]
    field = np.sin(xx / 8.0) * np.cos(yy / 6.0)
    heat = fg.PlotHeatmap(field, colormap="viridis", pixel_width=640, pixel_height=220, height=220, flex_grow=0)
    hist = fg.PlotHistogram(np.random.default_rng(0).normal(size=800), bins=28, pixel_width=640, pixel_height=220, height=200, flex_grow=1)
    bar = fg.PlotBar([3, 5, 2, 7, 4, 6], color=(0.95, 0.55, 0.35, 1.0), pixel_width=640, pixel_height=220, height=200, flex_grow=1)
    contour = fg.PlotContour(field, levels=8, filled=True, pixel_width=640, pixel_height=220, height=200, flex_grow=0)
    z = field[::2, ::2]
    surface = fg.PlotSurface(
        z,
        x=xx[0, ::2],
        y=yy[::2, 0],
        pixel_width=640,
        pixel_height=280,
        height=240,
        flex_grow=1,
    )
    scatter3d = fg.PlotScatter3D(
        xx[::4, ::4].ravel(),
        yy[::4, ::4].ravel(),
        field[::4, ::4].ravel(),
        point_radius=4.0,
        pixel_width=640,
        pixel_height=280,
        height=240,
        flex_grow=1,
    )
    # A small pyramid so the mesh path is obvious next to the height field.
    mesh_v = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 1.2]],
        dtype=np.float64,
    )
    mesh_f = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (0, 2, 1), (0, 3, 2)]
    mesh = fg.PlotMesh(mesh_v, mesh_f, colormap="magma", pixel_width=640, pixel_height=280, height=240, flex_grow=1)

    stop = threading.Event()

    def animate() -> None:
        t0 = time.monotonic()
        while not stop.is_set():
            t = time.monotonic() - t0
            # set_series keeps the current zoom/pan.
            line.set_series(
                [
                    (x, np.sin(x + t) * np.cos(0.25 * t), (0.35, 0.75, 1.0, 1.0)),
                    (x, np.cos(x + t), (0.95, 0.55, 0.35, 1.0)),
                ]
            )
            stop.wait(1 / 30)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    window.set_content(
        fg.ScrollArea(fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Label("PlotLine — wheel zoom, drag pan, double-click reset (zoom survives live updates)", font_size="small"),
                line,
                fg.Label("PlotScatter", font_size="small"),
                scatter,
                fg.Label("PlotHeatmap (viridis)", font_size="small"),
                heat,
                fg.Label("PlotHistogram / PlotBar", font_size="small"),
                fg.Box(direction="row", gap=10.0, flex_grow=0.0, children=[hist, bar]),
                fg.Label("PlotContour (filled) — same grid as the heatmap", font_size="small"),
                contour,
                fg.Label("PlotSurface / PlotScatter3D / PlotMesh — drag to orbit", font_size="small"),
                fg.Box(direction="row", gap=10.0, flex_grow=0.0, children=[surface, scatter3d, mesh]),
                status,
            ],
        ))
    )
    try:
        window.run()
    finally:
        stop.set()
        thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
