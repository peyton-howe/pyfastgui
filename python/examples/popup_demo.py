"""M7 demo: `Popup` overlays. "Menu ▾" opens a menu below the button (click outside or Escape
to dismiss; Tab stays inside it). "Rename…" opens a modal dialog centered over a dimmed window.
"Popup at point" opens one at a fixed spot. Everything draws over the live `Viewport` video.
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
    window = fg.Window(title="fastgui — M7 popups", width=560, height=420)
    status = fg.Label("Open a popup.", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))
    title = fg.Label("Untitled", font_size=20.0)
    viewport = fg.Viewport()

    def choose(name: str) -> None:
        status.set_text(f"Chose {name}")
        menu.close()

    menu = fg.Popup(
        fg.Box(
            direction="column",
            gap=4.0,
            children=[fg.Button(n, on_click=lambda n=n: choose(n)) for n in ("New", "Open…", "Save", "Quit")],
        ),
        on_dismiss=lambda: status.set_text("Menu dismissed"),
    )
    # Lambdas look `menu` / `menu_button` up when clicked, so defining them in any order is fine.
    menu_button = fg.Button("Menu ▾", on_click=lambda: menu.show(menu_button))

    name_field = fg.TextInput(placeholder="New title", width=220.0, on_submit=lambda _: rename())

    def rename() -> None:
        if name_field.text:
            title.set_text(name_field.text)
            status.set_text(f"Renamed to {name_field.text!r}")
        dialog.close()

    dialog = fg.Popup(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=10.0,
            children=[
                fg.Label("Rename", font_size=16.0),
                name_field,
                fg.Box(
                    direction="row",
                    gap=8.0,
                    children=[fg.Button("OK", on_click=rename), fg.Button("Cancel", on_click=lambda: dialog.close())],
                ),
            ],
        ),
        modal=True,
        on_dismiss=lambda: status.set_text("Dialog cancelled (Escape)"),
    )

    point_popup = fg.Popup(fg.Label("I'm at (360, 60). Click elsewhere.", font_size=13.0))

    window.set_content(
        fg.Box(
            direction="column",
            gap=12.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Box(
                    direction="row",
                    gap=10.0,
                    children=[
                        menu_button,
                        fg.Button("Rename…", on_click=lambda: window.show_popup(dialog)),
                        fg.Button("Popup at point", on_click=lambda: window.show_popup(point_popup, 360.0, 60.0)),
                    ],
                ),
                title,
                status,
                fg.Box(height=FEED_H, children=[viewport]),
            ],
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
