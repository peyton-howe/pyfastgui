"""Proves Window mutators are safe to call from any thread: several threads hammer
set_clear_color() concurrently while the main thread owns the render loop. No thread ever
touches render state directly -- every call is enqueued and applied by the render thread.
"""

import threading
import time

import fastgui as fg

PALETTE = [
    (0.75, 0.20, 0.20, 1.0),
    (0.20, 0.70, 0.25, 1.0),
    (0.20, 0.35, 0.80, 1.0),
    (0.80, 0.65, 0.15, 1.0),
    (0.55, 0.20, 0.75, 1.0),
]


def color_cycler(window: fg.Window, thread_index: int, stop: threading.Event) -> None:
    i = thread_index
    while not stop.is_set():
        r, g, b, a = PALETTE[i % len(PALETTE)]
        try:
            window.set_clear_color(r, g, b, a)
        except RuntimeError:
            return  # window closed while this thread was mid-loop
        i += 1
        stop.wait(0.05)


def main() -> None:
    window = fg.Window(title="fastgui — M1 threaded mutation", width=800, height=600)
    stop = threading.Event()
    threads = [
        threading.Thread(target=color_cycler, args=(window, i, stop), daemon=True)
        for i in range(8)
    ]
    for t in threads:
        t.start()

    try:
        window.run()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=1.0)


if __name__ == "__main__":
    main()
