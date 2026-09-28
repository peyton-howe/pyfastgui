"""M7 7B demo: checkbox, radio, toggle, spin, scrub, progress, combo, grid, image, text area."""

import numpy as np

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 form controls", width=640, height=680)
    status = fg.Label("Try the controls…", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    def set_status(text: str) -> None:
        status.set_text(text)

    checkbox = fg.Checkbox("Enable notifications", on_change=lambda v: set_status(f"checkbox={v}"))
    toggle = fg.Toggle(on_change=lambda v: set_status(f"toggle={v}"))
    notes = fg.TextArea(
        placeholder="Notes (Enter = newline, Cmd/Ctrl+Enter = submit)",
        height=96.0,
        flex_grow=1.0,
        on_change=lambda t: set_status(f"notes={len(t)} chars"),
        on_submit=lambda t: set_status(f"submitted notes ({len(t)} chars)"),
    )

    group = 42
    radios = fg.Box(
        direction="column",
        gap=6.0,
        children=[
            fg.Radio("Small", group=group, selected=True, on_select=lambda: set_status("size=Small")),
            fg.Radio("Medium", group=group, on_select=lambda: set_status("size=Medium")),
            fg.Radio("Large", group=group, on_select=lambda: set_status("size=Large")),
        ],
    )

    progress = fg.ProgressBar(value=0.35)
    spin = fg.SpinBox(
        value=35,
        min=0,
        max=100,
        step=5,
        decimals=0,
        width=120.0,
        on_change=lambda v: (progress.set_value(v / 100.0), set_status(f"spin={v:.0f}")),
    )
    scrub = fg.NumericScrub(
        value=12.5,
        min=0.0,
        max=50.0,
        speed=0.1,
        decimals=1,
        width=120.0,
        on_change=lambda v: set_status(f"scrub={v:.1f}"),
    )
    fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi"]
    combo = fg.ComboBox(
        items=fruits,
        placeholder="Pick a fruit…",
        width=200.0,
        on_change=lambda i: set_status(f"combo={fruits[i]}"),
    )

    image = fg.Image(width=160.0, height=96.0, flex_grow=0.0)
    # Simple gradient checker so the GPU image path is visibly exercised.
    h, w = 96, 160
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack(
        [
            (xx * 255 // max(w - 1, 1)).astype(np.uint8),
            (yy * 255 // max(h - 1, 1)).astype(np.uint8),
            np.full((h, w), 180, dtype=np.uint8),
        ],
        axis=-1,
    )
    image.set_image(rgb)

    grid = fg.Grid(
        columns=2,
        gap=12.0,
        children=[
            fg.Label("Checkbox", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            checkbox,
            fg.Label("Toggle", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            fg.Box(direction="row", gap=8.0, children=[toggle, fg.Label("Dark mode", font_size=14.0)]),
            fg.Label("Radio", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            radios,
            fg.Label("SpinBox", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            spin,
            fg.Label("NumericScrub", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            scrub,
            fg.Label("ComboBox", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            combo,
            fg.Label("Progress", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            progress,
            fg.Label("Image", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            image,
            fg.Label("TextArea", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            notes,
        ],
    )

    root = fg.Box(
        direction="column",
        gap=16.0,
        padding=24.0,
        background=(0.10, 0.11, 0.13, 1.0),
        children=[
            fg.Label("Tier 1 form controls", font_size=18.0),
            grid,
            status,
        ],
    )
    window.set_content(root)
    window.run()


if __name__ == "__main__":
    main()
