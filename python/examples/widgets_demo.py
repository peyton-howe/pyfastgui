"""M4 demo: a Box layout containing a Label, a Button, and a Slider, all wired up with
callbacks. Proves layout (taffy), text rendering (cosmic-text + tiny-skia), and input
dispatch (click + drag) end to end.
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M4 widgets", width=480, height=320)

    count = 0
    count_label = fg.Label(f"Count: {count}", font_size=28.0)
    step_label = fg.Label("Step: 0.50", font_size=16.0, color=(0.7, 0.7, 0.8, 1.0))

    def increment() -> None:
        nonlocal count
        count += 1
        count_label.set_text(f"Count: {count}")

    def on_slider_change(value: float) -> None:
        step_label.set_text(f"Step: {value:.2f}")

    button = fg.Button("Click me", on_click=increment, font_size=18.0)
    slider = fg.Slider(value=0.5, min=0.0, max=1.0, on_change=on_slider_change)

    root = fg.Box(
        direction="column",
        gap=16.0,
        padding=24.0,
        background=(0.10, 0.11, 0.13, 1.0),
        children=[count_label, button, step_label, slider],
    )

    window.set_content(root)
    window.run()


if __name__ == "__main__":
    main()
