"""M7 demo: `TextInput` fields with keyboard focus. Click or Tab into a field and type;
select with Shift+arrows, a drag, or a double-click; Cmd/Ctrl+A/C/X/V/Z work; Enter submits.
Tab / Shift+Tab also reach the button and slider (Space clicks, arrows slide).
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 text input", width=520, height=360)

    echo = fg.Label("Type something…", font_size=16.0, color=(0.7, 0.75, 0.85, 1.0))
    submitted = fg.Label("", font_size=14.0, color=(0.55, 0.8, 0.6, 1.0))

    def on_name(text: str) -> None:
        echo.set_text(f"Hello, {text}!" if text else "Type something…")

    def on_submit(text: str) -> None:
        submitted.set_text(f"Submitted: {text!r}")

    name = fg.TextInput(placeholder="Your name", on_change=on_name, on_submit=on_submit)
    notes = fg.TextInput(
        "A longer line that is wider than the field, to show horizontal scrolling as you type.",
        on_submit=on_submit,
    )

    def clear() -> None:
        name.set_text("")
        on_name("")
        submitted.set_text(f"Cleared (notes still says {notes.text[:16]!r}…)")

    root = fg.Box(
        direction="column",
        gap=12.0,
        padding=24.0,
        background=(0.10, 0.11, 0.13, 1.0),
        children=[
            fg.Label("Name", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            name,
            fg.Label("Notes", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0)),
            notes,
            echo,
            fg.Box(direction="row", gap=12.0, children=[fg.Button("Clear name", on_click=clear), fg.Box(flex_grow=1.0, children=[fg.Slider(value=0.5)])]),
            submitted,
        ],
    )

    window.set_content(root)
    window.run()


if __name__ == "__main__":
    main()
