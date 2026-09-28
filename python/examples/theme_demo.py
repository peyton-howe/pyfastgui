"""M7 demo: themes. Widgets here leave out their colors and font sizes (or name them —
`font_size="large"`, `gap="medium"`), so they all come from the current `fg.Theme`:

- "Light" / "Dark" restyle the running window with `Window.set_theme` (grey vs blue buttons).
- "Orange accent" turns the accent (focus ring, caret, slider thumb) and buttons orange.
- "Bigger text & spacing" raises the font-size and spacing tokens; the layout grows with them.
- "Monospace font" switches every piece of text to a monospace family.

Text typed into the field survives each switch.
"""

import sys

import fastgui as fg

MONOSPACE = {"darwin": "Menlo", "win32": "Consolas"}.get(sys.platform, "DejaVu Sans Mono")


def main() -> None:
    window = fg.Window(title="fastgui — M7 themes", width=640, height=500)
    notes = fg.TextInput(placeholder="Type something, then switch themes")
    status = fg.Label("Dark theme", font_size="small")

    def use(theme: fg.Theme, name: str) -> None:
        window.set_theme(theme)
        status.set_text(name)

    def orange(theme: fg.Theme) -> fg.Theme:
        return theme.replace(accent=(1.0, 0.55, 0.1, 1.0), button=(0.9, 0.45, 0.05, 1.0), button_text=(1.0, 1.0, 1.0, 1.0))

    def bigger(theme: fg.Theme) -> fg.Theme:
        return theme.replace(
            font_size_small=theme.font_size_small + 3,
            font_size=theme.font_size + 4,
            font_size_large=theme.font_size_large + 6,
            spacing_small=theme.spacing_small * 2,
            spacing=theme.spacing * 2,
            spacing_large=theme.spacing_large * 1.5,
        )

    rows = fg.ListView([f"Row {i}" for i in range(200)], height=120.0, flex_grow=0.0)
    menu = fg.Popup(fg.Label("A themed popup"), padding="medium")
    menu_button = fg.Button("Popup ▾", on_click=lambda: menu.show(menu_button))

    def action(label: str, make) -> fg.Button:
        return fg.Button(label, on_click=lambda: use(make(fg.get_theme()), label))

    left = fg.Panel(
        title="Controls",
        content=fg.Box(
            direction="column",
            gap="medium",
            padding="large",
            children=[
                fg.Label("Theme tokens", font_size="large"),
                fg.Box(
                    direction="row",
                    gap="small",
                    children=[
                        fg.Button("Light", on_click=lambda: use(fg.Theme.light(), "Light theme")),
                        fg.Button("Dark", on_click=lambda: use(fg.Theme.dark(), "Dark theme")),
                        action("Orange accent", orange),
                    ],
                ),
                fg.Box(
                    direction="row",
                    gap="small",
                    children=[
                        action("Bigger text & spacing", bigger),
                        action("Monospace font", lambda t: t.replace(font_family=MONOSPACE)),
                        action("Default font", lambda t: t.replace(font_family=None)),
                    ],
                ),
                notes,
                fg.Slider(value=0.4),
                menu_button,
                status,
            ],
        ),
    )
    right = fg.Tabs(
        [
            fg.Panel(title="List", content=fg.Box(direction="column", padding="small", children=[rows])),
            fg.Panel(title="About", content=fg.Box(padding="large", children=[fg.Label("Everything here comes from fg.Theme.")])),
        ]
    )
    window.set_content(fg.Splitter(left, right, ratio=0.6))
    window.run()


if __name__ == "__main__":
    main()
