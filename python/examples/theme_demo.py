"""M7 demo: themes. Every widget here leaves its colors out, so they all come from the current
`fg.Theme`. "Light" / "Dark" restyle the running window with `Window.set_theme`; "Orange accent"
tweaks one token with `Theme.replace`. Text typed into the field survives the switch.
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 themes", width=560, height=460)
    notes = fg.TextInput(placeholder="Type something, then switch themes")
    status = fg.Label("Dark theme")

    def use(theme: fg.Theme, name: str) -> None:
        window.set_theme(theme)
        status.set_text(name)

    rows = fg.ListView([f"Row {i}" for i in range(200)], height=120.0, flex_grow=0.0)
    menu = fg.Popup(fg.Label("A themed popup"))
    menu_button = fg.Button("Popup ▾", on_click=lambda: menu.show(menu_button))

    left = fg.Panel(
        title="Controls",
        content=fg.Box(
            direction="column",
            gap=10.0,
            padding=12.0,
            children=[
                fg.Box(
                    direction="row",
                    gap=8.0,
                    children=[
                        fg.Button("Light", on_click=lambda: use(fg.Theme.light(), "Light theme")),
                        fg.Button("Dark", on_click=lambda: use(fg.Theme.dark(), "Dark theme")),
                        fg.Button(
                            "Orange accent",
                            on_click=lambda: use(fg.get_theme().replace(accent=(1.0, 0.55, 0.1, 1.0)), "Orange accent"),
                        ),
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
            fg.Panel(title="List", content=fg.Box(direction="column", padding=8.0, children=[rows])),
            fg.Panel(title="About", content=fg.Box(padding=12.0, children=[fg.Label("Colors come from fg.Theme.")])),
        ]
    )
    window.set_content(fg.Splitter(left, right, ratio=0.55))
    window.run()


if __name__ == "__main__":
    main()
