"""M7 7C demo: Toolbar, StatusBar, Dialog, file/color pickers, GroupBox,
CollapsibleSection, and StackedWidget — plus the existing MenuBar."""

import sys

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — app chrome", width=640, height=480)
    status = fg.StatusBar("Ready.")

    def set_status(msg: str) -> None:
        status.set_text(msg)

    def quit_app() -> None:
        set_status("Quit")
        sys.exit(0)

    about = fg.Dialog(
        "About",
        fg.Label("fastgui M7 7C app chrome demo.", font_size="body"),
        buttons=[("OK", lambda: set_status("About closed"))],
        modal=True,
        on_dismiss=lambda: set_status("About dismissed"),
    )

    file_menu = fg.Menu(
        [
            fg.MenuItem("New", shortcut="Cmd+N", on_click=lambda: set_status("New")),
            fg.MenuItem(
                "Open…",
                shortcut="Cmd+O",
                on_click=lambda: set_status(
                    f"Open: {fg.open_file_dialog(title='Open', filter=[('Text', ['txt', 'md'])]) or 'cancelled'}"
                ),
            ),
            fg.MenuItem(
                "Save…",
                shortcut="Cmd+S",
                on_click=lambda: set_status(
                    f"Save: {fg.save_file_dialog(title='Save', filter=[('Text', ['txt'])]) or 'cancelled'}"
                ),
            ),
            fg.MenuSeparator(),
            fg.MenuItem("Quit", shortcut="Cmd+Q", on_click=quit_app),
        ]
    )
    help_menu = fg.Menu([fg.MenuItem("About…", on_click=lambda: about.show(window))])
    menubar = fg.MenuBar([("File", file_menu), ("Help", help_menu)])

    stack = fg.StackedWidget(
        [
            fg.Label("Page 1 — use the toolbar to switch pages.", font_size="body"),
            fg.Label("Page 2 — GroupBox and collapsible live below.", font_size="body"),
            fg.Label("Page 3 — dialogs and pickers are on the toolbar.", font_size="body"),
        ],
        index=0,
    )

    def go(i: int) -> None:
        stack.set_index(i)
        set_status(f"Page {i + 1}")

    toolbar = fg.Toolbar(
        [
            fg.Button("Page 1", on_click=lambda: go(0), tooltip="Show page 1"),
            fg.Button("Page 2", on_click=lambda: go(1), tooltip="Show page 2"),
            fg.Button("Page 3", on_click=lambda: go(2), tooltip="Show page 3"),
            fg.Box(width=12.0, children=[]),
            fg.Button(
                "Open…",
                on_click=lambda: set_status(
                    f"Open: {fg.open_file_dialog(title='Open file') or 'cancelled'}"
                ),
            ),
            fg.Button(
                "Save…",
                on_click=lambda: set_status(
                    f"Save: {fg.save_file_dialog(title='Save file') or 'cancelled'}"
                ),
            ),
            fg.Button(
                "Pick color…",
                on_click=lambda: fg.pick_color(
                    window,
                    initial=fg.get_theme().accent,
                    on_pick=lambda c: set_status(f"Color: {c}" if c else "Color cancelled"),
                ),
            ),
            fg.Button("About…", on_click=lambda: window.show_popup(about)),
        ]
    )

    group = fg.GroupBox(
        "Options",
        fg.Box(
            direction="column",
            gap="small",
            children=[
                fg.Checkbox("Remember window layout"),
                fg.Toggle(),
            ],
        ),
    )
    collapsible = fg.CollapsibleSection(
        "Advanced",
        fg.Box(
            direction="column",
            gap="small",
            children=[
                fg.Label("Hidden until expanded.", font_size="small"),
                fg.Slider(value=0.5),
            ],
        ),
        expanded=False,
    )

    body = fg.Box(
        direction="column",
        gap="medium",
        padding="large",
        flex_grow=1.0,
        children=[stack, group, collapsible],
    )

    window.set_content(
        fg.Box(
            direction="column",
            gap=0.0,
            children=[menubar, toolbar, body, status],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
