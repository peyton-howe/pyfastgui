"""M7 7C demo: MenuBar, tooltips, context menus, accelerators, Toolbar, StatusBar."""

import sys

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — menus / tooltips / accelerators", width=520, height=360)
    status = fg.StatusBar("Ready.")

    def set_status(msg: str) -> None:
        status.set_text(msg)

    def quit_app() -> None:
        set_status("Quit")
        sys.exit(0)

    recent = fg.Menu(
        [
            fg.MenuItem("report.py", icon="📄", on_click=lambda: set_status("Open recent: report.py")),
            fg.MenuItem("notes.md", icon="📄", on_click=lambda: set_status("Open recent: notes.md")),
            fg.MenuSeparator(),
            fg.MenuItem("Clear recent", on_click=lambda: set_status("Recent cleared")),
        ]
    )
    file_menu = fg.Menu(
        [
            fg.MenuItem("New", shortcut="Cmd+N", on_click=lambda: set_status("New")),
            fg.MenuItem("Open…", shortcut="Cmd+O", on_click=lambda: set_status("Open")),
            fg.MenuItem("Open recent", submenu=recent),
            fg.MenuSeparator(),
            fg.MenuItem("Save", shortcut="Cmd+S", on_click=lambda: set_status("Saved (Cmd/Ctrl+S)")),
            fg.MenuSeparator(),
            fg.MenuItem("Quit", shortcut="Cmd+Q", on_click=quit_app),
        ]
    )

    wrap_item = fg.MenuItem("Word wrap", checked=True)

    def toggle_wrap():
        wrap_item.checked = not bool(wrap_item.checked)
        set_status(f"Word wrap {'on' if wrap_item.checked else 'off'}")

    wrap_item.on_click = toggle_wrap

    dark_item = fg.MenuItem("Dark", radio_group="theme", checked=True)
    light_item = fg.MenuItem("Light", radio_group="theme", checked=False)

    def set_theme(name: str):
        dark_item.checked = name == "dark"
        light_item.checked = name == "light"
        set_status(f"Theme: {name}")

    dark_item.on_click = lambda: set_theme("dark")
    light_item.on_click = lambda: set_theme("light")

    view_menu = fg.Menu(
        [
            wrap_item,
            fg.MenuSeparator(),
            dark_item,
            light_item,
            fg.MenuItem("Disabled item", enabled=False),
        ]
    )
    edit_menu = fg.Menu(
        [
            fg.MenuItem("Copy status", shortcut="Cmd+Shift+C", on_click=lambda: set_status("Copied status line")),
            fg.MenuItem("Clear status", on_click=lambda: set_status("Ready.")),
        ]
    )
    menubar = fg.MenuBar([("File", file_menu), ("Edit", edit_menu), ("View", view_menu)])
    toolbar = fg.Toolbar(
        [
            fg.Button("New", on_click=lambda: set_status("Toolbar: New"), tooltip="New document"),
            fg.Button("Save", on_click=lambda: set_status("Toolbar: Save"), tooltip="Save"),
        ]
    )

    tip_label = fg.Label(
        "Hover me for a tooltip",
        tooltip="Tooltips open after a short delay and do not steal focus.",
    )
    tip_button = fg.Button(
        "Hover button",
        tooltip="Buttons can have tooltips too.",
        on_click=lambda: set_status("Button clicked"),
    )

    ctx = fg.Menu(
        [
            fg.MenuItem("From context", on_click=lambda: set_status("Context: From context")),
            fg.MenuSeparator(),
            fg.MenuItem("Ping", on_click=lambda: set_status("Context: Ping")),
        ]
    )
    panel = fg.Box(
        direction="column",
        gap=8.0,
        padding=12.0,
        background=(0.14, 0.15, 0.17, 1.0),
        context_menu=ctx,
        children=[
            fg.Label("Right-click this panel for a context menu.", color=(0.75, 0.78, 0.85, 1.0)),
            tip_label,
            tip_button,
            fg.Checkbox("Remember me", tooltip="Checkbox tooltip"),
            fg.Slider(value=0.4, tooltip="Drag me"),
        ],
    )

    window.set_content(
        fg.Box(
            direction="column",
            gap=0.0,
            children=[
                menubar,
                toolbar,
                fg.Box(direction="column", gap=12.0, padding=16.0, flex_grow=1.0, children=[panel]),
                status,
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()

