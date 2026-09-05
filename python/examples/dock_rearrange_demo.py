"""M6 demo: drag-to-rearrange. Four docked panels — grab any panel's title bar and drop it on
another panel's edges (splits that region) or its center (merges the two into a new `Tabs`
group). A translucent overlay previews where it'll land while you drag.

Known gaps this demo can't exercise (see ROADMAP.md's M6 status / DockArea's docstring): a panel
that's already inside a `Tabs` group isn't itself draggable yet, and dropping onto an existing
`Tabs`' center isn't supported — only forming a *new* two-tab group by dropping one Panel onto
another.
"""

import fastgui as fg


def make_panel(title: str, body: str, color: tuple[float, float, float, float]) -> fg.Panel:
    return fg.Panel(
        title=title,
        content=fg.Box(direction="column", padding=16.0, children=[fg.Label(body, font_size=14.0, color=color)]),
    )


def main() -> None:
    window = fg.Window(title="fast-gui — M6 drag-to-rearrange", width=1000, height=620)

    grey = (0.75, 0.78, 0.82, 1.0)
    panel_a = make_panel("Viewport", "Drag my title bar onto another panel.", grey)
    panel_b = make_panel("Controls", "Drop on an edge to split, center to tab.", grey)
    panel_c = make_panel("Inspector", "Third panel.", grey)
    panel_d = make_panel("Log", "Fourth panel.", grey)

    dock = fg.DockArea()
    dock.add_panel(panel_a, region="center")
    dock.add_panel(panel_b, region="right", size=0.3)
    dock.add_panel(panel_c, region="bottom", size=0.25)
    dock.add_panel(panel_d, region="left", size=0.2)

    window.set_content(dock)
    window.run()


if __name__ == "__main__":
    main()
