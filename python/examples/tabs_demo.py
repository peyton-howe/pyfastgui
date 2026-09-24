"""M6 demo: `Tabs` inside a `DockArea` region — multiple `Panel`s sharing one spot, switched by
clicking a header segment. Proves tab-bar rendering, click-to-switch, and that `Tabs` composes
into `DockArea`/`Splitter` like any other widget (no special-casing needed there).
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M6 tabs", width=900, height=560)

    count = 0
    count_label = fg.Label("Count: 0", font_size=20.0)

    def increment() -> None:
        nonlocal count
        count += 1
        count_label.set_text(f"Count: {count}")

    tab_a = fg.Panel(
        title="Overview",
        content=fg.Box(
            direction="column", padding=16.0, gap=12.0,
            children=[fg.Label("First tab.", font_size=14.0), count_label, fg.Button("Click me", on_click=increment)],
        ),
    )
    tab_b = fg.Panel(
        title="Settings",
        content=fg.Box(
            direction="column", padding=16.0,
            children=[fg.Label("Second tab — a slider.", font_size=14.0), fg.Slider(value=0.3)],
        ),
    )
    tab_c = fg.Panel(
        title="Logs",
        content=fg.Box(
            direction="column", padding=16.0,
            children=[fg.Label("Third tab.", font_size=14.0, color=(0.6, 0.65, 0.7, 1.0))],
        ),
    )

    tabs = fg.Tabs([tab_a, tab_b, tab_c])

    sidebar = fg.Panel(
        title="Sidebar",
        content=fg.Box(
            direction="column", padding=12.0,
            children=[fg.Label("Not tabbed.", font_size=13.0, color=(0.6, 0.65, 0.7, 1.0))],
        ),
    )

    dock = fg.DockArea()
    dock.add_panel(tabs, region="center")
    dock.add_panel(sidebar, region="right", size=0.28)

    window.set_content(dock)
    window.run()


if __name__ == "__main__":
    main()
