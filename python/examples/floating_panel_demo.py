"""M6 demo: floating panels — an always-on-top region simulated within the single window (not a
real second OS window; see ROADMAP.md's M6 status for why). Grab its title bar and drag it around
on top of the regular docked content. Not resizable and not re-dockable in this pass.
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fast-gui — M6 floating panel", width=900, height=560)

    dock = fg.DockArea()
    dock.add_panel(
        fg.Panel(
            title="Docked",
            content=fg.Box(
                direction="column", padding=16.0,
                children=[fg.Label("Regular docked content, underneath the floating panel.", font_size=14.0)],
            ),
        ),
        region="center",
    )
    window.set_content(dock)

    floating = fg.Panel(
        title="Floating",
        content=fg.Box(
            direction="column", padding=16.0,
            children=[fg.Label("Drag my title bar around.", font_size=14.0, color=(0.9, 0.9, 0.95, 1.0))],
        ),
        background=(0.16, 0.20, 0.28, 1.0),
        title_background=(0.22, 0.30, 0.45, 1.0),
    )
    window.add_floating_panel(floating, x=250.0, y=150.0, width=320.0, height=180.0)

    window.run()


if __name__ == "__main__":
    main()
