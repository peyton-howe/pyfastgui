"""M7 demo: a virtualized `ListView` of 1,000,000 rows. Only the rows in view are drawn, so
scrolling, dragging the scrollbar and filtering stay fast. Click or use the arrow keys /
PageUp / PageDown / Home / End to select; double-click or Enter to activate. Type in the filter
box to narrow the list.
"""

import fastgui as fg

# Plain digits (no thousands separators) so filtering for e.g. "4242" finds row 4242.
ROWS = [f"Row {i:>7} — item {i % 97:02d}" for i in range(1_000_000)]


def main() -> None:
    window = fg.Window(title="fastgui — M7 virtualized list", width=520, height=520)
    status = fg.Label(f"{len(ROWS):,} rows", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))
    shown = ROWS

    def on_select(index: int) -> None:
        status.set_text(f"Selected {shown[index]!r}")

    def on_activate(index: int) -> None:
        status.set_text(f"Activated {shown[index]!r}")

    rows = fg.ListView(ROWS, on_select=on_select, on_activate=on_activate)

    def on_filter(text: str) -> None:
        nonlocal shown
        shown = [row for row in ROWS if text in row] if text else ROWS
        rows.set_items(shown)
        status.set_text(f"{len(shown):,} rows match {text!r}" if text else f"{len(ROWS):,} rows")

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.TextInput(placeholder="Filter (e.g. 42)", on_change=on_filter),
                rows,
                fg.Box(
                    direction="row",
                    gap=8.0,
                    children=[
                        fg.Button("Jump to middle", on_click=lambda: rows.select(len(shown) // 2)),
                        fg.Button("Last", on_click=lambda: rows.select(len(shown) - 1)),
                    ],
                ),
                status,
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
