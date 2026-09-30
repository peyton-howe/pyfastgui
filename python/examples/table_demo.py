"""M7 7D demo: a virtualized `Table` of 1,000,000 rows fed from numpy columns.

Only the rows in view are drawn, so scrolling and selection stay fast. Click or use the
arrow keys / PageUp / PageDown / Home / End to select; double-click or Enter to activate.
Wide fixed column widths enable horizontal scrolling.
"""

import numpy as np

import fastgui as fg

N = 1_000_000
IDX = np.arange(N, dtype=np.int64)
X = np.sin(IDX * 0.001).astype(np.float64)
Y = np.cos(IDX * 0.001).astype(np.float64)
LABELS = np.array([f"item-{i % 97:02d}" for i in range(N)], dtype=object)


def main() -> None:
    window = fg.Window(title="fastgui — M7 virtualized table", width=720, height=520)
    status = fg.Label(f"{N:,} rows × 4 columns", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    def on_select(index: int) -> None:
        status.set_text(f"Selected row {index}: x={X[index]:.4f}  y={Y[index]:.4f}  {LABELS[index]}")

    def on_activate(index: int) -> None:
        status.set_text(f"Activated row {index}")

    table = fg.Table(
        columns={"i": IDX, "x": X, "y": Y, "label": LABELS},
        column_widths=[80.0, 140.0, 140.0, 200.0],
        on_select=on_select,
        on_activate=on_activate,
    )

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Label("Virtualized Table (numpy columns)", font_size="body"),
                table,
                fg.Box(
                    direction="row",
                    gap=8.0,
                    children=[
                        fg.Button("Jump to middle", on_click=lambda: table.select(N // 2)),
                        fg.Button("Last", on_click=lambda: table.select(N - 1)),
                        fg.Button("Clear", on_click=lambda: table.select(None)),
                    ],
                ),
                status,
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
