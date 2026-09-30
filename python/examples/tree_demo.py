"""M7 7D demo: a virtualized `TreeView` with expand/collapse and path selection.

Click the ▶/▼ gutter to expand; click a label to select. Arrow keys move among visible
rows; Left collapses or moves to the parent; Right expands or enters the first child.
"""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 tree view", width=480, height=420)
    status = fg.Label("Select a node", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    def on_select(path) -> None:
        status.set_text(f"Selected path {tuple(path)}")

    def on_activate(path) -> None:
        status.set_text(f"Activated path {tuple(path)}")

    tree = fg.TreeView(
        [
            fg.TreeNode(
                "Sensors",
                children=[
                    fg.TreeNode("Camera A"),
                    fg.TreeNode(
                        "IMU",
                        children=[fg.TreeNode("accel"), fg.TreeNode("gyro")],
                    ),
                    fg.TreeNode("GPS"),
                ],
            ),
            fg.TreeNode(
                "Logs",
                children=[
                    fg.TreeNode("session-001"),
                    fg.TreeNode("session-002"),
                ],
            ),
            fg.TreeNode("Config"),
        ],
        on_select=on_select,
        on_activate=on_activate,
    )
    tree.set_expanded((0,), True)

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Label("TreeView", font_size="body"),
                tree,
                fg.Box(
                    direction="row",
                    gap=8.0,
                    children=[
                        fg.Button("Select gyro", on_click=lambda: tree.select((0, 1, 1))),
                        fg.Button("Expand IMU", on_click=lambda: tree.set_expanded((0, 1), True)),
                        fg.Button("Collapse Sensors", on_click=lambda: tree.set_expanded((0,), False)),
                    ],
                ),
                status,
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
