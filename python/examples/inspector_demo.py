"""M7 7D demo: `PropertyInspector` (label + 7B editors) driven by a `TreeView` selection."""

import fastgui as fg


def main() -> None:
    window = fg.Window(title="fastgui — M7 property inspector", width=640, height=420)
    name = fg.TextInput("Camera A", on_change=lambda t: status.set_text(f"name → {t!r}"))
    gain = fg.SpinBox(value=1.0, min=0.0, max=10.0, step=0.1, decimals=1)
    enabled = fg.Toggle(checked=True)
    mode = fg.ComboBox(["Raw", "Calibrated", "Preview"], selected=0)
    status = fg.Label("Edit properties on the right", font_size=14.0, color=(0.7, 0.75, 0.85, 1.0))

    inspector = fg.PropertyInspector(
        [
            ("Name", name),
            ("Gain", gain),
            ("Enabled", enabled),
            ("Mode", mode),
        ],
        label_width=90.0,
    )

    def on_select(path) -> None:
        labels = {
            (0,): "Sensors",
            (0, 0): "Camera A",
            (0, 1): "IMU",
            (1,): "Logs",
        }
        label = labels.get(tuple(path), f"node {tuple(path)}")
        name.set_text(label)
        status.set_text(f"Inspecting {label}")

    tree = fg.TreeView(
        [
            fg.TreeNode(
                "Sensors",
                children=[fg.TreeNode("Camera A"), fg.TreeNode("IMU")],
            ),
            fg.TreeNode("Logs"),
        ],
        on_select=on_select,
        width=220.0,
        flex_grow=0.0,
    )
    tree.set_expanded((0,), True)

    window.set_content(
        fg.Box(
            direction="column",
            gap=10.0,
            padding=16.0,
            background=(0.10, 0.11, 0.13, 1.0),
            children=[
                fg.Label("Tree + PropertyInspector", font_size="body"),
                fg.Box(
                    direction="row",
                    gap=12.0,
                    flex_grow=1.0,
                    children=[tree, inspector],
                ),
                status,
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
