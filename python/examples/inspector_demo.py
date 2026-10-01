"""M7 7D demo: `PropertyInspector` (label + 7B editors) driven by a `TreeView` selection.

Each node has its own properties. Selecting a node loads them into the editors; editing a
field updates only the selected node, and editing Name renames it in the tree.
"""

import fastgui as fg

DEFAULTS = {"gain": 1.0, "enabled": True, "mode": 0}


def main() -> None:
    window = fg.Window(title="fastgui — M7 property inspector", width=640, height=420)
    # Per-node properties, keyed by tree path; the node's name lives in the tree itself.
    props: dict[tuple[int, ...], dict] = {}

    def selected_props() -> dict | None:
        path = tree.selected
        if path is None:
            return None
        return props.setdefault(tuple(path), dict(DEFAULTS))

    def edit(key: str, value) -> None:
        node = selected_props()
        # Loading a node into the editors echoes back through on_change; ignore no-op edits.
        if node is not None and node[key] != value:
            node[key] = value
            status.set_text(f"{tree.label(tree.selected)}: {key} = {value!r}")

    def on_rename(text: str) -> None:
        if tree.selected is not None:
            tree.set_label(tree.selected, text)
            status.set_text(f"Renamed to {text!r}")

    name = fg.TextInput("Camera A", on_change=on_rename)
    gain = fg.SpinBox(value=1.0, min=0.0, max=10.0, step=0.1, decimals=1, on_change=lambda v: edit("gain", round(v, 1)))
    enabled = fg.Toggle(checked=True, on_change=lambda c: edit("enabled", c))
    mode = fg.ComboBox(["Raw", "Calibrated", "Preview"], selected=0, on_change=lambda i: edit("mode", i))
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
        label = tree.label(path)
        node = selected_props()
        name.set_text(label)
        gain.set_value(node["gain"])
        enabled.set_checked(node["enabled"])
        mode.select(node["mode"])
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
    tree.select((0, 0))  # matches the editors' initial values

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
