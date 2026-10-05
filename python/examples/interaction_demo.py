"""M7 7F demo: hover and disabled states, widget drag-and-drop, and OS file drops.

- Hover any button, checkbox or slider: the theme's `hover` state layer. The toggle at the top
  disables the whole control box (`Box.set_enabled`): dimmed, no clicks, no focus, no Tab.
- Drag list rows to reorder them, or onto the drop zone.
- Drag tree nodes before / onto / after other nodes to move them.
- Drop files from Explorer / Finder onto the image.
- File > Save is a disabled menu item.
"""

import fastgui as fg

FRUITS = ["Apple", "Banana", "Cherry", "Damson", "Elderberry", "Fig", "Grape"]
TREE = [
    ["Sensors", [["Camera", []], ["IMU", [["accel", []], ["gyro", []]]]]],
    ["Logs", []],
    ["Exports", []],
]


def to_nodes(items):
    return [fg.TreeNode(label, to_nodes(children)) for label, children in items]


def take(items, path):
    """Remove and return the node at `path` from the nested `[label, children]` lists."""
    *parents, last = path
    for index in parents:
        items = items[index][1]
    return items.pop(last)


def siblings_and_index(items, path):
    *parents, last = path
    for index in parents:
        items = items[index][1]
    return items, last


def main() -> None:
    window = fg.Window(title="fastgui — 7F hover, disabled, drag-and-drop", width=920, height=620)
    status = fg.Label("Drag a row or a node; drop files on the image.", font_size="small")

    # --- enabled / disabled -----------------------------------------------------------------
    controls = fg.Box(
        direction="column",
        gap="small",
        padding="medium",
        background=fg.get_theme().surface,
        children=[
            fg.Button("Button", on_click=lambda: status.set_text("Button clicked")),
            fg.Checkbox("Checkbox"),
            fg.Slider(value=0.4),
            fg.TextInput(placeholder="Text input"),
            fg.ComboBox(["One", "Two", "Three"], placeholder="Combo box"),
            fg.Button("Always disabled", enabled=False),
        ],
    )

    def on_toggle(checked: bool) -> None:
        controls.set_enabled(checked)
        status.set_text("Controls enabled" if checked else "Controls disabled")

    enable_row = fg.Box(
        direction="row",
        gap="small",
        children=[fg.Toggle(checked=True, on_change=on_toggle), fg.Label("Enable controls")],
    )

    # --- list reorder + drop zone ------------------------------------------------------------
    fruits = list(FRUITS)
    fruit_list = fg.ListView(fruits, height=200, flex_grow=0)
    fruit_list.set_drag_source("fruit", data=lambda index: fruits[index])

    def reorder(tag: str, data: bytes, index: int) -> None:
        name = data.decode()
        source = fruits.index(name)
        fruits.pop(source)
        fruits.insert(index - (1 if source < index else 0), name)
        fruit_list.set_items(fruits)
        status.set_text(f"Moved {name} to row {fruits.index(name)}")

    fruit_list.set_drop_target("fruit", reorder)

    zone_label = fg.Label("Drop a fruit here", font_size="small")
    zone = fg.Box(
        direction="column",
        padding="medium",
        height=60,
        background=fg.get_theme().surface_alt,
        children=[zone_label],
    )
    zone.set_drop_target("fruit", lambda tag, data, x, y: zone_label.set_text(f"Dropped {data.decode()}"))

    # --- tree reparent -----------------------------------------------------------------------
    tree_items = [list(node) for node in TREE]
    tree = fg.TreeView(to_nodes(tree_items), height=200, flex_grow=0)
    tree.set_drag_source("node")  # default payload: the node's path, "0/1"

    def move(tag: str, data: bytes, path: list[int], place: str) -> None:
        source = [int(part) for part in data.decode().split("/")]
        if path[: len(source)] == source:
            status.set_text("Can't move a node into itself")
            return
        node = take(tree_items, source)
        # Removing the source shifts later siblings (and their subtrees) up by one.
        if len(source) <= len(path) and path[: len(source) - 1] == source[:-1] and path[len(source) - 1] > source[-1]:
            path = path[: len(source) - 1] + [path[len(source) - 1] - 1] + path[len(source):]
        if place == "inside":
            siblings, index = siblings_and_index(tree_items, path)
            siblings[index][1].append(node)
        else:
            siblings, index = siblings_and_index(tree_items, path)
            siblings.insert(index + (1 if place == "after" else 0), node)
        tree.set_nodes(to_nodes(tree_items))
        status.set_text(f"Moved {node[0]} {place} {'/'.join(map(str, path))}")

    tree.set_drop_target("node", move)

    # --- OS file drop ------------------------------------------------------------------------
    image = fg.Image(height=120, flex_grow=0, fit="contain")
    files_label = fg.Label("Drop files on the image above", font_size="small")

    def on_files(paths: list[str], x: float, y: float) -> None:
        names = ", ".join(p.replace("\\", "/").rsplit("/", 1)[-1] for p in paths)
        files_label.set_text(f"{len(paths)} file(s) at ({x:.0f}, {y:.0f}): {names}")

    image.set_file_drop(on_files)
    try:
        import numpy as np

        checker = (np.indices((60, 120)).sum(axis=0) // 10 % 2).astype(np.uint8)
        pixels = np.stack([checker * 120 + 60] * 3 + [np.full_like(checker, 255)], axis=-1).astype(np.uint8)
        image.set_image(pixels)
    except ImportError:
        pass

    menu_bar = fg.MenuBar(
        [
            (
                "File",
                fg.Menu(
                    [
                        fg.MenuItem("Open", on_click=lambda: status.set_text("Open")),
                        fg.MenuItem("Save", shortcut="Ctrl+S", enabled=False),
                        fg.MenuSeparator(),
                        fg.MenuItem("Export", on_click=lambda: status.set_text("Export")),
                    ]
                ),
            )
        ]
    )

    column = lambda title, *children: fg.Box(  # noqa: E731
        direction="column", gap="small", flex_grow=1.0, children=[fg.Label(title), *children]
    )
    window.set_content(
        fg.Box(
            direction="column",
            children=[
                menu_bar,
                fg.Box(
                    direction="row",
                    gap="large",
                    padding="large",
                    flex_grow=1.0,
                    children=[
                        column("Hover / disabled", enable_row, controls),
                        column("Reorder (drag rows)", fruit_list, zone),
                        column("Move nodes", tree, image, files_label),
                    ],
                ),
                fg.Box(padding="medium", children=[status]),
            ],
        )
    )
    window.run()


if __name__ == "__main__":
    main()
