"""`enabled` and drag-and-drop on widgets (M7 7F.1 / 7F.2), headless.

The input behavior (presses skipped, drop targets resolved, files routed) is covered by
fastgui-core's Rust tests; these check the Python API: arguments, defaults, validation, and that
state set before a widget is shown is kept.
"""

import unittest

import fastgui as fg

ENABLED_WIDGETS = [
    lambda **kw: fg.Button("Go", **kw),
    lambda **kw: fg.Slider(**kw),
    lambda **kw: fg.TextInput(**kw),
    lambda **kw: fg.TextArea(**kw),
    lambda **kw: fg.ListView(["a", "b"], **kw),
    lambda **kw: fg.Table({"x": [1, 2]}, **kw),
    lambda **kw: fg.TreeView([fg.TreeNode("root")], **kw),
    lambda **kw: fg.Checkbox("Check", **kw),
    lambda **kw: fg.Radio("Pick", **kw),
    lambda **kw: fg.Toggle(**kw),
    lambda **kw: fg.SpinBox(**kw),
    lambda **kw: fg.NumericScrub(**kw),
    lambda **kw: fg.ComboBox(["one", "two"], **kw),
    lambda **kw: fg.Box([fg.Label("inside")], **kw),
]


class EnabledTests(unittest.TestCase):
    def test_every_control_and_box_takes_enabled(self):
        for make in ENABLED_WIDGETS:
            widget = make()
            with self.subTest(widget=type(widget).__name__):
                self.assertTrue(widget.enabled, "enabled by default")
                self.assertFalse(make(enabled=False).enabled)
                widget.set_enabled(False)  # before it's shown: kept for when it is
                self.assertFalse(widget.enabled)
                widget.set_enabled(True)
                self.assertTrue(widget.enabled)

    def test_disabled_menu_items_are_disabled_buttons(self):
        menu = fg.Menu([fg.MenuItem("Open", on_click=lambda: None), fg.MenuItem("Save", enabled=False)])
        popup = menu.as_popup()
        self.assertIsNotNone(popup)

    def test_theme_has_hover_and_disabled_tokens(self):
        for theme in (fg.Theme.dark(), fg.Theme.light()):
            with self.subTest(theme=theme):
                self.assertEqual(len(theme.hover), 4)
                self.assertLess(theme.hover[3], 1.0, "a translucent state layer")
                self.assertLess(theme.disabled[3], 1.0, "a translucent veil")
        custom = fg.Theme.dark().replace(hover=(1.0, 0.0, 0.0, 0.25), disabled=(0.0, 0.0, 0.0, 0.5))
        self.assertEqual(custom.hover, (1.0, 0.0, 0.0, 0.25))
        self.assertEqual(custom.disabled, (0.0, 0.0, 0.0, 0.5))


class DragAndDropTests(unittest.TestCase):
    def test_sources_and_targets_on_lists_trees_and_boxes(self):
        list_view = fg.ListView(["a", "b", "c"])
        tree = fg.TreeView([fg.TreeNode("root", [fg.TreeNode("leaf")])])
        box = fg.Box([fg.Label("drop here")])
        for widget in (list_view, tree, box):
            with self.subTest(widget=type(widget).__name__):
                widget.set_drag_source("thing")
                widget.set_drag_source("thing", data=lambda *args: b"payload")
                widget.set_drag_source(None)
                widget.set_drop_target("thing", lambda *args: None)
                widget.set_drop_target(["thing", "other"], lambda *args: None)
                widget.set_drop_target(None, lambda *args: None)  # any tag
                widget.set_drop_target("thing", None)  # stop accepting

    def test_accept_must_be_tags(self):
        with self.assertRaises(TypeError):
            fg.ListView([]).set_drop_target(42, lambda *args: None)
        with self.assertRaises(TypeError):
            fg.Box([]).set_drop_target([1, 2], lambda *args: None)

    def test_file_drops_on_viewports_and_images(self):
        for widget in (fg.Viewport(), fg.Image()):
            with self.subTest(widget=type(widget).__name__):
                widget.set_file_drop(lambda paths, x, y: None)
                widget.set_file_drop(None)

    def test_drag_state_survives_into_a_window(self):
        # Set before the window exists, applied when the content is attached (no run()).
        window = fg.Window(title="dnd", width=200, height=200)
        items = fg.ListView(["a", "b"])
        items.set_drag_source("row")
        items.set_drop_target("row", lambda tag, data, index: None)
        items.set_enabled(False)
        window.set_content(fg.Box([items]))
        self.assertFalse(items.enabled)


def write_png(path, width, height, rgb):
    """A tiny RGB PNG with the standard library only."""
    import struct
    import zlib

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(rows)))
        f.write(chunk(b"IEND", b""))


class ImageLoadTests(unittest.TestCase):
    def test_png_loads_and_bad_files_raise(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "red.png")
            write_png(png, 4, 3, (255, 0, 0))
            image = fg.Image()
            image.load(png)
            image.load(pathlib_path(png))
            not_image = os.path.join(tmp, "notes.txt")
            with open(not_image, "w") as f:
                f.write("hello")
            with self.assertRaises(ValueError):
                image.load(not_image)
            with self.assertRaises(OSError):
                image.load(os.path.join(tmp, "missing.png"))

    def test_jpeg_loads(self):
        import os

        sample = r"C:\Windows\Web\Wallpaper\Windows\img0.jpg"
        if not os.path.exists(sample):
            self.skipTest("no sample JPEG on this machine")
        fg.Image().load(sample)


def pathlib_path(path):
    import pathlib

    return pathlib.Path(path)


class TreeMoveTests(unittest.TestCase):
    def tree(self):
        return fg.TreeView([
            fg.TreeNode("Sensors", [fg.TreeNode("Camera"), fg.TreeNode("IMU", [fg.TreeNode("gyro")])]),
            fg.TreeNode("Logs"),
        ])

    def test_move_keeps_selection_on_its_node(self):
        tree = self.tree()
        tree.select([0, 1, 0])  # gyro
        tree.move_node([0, 1], [1], "inside")  # IMU into Logs
        self.assertEqual(tree.label([1, 0]), "IMU")
        self.assertEqual(tree.label([1, 0, 0]), "gyro")
        self.assertEqual(tree.selected, [1, 0, 0], "selection followed gyro")
        tree.move_node([1], [0], "before")
        self.assertEqual(tree.label([0]), "Logs")
        tree.move_node([1, 0], [0], "after")  # Camera after Logs, at the root
        self.assertEqual(tree.label([1]), "Camera")

    def test_bad_moves_raise(self):
        tree = self.tree()
        with self.assertRaises(ValueError):
            tree.move_node([0], [0, 1], "inside")  # into its own subtree
        with self.assertRaises(ValueError):
            tree.move_node([0], [0], "after")
        with self.assertRaises(ValueError):
            tree.move_node([5], [0], "after")
        with self.assertRaises(ValueError):
            tree.move_node([1], [0], "below")


if __name__ == "__main__":
    unittest.main()
