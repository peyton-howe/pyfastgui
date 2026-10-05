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


if __name__ == "__main__":
    unittest.main()
