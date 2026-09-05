"""Headless tests for DockArea tree surgery — no window, no GPU."""

import unittest

from fastgui import DockArea, Label, Panel, Tabs
from fastgui import _ROOT_REGION_ID, _extract, _find_leaf, _replace


def _panel(title: str) -> Panel:
    return Panel(title=title, content=Label(title))


class DockAreaTreeTests(unittest.TestCase):
    def test_extract_collapses_parent_split(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        remainder, extracted = _extract(dock._root, b.id)
        self.assertIsNotNone(extracted)
        self.assertEqual(extracted["widget"].id, b.id)
        self.assertEqual(remainder["kind"], "leaf")
        self.assertEqual(remainder["widget"].id, a.id)

    def test_find_and_replace_leaf(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        found = _find_leaf(dock._root, a.id)
        self.assertIsNotNone(found)
        replacement = {"kind": "leaf", "widget": _panel("C")}
        new_root = _replace(dock._root, a.id, replacement)
        self.assertIsNone(_find_leaf(new_root, a.id))
        self.assertIsNotNone(_find_leaf(new_root, replacement["widget"].id))

    def test_rearrange_edge_split(self):
        a, b, c = _panel("A"), _panel("B"), _panel("C")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock.add_panel(c, region="bottom", size=0.2)
        dock._on_rearrange(c.id, a.id, "left")
        self.assertIsNotNone(_find_leaf(dock._root, c.id))
        self.assertIsNotNone(_find_leaf(dock._root, a.id))
        self.assertIsNotNone(_find_leaf(dock._root, b.id))

    def test_center_drop_makes_tabs(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock._on_rearrange(b.id, a.id, "center")
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertIsInstance(dock._root["widget"], Tabs)
        ids = [p.id for p in dock._root["widget"].panels]
        self.assertEqual(set(ids), {a.id, b.id})

    def test_ungroup_tab_collapses_to_panel(self):
        a, b, c = _panel("A"), _panel("B"), _panel("C")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock.add_panel(c, region="bottom", size=0.2)
        dock._on_rearrange(b.id, a.id, "center")
        dock._on_rearrange(b.id, c.id, "right")
        # A should be a plain panel again; B next to C.
        self.assertFalse(isinstance(_find_leaf(dock._root, a.id)["widget"], Tabs))
        self.assertIsNotNone(_find_leaf(dock._root, b.id))
        self.assertIsNotNone(_find_leaf(dock._root, c.id))

    def test_root_edge_wraps_whole_tree(self):
        a, b, c = _panel("A"), _panel("B"), _panel("C")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock.add_panel(c, region="bottom", size=0.2)
        dock._on_rearrange(c.id, _ROOT_REGION_ID, "top")
        self.assertEqual(dock._root["kind"], "split")
        self.assertEqual(dock._root["direction"], "column")
        self.assertEqual(dock._root["first"]["widget"].id, c.id)
        # The other side still contains both remaining panels.
        self.assertIsNotNone(_find_leaf(dock._root["second"], a.id))
        self.assertIsNotNone(_find_leaf(dock._root["second"], b.id))

    def test_drop_on_self_and_unknown_are_noops(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        before = dock._root
        dock._on_rearrange(a.id, a.id, "left")
        self.assertIs(dock._root, before)
        dock._on_rearrange(999999, b.id, "left")
        self.assertIs(dock._root, before)

    def test_rebuilds_widget_tree_without_raising(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock._on_rearrange(b.id, a.id, "center")
        widget = dock._fastgui_widget
        self.assertIsNotNone(widget)


if __name__ == "__main__":
    unittest.main()
