"""Headless tests for DockArea tree surgery — no window, no GPU."""

import unittest

from fastgui import DockArea, Label, Panel, Tabs, Window
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

    def test_center_drop_grows_existing_tabs(self):
        a, b, c = _panel("A"), _panel("B"), _panel("C")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        dock.add_panel(c, region="bottom", size=0.2)
        dock._on_rearrange(b.id, a.id, "center")
        tabs_id = dock._root["first"]["widget"].id if dock._root["kind"] == "split" else dock._root["widget"].id
        # After merging B onto A, C is still a sibling split; drop C onto the Tabs center.
        dock._on_rearrange(c.id, tabs_id, "center")
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertIsInstance(dock._root["widget"], Tabs)
        ids = [p.id for p in dock._root["widget"].panels]
        self.assertEqual(set(ids), {a.id, b.id, c.id})
        self.assertEqual(dock._root["widget"].panels[dock._root["widget"].active].id, c.id)

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

    def test_floating_redock_center_makes_tabs(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        self.assertIsNotNone(window._peek_floating_panel(b.id))
        dock._on_rearrange(b.id, a.id, "center")
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertIsInstance(dock._root["widget"], Tabs)
        ids = [p.id for p in dock._root["widget"].panels]
        self.assertEqual(set(ids), {a.id, b.id})
        self.assertIsNone(window._peek_floating_panel(b.id))
        # Re-describe as docked content must not raise, and the floater must not come back.
        widget = dock._fastgui_widget
        self.assertIsNotNone(widget)

    def test_floating_redock_edge_splits(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        dock._on_rearrange(b.id, a.id, "right")
        self.assertEqual(dock._root["kind"], "split")
        self.assertEqual(dock._root["direction"], "row")
        self.assertEqual(dock._root["first"]["widget"].id, a.id)
        self.assertEqual(dock._root["second"]["widget"].id, b.id)
        self.assertIsNone(window._peek_floating_panel(b.id))

    def test_floating_redock_root_edge_wraps_tree(self):
        a, b, c = _panel("A"), _panel("B"), _panel("C")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(c, 10.0, 10.0, 200.0, 100.0)
        dock._on_rearrange(c.id, _ROOT_REGION_ID, "top")
        self.assertEqual(dock._root["kind"], "split")
        self.assertEqual(dock._root["direction"], "column")
        self.assertEqual(dock._root["first"]["widget"].id, c.id)
        self.assertIsNotNone(_find_leaf(dock._root["second"], a.id))
        self.assertIsNotNone(_find_leaf(dock._root["second"], b.id))
        self.assertIsNone(window._peek_floating_panel(c.id))

    def test_floating_added_before_set_content_can_redock(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        window.set_content(dock)
        dock._on_rearrange(b.id, a.id, "right")
        self.assertEqual(dock._root["second"]["widget"].id, b.id)
        self.assertIsNone(window._peek_floating_panel(b.id))

    def test_floating_survives_switch_to_non_dock_content(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        window.set_content(Label("plain"))
        window.set_content(dock)
        self.assertIsNotNone(window._peek_floating_panel(b.id))

    def test_floating_unknown_target_leaves_floater(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        dock._on_rearrange(b.id, 999999, "left")
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertEqual(dock._root["widget"].id, a.id)
        self.assertIsNotNone(window._peek_floating_panel(b.id))

    def test_close_docked_panel(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        window = Window()
        window.set_content(dock)
        dock._on_close(b.id)
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertEqual(dock._root["widget"].id, a.id)

    def test_close_floating_panel(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        window.add_floating_panel(b, 10.0, 10.0, 200.0, 100.0)
        dock._on_close(b.id)
        self.assertIsNone(window._peek_floating_panel(b.id))
        self.assertEqual(dock._root["widget"].id, a.id)

    def test_close_last_docked_leaves_empty(self):
        a = _panel("A")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        dock._on_close(a.id)
        self.assertIsNone(dock._root)
        self.assertIsNotNone(dock._fastgui_widget)

    def test_undock_to_floating(self):
        a, b = _panel("A"), _panel("B")
        dock = DockArea()
        dock.add_panel(a, region="center")
        dock.add_panel(b, region="right", size=0.3)
        window = Window()
        window.set_content(dock)
        dock._on_rearrange(b.id, 0, "float", 40.0, 50.0, 220.0, 160.0)
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertEqual(dock._root["widget"].id, a.id)
        self.assertIsNotNone(window._peek_floating_panel(b.id))
        self.assertIsNone(_find_leaf(dock._root, b.id))

    def test_undock_last_panel_leaves_empty_dock(self):
        a = _panel("A")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        dock._on_rearrange(a.id, 0, "float", 10.0, 10.0, 200.0, 120.0)
        self.assertIsNone(dock._root)
        self.assertIsNotNone(window._peek_floating_panel(a.id))
        # Empty DockArea must still describe without raising so set_content can keep it.
        self.assertIsNotNone(dock._fastgui_widget)

    def test_redock_into_empty_dock(self):
        a = _panel("A")
        dock = DockArea()
        dock.add_panel(a, region="center")
        window = Window()
        window.set_content(dock)
        dock._on_rearrange(a.id, 0, "float", 10.0, 10.0, 200.0, 120.0)
        dock._on_rearrange(a.id, 0, "center")
        self.assertEqual(dock._root["kind"], "leaf")
        self.assertEqual(dock._root["widget"].id, a.id)
        self.assertIsNone(window._peek_floating_panel(a.id))


if __name__ == "__main__":
    unittest.main()
