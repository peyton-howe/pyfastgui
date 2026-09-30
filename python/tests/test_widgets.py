"""Headless tests for M7 widgets' Python API — no window shown, no GPU."""

import unittest

import fastgui as fg


class TextInputTests(unittest.TestCase):
    def test_text_flattens_newlines_and_set_text_works_before_showing(self):
        field = fg.TextInput("one\ntwo", placeholder="name")
        self.assertEqual(field.text, "one two")
        field.set_text("replaced\r\nagain")
        self.assertEqual(field.text, "replaced again")

    def test_rejects_bad_font_size(self):
        with self.assertRaises(ValueError):
            fg.TextInput(font_size="enormous")


class ListViewTests(unittest.TestCase):
    def test_len_set_items_and_select_before_showing(self):
        rows = fg.ListView([f"row {i}" for i in range(10)])
        self.assertEqual((len(rows), rows.selected), (10, None))
        rows.select(3)
        self.assertEqual(rows.selected, 3)
        rows.select(99)
        self.assertEqual(rows.selected, 9, "clamped to the last row")
        rows.select(None)
        self.assertIsNone(rows.selected)
        rows.set_items(["a", "b"])
        self.assertEqual((len(rows), rows.selected), (2, None))

    def test_empty_list_and_bad_row_height(self):
        empty = fg.ListView([])
        empty.select(0)
        self.assertIsNone(empty.selected, "nothing to select")
        with self.assertRaises(ValueError):
            fg.ListView(["a"], row_height=0)

    def test_a_million_rows(self):
        self.assertEqual(len(fg.ListView([str(i) for i in range(1_000_000)])), 1_000_000)


class TableTests(unittest.TestCase):
    def test_dict_columns_select_and_set_columns(self):
        table = fg.Table(columns={"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self.assertEqual((len(table), table.selected), (3, None))
        table.select(1)
        self.assertEqual(table.selected, 1)
        table.select(99)
        self.assertEqual(table.selected, 2)
        table.select(None)
        self.assertIsNone(table.selected)
        table.set_columns([("n", [10, 20])])
        self.assertEqual((len(table), table.selected), (2, None))

    def test_rejects_mismatched_lengths_and_bad_sizes(self):
        with self.assertRaises(ValueError):
            fg.Table(columns={"a": [1, 2], "b": [1]})
        with self.assertRaises(ValueError):
            fg.Table(columns={"a": [1]}, row_height=0)
        with self.assertRaises(ValueError):
            fg.Table(columns={"a": [1]}, column_widths=[10.0, 20.0])

    def test_numpy_columns(self):
        import numpy as np

        n = 10_000
        table = fg.Table(
            columns={"i": np.arange(n), "x": np.linspace(0.0, 1.0, n)},
            column_widths=[60.0, 80.0],
        )
        self.assertEqual(len(table), n)


class TreeViewTests(unittest.TestCase):
    def test_select_and_expand_before_showing(self):
        tree = fg.TreeView(
            [
                fg.TreeNode("root", children=[fg.TreeNode("child"), fg.TreeNode("other")]),
                fg.TreeNode("leaf"),
            ]
        )
        self.assertIsNone(tree.selected)
        tree.select((0, 1))
        self.assertEqual(tree.selected, [0, 1])
        tree.set_expanded((0,), True)
        tree.select(None)
        self.assertIsNone(tree.selected)
        tree.set_nodes([fg.TreeNode("only")])
        self.assertIsNone(tree.selected)

    def test_rejects_bad_row_height(self):
        with self.assertRaises(ValueError):
            fg.TreeView([fg.TreeNode("x")], row_height=0)


class PropertyInspectorTests(unittest.TestCase):
    def test_builds_composite(self):
        inspector = fg.PropertyInspector(
            [
                ("Name", fg.TextInput("a")),
                ("On", fg.Toggle(checked=True)),
            ]
        )
        widget = inspector._fastgui_widget
        self.assertIsNotNone(widget)


class PopupTests(unittest.TestCase):
    def test_closed_until_shown_and_close_is_a_no_op(self):
        popup = fg.Popup(fg.Label("hi"))
        self.assertFalse(popup.is_open)
        popup.close()
        self.assertFalse(popup.is_open)

    def test_show_needs_a_shown_anchor_and_a_valid_side(self):
        popup = fg.Popup(fg.Label("hi"))
        with self.assertRaises(RuntimeError):
            popup.show(fg.Button("not in a window"))
        with self.assertRaises(ValueError):
            popup.show(fg.Button("x"), side="diagonal")

    def test_is_not_layout_content(self):
        window = fg.Window(title="test", width=100, height=100)
        with self.assertRaises(TypeError):
            window.set_content(fg.Popup(fg.Label("hi")))
        with self.assertRaises(ValueError):
            window.show_popup(fg.Popup(fg.Label("hi")), x=10.0)


class ScrollAreaTests(unittest.TestCase):
    def test_scroll_to_works_before_showing(self):
        area = fg.ScrollArea(fg.Box(children=[fg.Label(str(i)) for i in range(50)]))
        area.scroll_to(0, 120)
        area.scroll_to(-5, -5)


class CheckboxTests(unittest.TestCase):
    def test_checked_and_set_checked_before_showing(self):
        box = fg.Checkbox("Accept", checked=False)
        self.assertFalse(box.checked)
        box.set_checked(True)
        self.assertTrue(box.checked)
        box.set_checked(False)
        self.assertFalse(box.checked)


class RadioTests(unittest.TestCase):
    def test_group_exclusivity_and_set_selected_before_showing(self):
        a = fg.Radio("A", group=1, selected=True)
        b = fg.Radio("B", group=1, selected=False)
        self.assertEqual(a.group, b.group)
        self.assertTrue(a.selected)
        self.assertFalse(b.selected)
        b.set_selected(True)
        self.assertFalse(a.selected)
        self.assertTrue(b.selected)
        b.set_selected(False)
        self.assertFalse(b.selected)

    def test_auto_group_uses_high_bit_and_rejects_colliding_user_ids(self):
        auto = fg.Radio("auto")
        self.assertGreaterEqual(auto.group, 1 << 63)
        with self.assertRaises(ValueError):
            fg.Radio("bad", group=1 << 63)


class ToggleTests(unittest.TestCase):
    def test_checked_and_set_checked_before_showing(self):
        toggle = fg.Toggle(checked=True)
        self.assertTrue(toggle.checked)
        toggle.set_checked(False)
        self.assertFalse(toggle.checked)


class SpinBoxTests(unittest.TestCase):
    def test_value_clamps_rounds_and_rejects_bad_inputs(self):
        spin = fg.SpinBox(value=3.6, min=0, max=10, step=1, decimals=0)
        self.assertEqual(spin.value, 4.0)
        spin.set_value(9.4)
        self.assertEqual(spin.value, 9.0)
        spin.set_value(99)
        self.assertEqual(spin.value, 10.0)
        with self.assertRaises(ValueError):
            fg.SpinBox(value=float("nan"))
        with self.assertRaises(ValueError):
            fg.SpinBox(step=0)
        with self.assertRaises(ValueError):
            spin.set_value(float("nan"))


class NumericScrubTests(unittest.TestCase):
    def test_value_rounds_and_rejects_bad_speed(self):
        scrub = fg.NumericScrub(value=1.24, min=0, max=10, speed=0.25, decimals=1)
        self.assertAlmostEqual(scrub.value, 1.2)
        scrub.set_value(2.26)
        self.assertAlmostEqual(scrub.value, 2.3)
        with self.assertRaises(ValueError):
            fg.NumericScrub(speed=0)
        with self.assertRaises(ValueError):
            fg.NumericScrub(speed=float("nan"))
        with self.assertRaises(ValueError):
            scrub.set_value(float("nan"))


class ProgressBarTests(unittest.TestCase):
    def test_value_and_set_value_before_showing(self):
        bar = fg.ProgressBar(value=0.25, min=0, max=1)
        self.assertAlmostEqual(bar.value, 0.25)
        bar.set_value(0.8)
        self.assertAlmostEqual(bar.value, 0.8)
        bar.set_value(2.0)
        self.assertAlmostEqual(bar.value, 1.0)


class ComboBoxTests(unittest.TestCase):
    def test_items_select_and_set_items_before_showing(self):
        combo = fg.ComboBox(["a", "b", "c"], selected=1)
        self.assertEqual(combo.selected, 1)
        combo.select(2)
        self.assertEqual(combo.selected, 2)
        combo.select(None)
        self.assertIsNone(combo.selected)
        combo.set_items(["x", "y"])
        self.assertIsNone(combo.selected)
        combo.select(0)
        self.assertEqual(combo.selected, 0)


class ImageTests(unittest.TestCase):
    def test_constructs_and_rejects_empty_or_oversized_image(self):
        import numpy as np

        image = fg.Image(width=32, height=24)
        with self.assertRaises(ValueError):
            image.set_image(np.zeros((0, 8, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            image.set_image(np.zeros((1, 16385, 4), dtype=np.uint8))


class TextAreaTests(unittest.TestCase):
    def test_text_and_set_text_before_showing(self):
        area = fg.TextArea("hello\nworld", placeholder="notes")
        self.assertEqual(area.text, "hello\nworld")
        area.set_text("replaced\r\nagain")
        self.assertEqual(area.text, "replaced\nagain")


class GridTests(unittest.TestCase):
    def test_constructs_and_rejects_bad_columns(self):
        grid = fg.Grid([fg.Label("a"), fg.Label("b")], columns=2)
        self.assertIsNotNone(grid)
        with self.assertRaises(ValueError):
            fg.Grid([fg.Label("a")], columns=0)


class AppChromeTests(unittest.TestCase):
    def test_stacked_widget_index_and_empty_rejected(self):
        stack = fg.StackedWidget([fg.Label("a"), fg.Label("b"), fg.Label("c")], index=1)
        self.assertEqual(stack.index, 1)
        stack.set_index(2)
        self.assertEqual(stack.index, 2)
        stack.set_index(99)
        self.assertEqual(stack.index, 2)
        with self.assertRaises(ValueError):
            fg.StackedWidget([])

    def test_collapsible_tracks_expanded(self):
        section = fg.CollapsibleSection("More", fg.Label("body"), expanded=False)
        self.assertFalse(section.expanded)
        section.set_expanded(True)
        self.assertTrue(section.expanded)

    def test_dialog_and_status_bar_api(self):
        bar = fg.StatusBar("hi")
        bar.set_text("there")
        dialog = fg.Dialog("Title", fg.Label("body"), buttons=[("OK", None)], modal=False)
        self.assertFalse(dialog.is_open)
        dialog.close()
        self.assertFalse(dialog.is_open)

    def test_box_set_display_before_attach(self):
        box = fg.Box(children=[fg.Label("x")], visible=False)
        box.set_display(True)


class MenuTests(unittest.TestCase):
    def test_item_labels_for_check_radio_icon_submenu(self):
        nested = fg.Menu([fg.MenuItem("Child")])
        menu = fg.Menu(
            [
                fg.MenuItem("Save", checked=True),
                fg.MenuItem("Plain", checked=False),
                fg.MenuItem("Dark", radio_group="t", checked=True),
                fg.MenuItem("Light", radio_group="t", checked=False),
                fg.MenuItem("Doc", icon="📄"),
                fg.MenuItem("Recent", submenu=nested),
            ]
        )
        self.assertEqual(menu._item_label(menu.items[0]), "✓  Save")
        self.assertEqual(menu._item_label(menu.items[1]), "   Plain")
        self.assertEqual(menu._item_label(menu.items[2]), "●  Dark")
        self.assertEqual(menu._item_label(menu.items[3]), "○  Light")
        self.assertEqual(menu._item_label(menu.items[4]), "📄  Doc")
        self.assertEqual(menu._item_label(menu.items[5]), "Recent    ▶")

    def test_close_runs_on_dismiss_once(self):
        hits = []
        menu = fg.Menu([fg.MenuItem("Open", on_click=lambda: hits.append("click"))])
        menu.as_popup(on_dismiss=lambda: hits.append("dismiss"))
        menu.close()
        self.assertEqual(hits, ["dismiss"])
        menu.close()
        self.assertEqual(hits, ["dismiss"], "second close must not re-fire dismiss")

    def test_rejects_shortcut_with_submenu(self):
        nested = fg.Menu([fg.MenuItem("Child")])
        with self.assertRaises(ValueError):
            fg.MenuItem("Recent", shortcut="Cmd+R", submenu=nested)

    def test_menubar_collects_nested_accelerators(self):
        nested = fg.Menu([fg.MenuItem("Deep", shortcut="Cmd+D", on_click=lambda: None)])
        menu = fg.Menu(
            [
                fg.MenuItem("Top", shortcut="Cmd+T", on_click=lambda: None),
                fg.MenuItem("More", submenu=nested),
            ]
        )
        accels = fg._menu_accelerators(menu)
        self.assertEqual([s for s, _ in accels], ["Cmd+T", "Cmd+D"])



class MenuChainTests(unittest.TestCase):
    """Menu composite logic, with `Button` / `Popup` captured instead of shown."""

    def setUp(self):
        self._saved = (fg.Button, fg.Popup)
        self.buttons = {}
        self.hovers = {}
        self.popups = []
        test = self

        class FakeButton:
            def __init__(self, text, on_click=None, on_hover=None, **kwargs):
                label = text.split("    ")[0].strip()
                test.buttons[label] = on_click
                test.hovers[label] = on_hover

            def set_background(self, color=None):
                pass

        class FakePopup:
            def __init__(self, content, **kwargs):
                self.kwargs = kwargs
                self.is_open = False
                test.popups.append(self)

            def show(self, anchor, side="below"):
                self.is_open = True

            def close(self):
                self.is_open = False

        fg.Button, fg.Popup = FakeButton, FakePopup

    def tearDown(self):
        fg.Button, fg.Popup = self._saved

    def test_picking_in_a_submenu_closes_the_whole_chain(self):
        picked, dismissed = [], []
        sub = fg.Menu([fg.MenuItem("Doc", on_click=lambda: picked.append("doc"))])
        root = fg.Menu([fg.MenuItem("Recent", submenu=sub)])
        root.show(object(), on_dismiss=lambda: dismissed.append("root"))
        self.buttons["Recent"]()  # open the submenu
        self.assertTrue(sub.is_open and root.is_open)
        self.buttons["Doc"]()  # pick in it
        self.assertEqual(picked, ["doc"])
        self.assertFalse(sub.is_open, "the submenu closes")
        self.assertFalse(root.is_open, "and so does its parent")
        self.assertEqual(dismissed, ["root"], "the top menu's on_dismiss runs (menu bar highlight)")

    def test_menu_bar_menus_let_the_dismissing_click_through(self):
        menu = fg.Menu([fg.MenuItem("Open")])
        menu.show(object(), click_through=True)
        self.assertTrue(self.popups[-1].kwargs["click_through"])
        menu.show(object())
        self.assertFalse(self.popups[-1].kwargs["click_through"], "other menus keep the default")


class BoxTests(unittest.TestCase):
    def test_set_background_before_showing(self):
        box = fg.Box(children=[])
        box.set_background((1.0, 0.0, 0.0, 1.0))
        box.set_background(None)


class SubmenuHoverTests(MenuChainTests):
    """Submenus open on hover (the app runs `on_hover` after a short rest), no click needed."""

    def test_hovering_a_submenu_row_opens_it_and_another_row_closes_it(self):
        sub = fg.Menu([fg.MenuItem("Doc")])
        root = fg.Menu([fg.MenuItem("Recent", submenu=sub), fg.MenuItem("Save")])
        root.show(object())
        self.hovers["Recent"]()
        self.assertTrue(sub.is_open, "resting on the row opens its submenu")
        self.assertFalse(self.popups[-1].kwargs["closes_on_anchor_click"], "clicking the row keeps it open")
        self.buttons["Recent"]()
        self.hovers["Recent"]()
        self.assertTrue(sub.is_open, "clicking / hovering it again doesn't toggle it shut")
        self.hovers["Save"]()
        self.assertFalse(sub.is_open, "resting on another row closes it")
        self.assertTrue(root.is_open)

    def test_disabled_submenu_rows_do_not_open_on_hover(self):
        sub = fg.Menu([fg.MenuItem("Doc")])
        root = fg.Menu([fg.MenuItem("Recent", submenu=sub, enabled=False)])
        root.show(object())
        self.hovers["Recent"]()
        self.assertFalse(sub.is_open)


class ShortcutAndTextPersistenceTests(unittest.TestCase):
    def test_menu_item_rejects_shortcuts_that_can_never_fire(self):
        for bad in ("Ctrl++", "Ctrl+Bogus", "Alt+F4", ""):
            if bad:
                with self.assertRaises(ValueError, msg=bad):
                    fg.MenuItem("x", shortcut=bad)
        for good in ("Ctrl+S", "Cmd+Shift+N", "F5", "Del", "Ctrl+Enter", "Alt+Left"):
            fg.MenuItem("x", shortcut=good)

    def test_label_and_button_set_text_before_showing(self):
        label = fg.Label("a")
        label.set_text("b")
        button = fg.Button("a")
        button.set_text("b")


class ShortcutLabelTests(unittest.TestCase):
    def test_primary_modifier_reads_as_this_platforms_key(self):
        from unittest import mock

        from fastgui import _shortcut_label

        with mock.patch("sys.platform", "linux"):
            self.assertEqual(_shortcut_label("Cmd+Shift+N"), "Ctrl+Shift+N")
            self.assertEqual(_shortcut_label("Ctrl+S"), "Ctrl+S")
        with mock.patch("sys.platform", "darwin"):
            self.assertEqual(_shortcut_label("Ctrl+S"), "Cmd+S")
            self.assertEqual(_shortcut_label("Alt+F4"), "Alt+F4")
        self.assertEqual(_shortcut_label("F5"), "F5")


if __name__ == "__main__":
    unittest.main()
