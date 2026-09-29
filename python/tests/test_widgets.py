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


if __name__ == "__main__":
    unittest.main()
