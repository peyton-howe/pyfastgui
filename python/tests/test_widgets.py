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


if __name__ == "__main__":
    unittest.main()
