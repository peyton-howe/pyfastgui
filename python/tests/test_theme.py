"""Headless tests for fg.Theme — no window, no GPU."""

import unittest

import fastgui as fg


class ThemeTests(unittest.TestCase):
    def tearDown(self):
        fg.set_theme(fg.Theme.dark())

    def test_presets_differ_where_it_shows(self):
        dark, light = fg.Theme.dark(), fg.Theme.light()
        for token in ("background", "surface", "text", "button", "button_text"):
            self.assertNotEqual(getattr(dark, token), getattr(light, token), token)
        # Readable button labels on both: light text on dark buttons and vice versa.
        self.assertGreater(sum(dark.button_text[:3]), sum(dark.button[:3]))
        self.assertLess(sum(light.button_text[:3]), sum(light.button[:3]))

    def test_default_is_dark_and_keywords_override(self):
        self.assertEqual(fg.get_theme().background, fg.Theme.dark().background)
        theme = fg.Theme(button=(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(theme.button, (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(theme.accent, fg.Theme.dark().accent)

    def test_replace_copies_and_rejects_unknown_tokens(self):
        base = fg.Theme.light()
        orange = base.replace(accent=(1.0, 0.55, 0.1, 1.0), button=(0.9, 0.45, 0.05, 1.0))
        for got, want in zip(orange.button, (0.9, 0.45, 0.05, 1.0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(base.button, fg.Theme.light().button, "the original is unchanged")
        with self.assertRaises(TypeError):
            base.replace(buttn=(1.0, 1.0, 1.0, 1.0))

    def test_set_theme_is_global(self):
        fg.set_theme(fg.Theme.light())
        self.assertEqual(fg.get_theme().button, fg.Theme.light().button)

    def test_widgets_accept_theme_defaults_and_explicit_colors(self):
        fg.set_theme(fg.Theme.light())
        fg.Button("themed")
        fg.Button("explicit", background=(0.0, 0.5, 0.0, 1.0), text_color=(1.0, 1.0, 1.0, 1.0))
        fg.TextInput()
        fg.ListView(["a"])


if __name__ == "__main__":
    unittest.main()
