"""Viewport.submit_metal / Image.submit_metal without opening a window.

The real retain, format, and shared-event checks live in fastgui-render-mtl. This only checks
that the Python entry point rejects a null pointer on macOS and refuses the call elsewhere.
"""

import sys
import unittest

import fastgui as fg


class SubmitMetalTests(unittest.TestCase):
    def test_null_texture_is_rejected(self):
        viewport = fg.Viewport()
        image = fg.Image()
        if sys.platform == "darwin":
            with self.assertRaises(ValueError):
                viewport.submit_metal(0)
            with self.assertRaises(ValueError):
                image.submit_metal(0)
        else:
            with self.assertRaises(RuntimeError):
                viewport.submit_metal(0)
            with self.assertRaises(RuntimeError):
                image.submit_metal(0)

    def test_object_without_mtl_ptr_is_rejected_on_macos(self):
        if sys.platform != "darwin":
            self.skipTest("Metal import is macOS-only")
        with self.assertRaises(ValueError):
            fg.Viewport().submit_metal(object())


class MtlpySubmitTests(unittest.TestCase):
    def test_mtlpy_texture_is_accepted(self):
        try:
            import mtlpy
        except ImportError:
            self.skipTest("mtlpy is not installed")
        import numpy as np

        image = np.zeros((4, 6, 4), np.uint8)
        image[..., 0] = 255
        image[..., 3] = 255
        device = mtlpy.Device()
        texture = device.texture(image, "rgba8Unorm")
        fg.Viewport().submit_metal(texture)
        fg.Image().submit_metal(texture.mtl_ptr)
        private = device.empty_texture((4, 6), "rgba8Unorm", readable=True, writable=False, private=True)
        private.upload(image)
        fg.Viewport().submit_metal(private)


class SubmitDmaBufTests(unittest.TestCase):
    def test_refused_off_linux_and_rejects_a_modifier(self):
        viewport = fg.Viewport()
        if sys.platform == "linux":
            with self.assertRaises(ValueError):
                viewport.submit_dmabuf(0, 4, 4, modifier=1)
            with self.assertRaises(ValueError):
                viewport.submit_dmabuf(-1, 4, 4)
        else:
            with self.assertRaises(RuntimeError):
                viewport.submit_dmabuf(0, 4, 4)
            with self.assertRaises(RuntimeError):
                fg.Image().submit_dmabuf(0, 4, 4)
