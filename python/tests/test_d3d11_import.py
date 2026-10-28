"""Viewport.submit_d3d11 / Image.submit_d3d11 argument handling without opening a window.

The real import (a D3D11 producer, same-adapter Vulkan, pixel readback, fences, size and format
checks) is covered by the Rust tests in `fastgui-render-vk`'s `upload_bench::d3d11_tests`, and by
`examples/d3d11_viewport.py` in a real window. This only checks that the Python entry points reject
bad input on Windows and refuse the call elsewhere, before anything is queued.
"""

import sys
import unittest

import fastgui as fg


class SubmitD3d11Tests(unittest.TestCase):
    def test_bad_arguments(self):
        for widget in (fg.Viewport(), fg.Image()):
            if sys.platform == "win32":
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0, 4, 4)  # null handle
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(-1, 4, 4)  # current-process pseudo-handle
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0x7FFFFFF0, 0, 4)  # empty texture
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0x7FFFFFF0, 4, 4)  # not a handle in this process
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0x7FFFFFF0, 4, 4, format="rgb565")
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0x7FFFFFF0, 4, 4, wait_value=3)  # a value needs a fence
                with self.assertRaises(ValueError):
                    widget.submit_d3d11(0x7FFFFFF0, 4, 4, fence=0x7FFFFFF0)  # neither is a handle
            else:
                with self.assertRaises(RuntimeError):
                    widget.submit_d3d11(0, 4, 4)

    def test_status_is_unused_until_the_render_thread_has_seen_a_submission(self):
        self.assertEqual(fg.Viewport().d3d11_status, "unused")
        self.assertEqual(fg.Image().d3d11_status, "unused")

    def test_adapter_luid_is_a_windows_adapter_id_or_none(self):
        luid = fg.adapter_luid()
        if sys.platform != "win32":
            self.assertIsNone(luid)
        else:
            self.assertTrue(luid is None or (isinstance(luid, int) and luid > 0))


if __name__ == "__main__":
    unittest.main()
