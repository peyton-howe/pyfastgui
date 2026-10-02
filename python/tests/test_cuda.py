"""Viewport.submit_cuda / create_cuda_surface without a window or (necessarily) a GPU.

Fake `__cuda_array_interface__` objects exercise everything that happens before the CUDA
driver is touched: shape/dtype/stride validation, stream-argument parsing, and the host-array
fallthrough. Past validation, a fake pointer goes to the host-copy path, which needs the driver;
on a machine without one that has to be a clean RuntimeError, not a crash.
"""

import unittest

import numpy as np

import fastgui as fg


class FakeCudaArray:
    def __init__(self, shape=(4, 6, 4), typestr="|u1", strides=None, stream=1, ptr=0x7F0000001000, **extra):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "strides": strides,
            "version": 3,
            **({} if stream == "absent" else {"stream": stream}),
            **extra,
        }


def driver_available() -> bool:
    """True when submitting a well-formed fake array gets past driver loading (it then fails
    on the bogus pointer instead)."""
    try:
        fg.Viewport().submit_cuda(FakeCudaArray())
    except RuntimeError as err:
        return "failed to load the CUDA driver" not in str(err)
    return True


class SubmitCudaValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewport = fg.Viewport()

    def assert_rejected(self, array, message: str, error=ValueError) -> None:
        with self.assertRaises(error) as ctx:
            self.viewport.submit_cuda(array)
        self.assertIn(message, str(ctx.exception))

    def test_rgb_is_rejected_with_a_hint(self) -> None:
        self.assert_rejected(FakeCudaArray(shape=(4, 6, 3)), "add an alpha channel")

    def test_channels_first_is_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(shape=(4, 6, 6)), "permute")

    def test_non_uint8_is_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(typestr="<f4"), "<f4")

    def test_empty_frame_is_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(shape=(0, 6, 4)), "at least 1x1")

    def test_oversized_frame_is_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(shape=(4, 100_000, 4)), "frame edge")

    def test_non_contiguous_pixels_are_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(strides=(48, 8, 2)), "strides")
        self.assert_rejected(FakeCudaArray(strides=(16, 4, 1)), "strides")  # rows overlap

    def test_masked_arrays_are_rejected(self) -> None:
        self.assert_rejected(FakeCudaArray(mask=object()), "masked")

    def test_bad_interface_type(self) -> None:
        class NotADict:
            __cuda_array_interface__ = [1, 2, 3]

        self.assert_rejected(NotADict(), "must be a dict", TypeError)

    def test_bad_stream_argument(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            self.viewport.submit_cuda(FakeCudaArray(), stream=object())
        self.assertIn("stream must be", str(ctx.exception))

    def test_well_formed_arrays_get_past_validation(self) -> None:
        class TorchLikeStream:
            cuda_stream = 0x1234

        class ProtocolStream:
            def __cuda_stream__(self):
                return (0, 0x5678)

        cases = [
            (FakeCudaArray(), None),
            (FakeCudaArray(strides=(6 * 4 + 8, 4, 1)), None),  # padded rows
            (FakeCudaArray(typestr="<u1", stream=None), None),
            (FakeCudaArray(stream="absent"), None),
            (FakeCudaArray(), 7),
            (FakeCudaArray(), TorchLikeStream()),
            (FakeCudaArray(), ProtocolStream()),
        ]
        for array, stream in cases:
            with self.subTest(cai=array.__cuda_array_interface__, stream=stream):
                # No window, so this goes to the host copy, which reads the (bogus) pointer
                # through the driver: an error either way here, but never a ValueError/TypeError.
                with self.assertRaises(RuntimeError) as ctx:
                    self.viewport.submit_cuda(array, stream=stream)
                self.assertIn("can't read the CUDA array", str(ctx.exception))

    @unittest.skipIf(driver_available(), "a CUDA driver is installed")
    def test_missing_driver_is_a_clean_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self.viewport.submit_cuda(FakeCudaArray())
        self.assertIn("failed to load the CUDA driver", str(ctx.exception))


class SubmitCudaHostArrayTests(unittest.TestCase):
    def test_host_arrays_go_through_submit_frame(self) -> None:
        viewport = fg.Viewport()
        viewport.submit_cuda(np.zeros((4, 6, 4), dtype=np.uint8))
        viewport.submit_cuda(np.zeros((4, 6, 3), dtype=np.uint8))  # RGB is fine on the host path
        self.assertEqual(viewport.cuda_status, "unused")
        with self.assertRaises(ValueError):
            viewport.submit_cuda(np.zeros((4, 6), dtype=np.uint8))


class PlotCudaIngestTests(unittest.TestCase):
    """Plots read CUDA float arrays through fastgui_interop_cuda.copy_to_host."""

    @staticmethod
    def fake(**overrides):
        class FakeFloats:
            __cuda_array_interface__ = {
                "shape": (4,),
                "typestr": "<f4",
                "data": (0x7F0000001000, False),
                "strides": None,
                "version": 3,
                **overrides,
            }

        return FakeFloats()

    @unittest.skipIf(driver_available(), "a CUDA driver is installed")
    def test_valid_array_reaches_the_driver(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            fg.PlotLine(self.fake(), [0.0, 1.0, 2.0, 3.0])
        self.assertIn("CUDA device array ingest failed", str(ctx.exception))
        self.assertIn("failed to load the CUDA driver", str(ctx.exception))

    def test_bad_stream_is_rejected_before_any_copy(self) -> None:
        with self.assertRaises(TypeError):
            fg.PlotLine(self.fake(stream="default"), [0.0, 1.0, 2.0, 3.0])


class CudaSurfaceTests(unittest.TestCase):
    def test_create_needs_a_window(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            fg.Viewport().create_cuda_surface(64, 32)
        self.assertIn("set_viewport", str(ctx.exception))

    def test_create_rejects_bad_sizes(self) -> None:
        for size in [(0, 32), (64, 0), (100_000, 32)]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                fg.Viewport().create_cuda_surface(*size)

    def test_classes_are_exported(self) -> None:
        self.assertIn("CudaSurface", fg.__all__)
        self.assertIn("CudaFrame", fg.__all__)
        self.assertEqual(fg.Viewport().cuda_status, "unused")


if __name__ == "__main__":
    unittest.main()
