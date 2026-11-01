"""Time a numpy frame against mtlpy's fast texture path.

The fast texture is the one mtlpy documents for GPU sampling:
`empty_texture(..., readable=True, writable=False, private=True)`.
Private storage is GPU-only, and dropping shader-write lets Metal keep a
tighter layout. In a hot loop the fast upload reuses one Buffer:
`contents[:] = frame`, then `upload_from_buffer` (a blit, no fresh staging
buffer). `submit_frame` is the CPU path: a host copy, then `replaceRegion`
on the render thread.

No window. Prints milliseconds per frame.
"""

import time

import numpy as np

import fastgui as fg
import mtlpy


def median(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2]


def time_call(fn, warmup: int = 8, frames: int = 40) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(frames):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1e3)
    return median(samples)


def main() -> None:
    device = mtlpy.Device()
    viewport = fg.Viewport()
    for height, width in ((1080, 1920), (2160, 3840)):
        frame = np.zeros((height, width, 4), np.uint8)
        frame[..., 0] = 220
        frame[..., 3] = 255
        texture = device.empty_texture(
            (height, width), "rgba8Unorm", readable=True, writable=False, private=True
        )
        stage = device.empty(frame.size, np.uint8)
        flat = frame.reshape(-1)
        print(
            f"{width}x{height}  private={texture.is_private} "
            f"readable={texture.readable} writable={texture.writable}  "
            f"{frame.nbytes / 1e6:.1f} MB"
        )
        print(f"  numpy submit_frame          {time_call(lambda: viewport.submit_frame(frame)):7.3f} ms")
        print(
            f"  buffer copy + blit upload   "
            f"{time_call(lambda: (stage.contents.__setitem__(slice(None), flat), texture.upload_from_buffer(stage))):7.3f} ms"
        )
        print(f"  texture.upload (staging)    {time_call(lambda: texture.upload(frame)):7.3f} ms")
        print(f"  submit_metal (same texture) {time_call(lambda: viewport.submit_metal(texture)):7.3f} ms")


if __name__ == "__main__":
    main()
