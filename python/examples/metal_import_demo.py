"""Sample an mtlpy texture in a fastgui viewport.

The texture stays on the GPU: `submit_metal` takes `Texture.mtl_ptr` and the
window draws it. Left half is red, right half is green, and the top half adds
blue. Close the window to exit.

Requires mtlpy (`pip install` the wheel from the mtlpy repo) and macOS.
"""

import numpy as np

import fastgui as fg

try:
    import mtlpy
except ImportError as exc:
    raise SystemExit("mtlpy is not installed") from exc


WIDTH, HEIGHT = 640, 480


def pattern() -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    image[:, : WIDTH // 2, 0] = 255
    image[:, WIDTH // 2 :, 1] = 255
    image[: HEIGHT // 2, :, 2] = 255
    image[..., 3] = 255
    return image


def main() -> None:
    device = mtlpy.Device()
    # Private storage is what a compute texture usually is. Upload blits into it.
    texture = device.empty_texture((HEIGHT, WIDTH), "rgba8Unorm", readable=True, writable=False, private=True)
    texture.upload(pattern())
    window = fg.Window(title="fastgui — Metal import", width=WIDTH, height=HEIGHT)
    viewport = fg.Viewport()
    window.set_viewport(viewport)
    viewport.submit_metal(texture)
    window.run()


if __name__ == "__main__":
    main()
