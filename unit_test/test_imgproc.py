import numpy as np
import pytest
from PIL import Image

from easyocr.imgproc import loadImage


@pytest.mark.parametrize("height", [1, 2, 3])
@pytest.mark.parametrize("channels", [None, 3, 4])
def test_load_image_preserves_rows(tmp_path, height, channels):
    shape = (height, 7) if channels is None else (height, 7, channels)
    pixels = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)

    actual = loadImage(str(path))

    expected = (np.repeat(pixels[:, :, None], 3, axis=2)
                if channels is None else pixels[:, :, :3])
    np.testing.assert_array_equal(actual, expected)


def test_load_image_keeps_first_frame_of_two_frame_gif(tmp_path):
    first = np.full((5, 7, 3), [255, 0, 0], dtype=np.uint8)
    second = np.full((5, 7, 3), [0, 0, 255], dtype=np.uint8)
    path = tmp_path / "animated.gif"
    Image.fromarray(first).save(
        path, save_all=True, append_images=[Image.fromarray(second)], duration=100
    )

    np.testing.assert_array_equal(loadImage(str(path)), first)
