import numpy as np
import pytest

from easyocr.utils import reformat_input


@pytest.mark.parametrize("shape", [(1, 7), (7, 1), (1, 1), (3, 7)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_single_channel_input_preserves_spatial_dimensions(shape, dtype):
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    image = pixels[:, :, None]

    color, grayscale = reformat_input(image)

    assert grayscale.shape == shape
    assert color.shape == (*shape, 3)
    np.testing.assert_array_equal(grayscale, pixels)
    np.testing.assert_array_equal(color, np.repeat(image, 3, axis=2))
    assert grayscale.dtype == dtype
    assert color.dtype == dtype
