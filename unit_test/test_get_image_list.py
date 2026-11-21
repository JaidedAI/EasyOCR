"""Regression tests for empty/out-of-bounds crops in get_image_list (#1437)."""

import numpy as np

from easyocr.utils import get_image_list


def test_get_image_list_skips_empty_and_out_of_bounds_boxes():
    img = np.zeros((100, 100), dtype=np.uint8)
    horizontal_list = [
        [105, 120, 10, 50],  # x_min > x_max after clamping -> empty crop, negative width
        [10, 50, 110, 130],  # y_min > y_max after clamping -> empty crop, negative height
        [20, 20, 10, 40],  # zero width
        [10, 40, 30, 30],  # zero height
        [10, 50, 10, 30],  # valid horizontal box
        [10, 20, 10, 50],  # valid vertical box
    ]
    image_list, max_width = get_image_list(
        horizontal_list, [], img, model_height=64, sort_output=True
    )

    assert len(image_list) == 2
    assert max_width >= 64
    for _, crop in image_list:
        assert crop.size > 0
