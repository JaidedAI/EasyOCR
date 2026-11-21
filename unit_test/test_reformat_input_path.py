import cv2
import numpy as np

from easyocr.utils import reformat_input


def test_reformat_input_expands_user_in_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    pixels = np.random.RandomState(0).randint(0, 255, (20, 30, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "text.png"), pixels)

    img, img_cv_grey = reformat_input("~/text.png")
    expected_img, expected_grey = reformat_input(str(tmp_path / "text.png"))

    assert img_cv_grey is not None
    assert img_cv_grey.shape == (20, 30)
    assert np.array_equal(img_cv_grey, expected_grey)
    assert np.array_equal(img, expected_img)
