import importlib.util
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


def _load_utils():
    root = Path(__file__).resolve().parent
    pkg = types.ModuleType("easyocr")
    pkg.__path__ = [str(root / "easyocr")]
    sys.modules["easyocr"] = pkg
    imgproc = types.ModuleType("easyocr.imgproc")
    imgproc.loadImage = lambda path: None
    sys.modules["easyocr.imgproc"] = imgproc
    path = root / "easyocr" / "utils.py"
    spec = importlib.util.spec_from_file_location("easyocr.utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["easyocr.utils"] = module
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def _png(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


class TestByteGray(unittest.TestCase):
    def test_a_red_byte_image_keeps_the_red_weight(self):
        bgr = np.zeros((2, 2, 3), np.uint8)
        bgr[:, :, 2] = 255
        image, gray = utils.reformat_input(_png(bgr))
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(int(gray[0, 0]), 76)

    def test_a_blue_byte_image_keeps_the_blue_weight(self):
        bgr = np.zeros((2, 2, 3), np.uint8)
        bgr[:, :, 0] = 255
        image, gray = utils.reformat_input(_png(bgr))
        self.assertEqual(image[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(int(gray[0, 0]), 29)

    def test_a_neutral_byte_image_stays_neutral(self):
        bgr = np.full((2, 2, 3), 40, np.uint8)
        image, gray = utils.reformat_input(_png(bgr))
        self.assertEqual(int(gray[0, 0]), 40)


if __name__ == "__main__":
    unittest.main()
