try:
    import cv2
except ImportError as imp_e:
    print("Module cv2 was not found. This project was not compiled correctly")
    raise imp_e

try:
    import torch
except ImportError as imp_e:
    print("Module torch was not found. This project was not compiled correctly")
    raise imp_e

from .easyocr import Reader

__version__ = '1.7.0'
