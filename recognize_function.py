import easyocr
import os
import cv2
import numpy as np
from easyocr.easyocr import Reader

def recognize_text_from_images(image_pieces, models_directory, recog_network='best_accuracy', gpu=False):
    """
    Recognizes text from a list of image pieces using EasyOCR.

    Parameters:
    - image_pieces (list): List of image pieces as PIL Image objects.
    - models_directory (str): Path to the models directory.
    - recog_network (str): Recognition network to use (default is 'best_accuracy').
    - gpu (bool): Whether to use GPU for OCR (default is False).

    Returns:
    - List of recognized texts.
    """
    model_storage_directory = os.path.join(models_directory, "model")
    user_network_directory = os.path.join(models_directory, "user_network")

    # Initialize EasyOCR reader
    reader = Reader(['ru'], recog_network=recog_network, gpu=gpu,
                            model_storage_directory=model_storage_directory,
                            user_network_directory=user_network_directory)

    recognized_texts = []
    for image_piece in image_pieces:
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_piece), cv2.COLOR_RGB2BGR)
        # Perform text recognition
        result = reader.readtext(image_cv, detail=0)
        recognized_texts.append(" ".join(result))
    
    return recognized_texts