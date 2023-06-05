import cv2
from PIL import Image
import os
import numpy as np
from typing import Tuple, Union
import math

angle_zero = 0.16
target_size = (400, 72)

# Path to the directory containing image files
dir_path = 'C:/Users/yilma/OneDrive/Desktop/Work/Project3/EasyOCR-master/tests/inputs'
out_dir = 'C:/Users/yilma/OneDrive/Desktop/Work/Project3/EasyOCR-master/tests/outputs'

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

# Loop through all image files in the directory
for filename in os.listdir(dir_path):
    # Load the image and convert it to a NumPy array
    image_path = os.path.join(dir_path, filename)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    w, h = image.shape

    # Crop the left 12.5% of the image
    crop_width = image.shape[1] // 8
    image_cropped = image[:, :crop_width]

    # Split the cropped image into top and bottom halves
    height, width = image_cropped.shape
    top_half = image_cropped[:height//2, :]
    bottom_half = image_cropped[height//2:, :]

    # Compute the mean intensity values of the top and bottom halves
    top_whiteness = np.mean(255 - top_half)
    bottom_whiteness = np.mean(255 - bottom_half)

    # Print the aspect ratio, top whiteness, and bottom whiteness
    aspect_ratio = round(float(w) / float(h), 2)
    print(f"Aspect Ratio: {aspect_ratio}, Top Whiteness: {top_whiteness}, Bottom Whiteness: {bottom_whiteness}")


    image_np = np.array(image)
    image_np = np.rot90(image_np, k=1)

    if aspect_ratio >= angle_zero:
        angle = math.degrees(math.atan(aspect_ratio - angle_zero)) - 90
        image_rot1 = rotate(image_np, angle, (0, 0, 0))
        image_rot2 = rotate(image_np, 180-angle, (0, 0, 0))

        # Compare the mean intensity values of the top and bottom halves
        threshold = 0
        if top_whiteness > bottom_whiteness * (1 + threshold):
            image_np = image_rot2
            print(f"{filename}: top half is brighter than bottom half")
        elif bottom_whiteness > top_whiteness * (1 + threshold):
            print(f"{filename}: bottom half is brighter than top half")
            image_np = image_rot1
        else:
            print(f"{filename}: top and bottom halves have similar brightness")


    # Save the resized image to a file
    out_path = os.path.join(out_dir, f'{filename}_processed.jpg')
    Image.fromarray(image_np).save(out_path)

