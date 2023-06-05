import cv2
import easyocr
from PIL import Image
import os
import numpy as np
from typing import Tuple, Union
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

angle_zero = 0.16
target_size = (400, 72)

# Initialize the Reader object with the languages you want to recognize and the desired parameters
reader = easyocr.Reader(['en', 'tr'], gpu=True)

reader.model_path = './EasyOCR-master/trainer/saved_models/custom_model/best_model.pth'

# Path to the directory containing image files
dir_path = './EasyOCR-master/tests/inputs'
out_dir = './EasyOCR-master/tests/outputs'

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


        # Perform OCR on the image
        result = reader.readtext(image_np, allowlist='0123456789:',
                                 contrast_ths=0.1, adjust_contrast=0.5,
                         text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560,
                           mag_ratio=1.0, slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5,
                             width_ths=0.5, add_margin=0.1, x_ths=1.0, y_ths=0.5,
                         decoder='greedy', beamWidth=50, batch_size=2)

    # Plot the image with bounding boxes and recognized text
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image_np, cmap='gray')


    for r in result:
        bbox = r[0]
        text = r[1]
        confidence = r[2]
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        x3, y3 = bbox[2]
        x4, y4 = bbox[3]
        poly = plt.Polygon(bbox, facecolor=None, edgecolor='green', linewidth=2, fill=False)
        ax.add_patch(poly)
        ax.text(x1, y1-10, f'{text} ({confidence:.2f})', fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    # Save the plotted image to a file
    plt.savefig(os.path.join(out_dir, f'{filename}'))

    # Close the figure to release memory resources
    plt.close(fig)

