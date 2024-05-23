import easyocr
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Mock function to return bounding boxes for text lines
def get_text_line_boxes(image):
    # This function should return a list of bounding boxes for text lines
    # Here we return mock bounding boxes as an example
    height, width, _ = image.shape
    return [
        [(50, 50), (width - 50, 50), (width - 50, 100), (50, 100)],
        [(50, 150), (width - 50, 150), (width - 50, 200), (50, 200)],
        [(50, 250), (width - 50, 250), (width - 50, 300), (50, 300)]
    ]

# Load the image
image_path = "test2 (1).jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL Image for better font rendering
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)

# Load a font (make sure the path to the font file is correct)
font_path = "arial.ttf"  # Replace with the path to your .ttf font file
font = ImageFont.truetype(font_path, 60)  # Increased font size to 60

# Get bounding boxes for text lines
line_boxes = get_text_line_boxes(image_rgb)

# Initialize EasyOCR reader
reader = easyocr.Reader(['ru'], recog_network='best_accuracy')

# Process each bounding box with EasyOCR
for bbox in line_boxes:
    # Convert bbox to format expected by EasyOCR (top_left, top_right, bottom_right, bottom_left)
    top_left, top_right, bottom_right, bottom_left = bbox

    # Crop the image to the bounding box
    cropped_image = image_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Save the cropped image temporarily
    cropped_image_pil = Image.fromarray(cropped_image)
    cropped_image_path = "temp_cropped_image.jpg"
    cropped_image_pil.save(cropped_image_path)

    # Perform text detection and recognition
    results = reader.readtext(cropped_image_path)

    # Print the results
    print(results)

    # Draw bounding boxes and recognized text on the original image
    for (sub_bbox, text, prob) in results:
        # Adjust sub_bbox coordinates relative to the original image
        sub_top_left, sub_top_right, sub_bottom_right, sub_bottom_left = sub_bbox
        sub_top_left = (sub_top_left[0] + top_left[0], sub_top_left[1] + top_left[1])
        sub_bottom_right = (sub_bottom_right[0] + top_left[0], sub_bottom_right[1] + top_left[1])

        # Draw the bounding box on the image
        draw.rectangle([sub_top_left, sub_bottom_right], outline="green", width=2)

        # Put the recognized text
        draw.text((sub_top_left[0], sub_top_left[1] - 40), text, font=font, fill="green")

# Convert back to OpenCV format
image_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Display the image with bounding boxes and recognized text
plt.figure(figsize=(10, 10))  # Adjusted the figure size for better visibility
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig("text.png")
plt.show()