import easyocr
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


# Load the image
image_path = "test2 (1).jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL Image for better font rendering
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)

# Load a font (make sure the path to the font file is correct)
font_path = "arial.ttf"  # Replace with the path to your .ttf font file
font = ImageFont.truetype(font_path, 60)  # Increased font size to 40

models_directory = "C:/Users/user/Desktop/modelsocr"
model_storage_directory = os.path.join(models_directory, "model")
user_network_directory = os.path.join(models_directory, "user_network")
# Initialize EasyOCR reader
reader = easyocr.Reader(['ru'], recog_network='best_accuracy', gpu = False,
                        model_storage_directory=model_storage_directory,
                        user_network_directory=user_network_directory)

# Perform text detection and recognition
results = reader.readtext(image_path)

# Print the results
print(results)

# Draw bounding boxes and placeholder text on the image
for (bbox, text, prob) in results:
    # Replace the recognized text with "вопросы"
    
    # Unpack the bounding box coordinates
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple([int(val) for val in top_left])
    bottom_right = tuple([int(val) for val in bottom_right])

    # Draw the bounding box on the image
    draw.rectangle([top_left, bottom_right], outline="green", width=2)

    # Put the placeholder text
    draw.text((top_left[0], top_left[1] - 40), text, font=font, fill="green")  # Adjusted y-coordinate for larger font

# Convert back to OpenCV format
image_rgb = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Display the image with bounding boxes and placeholder text
plt.figure(figsize=(60, 60))  # Adjusted the figure size for better visibility
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig("text.png")
plt.show()