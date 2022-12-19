import json

import cv2
import numpy as np
from scipy import stats as st


class StyleDetector:

    font_width_map = {1: "light", 2: "regular", 3: "medium", 4: "semi_bold", 5: "bold", 6: "extra_bold"}

    def __init__(self, image, boundary):
        self.image = image
        self.boundary = boundary
        self.cropped_image = None
        self.binary_image = None
        self.text = None
        self.font_height = None
        self.font_width = None
        self.font_color = None
        self.background_color = None

    def get_text_properties(self, text):
        self.text = text

        self.crop_bounding_region()
        self.create_binary_image()
        self.font_height = self.get_text_height()
        self.font_width = self.get_text_width()
        colors = self.get_component_colors()
        self.font_color = colors[0]
        self.background_color = colors[1]

        output = {'font-height': self.font_height, 'font-width': self.font_width, 'font-color': list(self.font_color),
                  'background-color': list(self.background_color)}

        return output

    def get_text_height(self):
        column_wise_sum = sum(self.binary_image)
        column_wise_sum = column_wise_sum != 0
        column_wise_sum.astype(int)

        column_wise_sum = np.concatenate(([0], column_wise_sum, [0]))
        abs_diff = np.abs(np.diff(column_wise_sum))
        index_ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)

        letter_heights = np.apply_along_axis(StyleDetector.computer_letter_wise_height, axis=1, arr=index_ranges,
                                             img=self.binary_image)

        text = self.text.replace(" ", "")[:len(letter_heights)]

        font_size = []

        if text.upper().isupper():
            for i, j in zip(text, letter_heights):
                if i.isalpha() and i != 'S':
                    font_size.append(j)
        else:
            for i, j in zip(text, letter_heights):
                if i.isalnum() and i != 'S':
                    font_size.append(j)

        if len(font_size) > 0:
            return round(max(font_size) * 1.333333)
        else:
            return round(max(letter_heights) * 1.333333)

    def get_text_width(self):

        row_wise_weights_mode = np.apply_along_axis(StyleDetector.compute_row_wise_width, axis=1, arr=self.binary_image)
        font_weight = st.mode(row_wise_weights_mode, keepdims=True)[0][0]
        font_width = self.font_width_map.get(font_weight, "bold")
        return font_width

    @staticmethod
    def computer_letter_wise_height(array, img):
        """
        Helper Function to calculate each letter height of text present in the image
        @param array: Index Position of Each Letter in the Image Array
        @param img: Binary Image of text
        @return: Height of the Letter
        """

        letter = img[:, array[0]:array[1]]
        row_wise_sum = letter.sum(axis=1)
        non_zero_indexes = np.nonzero(row_wise_sum)[0]

        if len(non_zero_indexes) > 1:
            return non_zero_indexes[len(non_zero_indexes) - 1] - non_zero_indexes[0]
        elif len(non_zero_indexes) == 1:
            return 1
        else:
            return 0

    @staticmethod
    def compute_row_wise_width(img_row):
        img_arr = np.concatenate(([0], img_row, [0]))
        abs_diff = np.abs(np.diff(img_arr))
        index_ranges = np.where(abs_diff == 1)[0].reshape(-1, 2)
        if len(index_ranges) == 0:
            return 0
        return st.mode(np.diff(index_ranges).reshape(-1), keepdims=True)[0][0]

    def crop_bounding_region(self):
        x_min, y_min, x_max, y_max = self.boundary
        self.cropped_image = self.image[y_min:y_max, x_min:x_max]

    def create_binary_image(self):
        grayscale_img = cv2.cvtColor(self.cropped_image, cv2.COLOR_RGB2GRAY)
        thresh, binary_image = cv2.threshold(grayscale_img, 150, 255, cv2.THRESH_OTSU)
        binary_image[binary_image == 255] = 1

        arr = np.sort(binary_image.sum(axis=0))
        one_sum = arr[len(arr) - 1]
        zero_sum = binary_image.shape[0] - arr[0]

        if zero_sum < one_sum:
            binary_image = (1 - binary_image)

        self.binary_image = binary_image

    def get_component_colors(self):
        var1 = np.where(self.binary_image == 0)
        colors = self.cropped_image[var1[0], var1[1]]
        background_color = colors.mean(axis=0, dtype=int)

        var1 = np.where(self.binary_image == 1)
        colors = self.cropped_image[var1[0], var1[1]]
        foreground_color = colors.mean(axis=0, dtype=int)

        return [foreground_color, background_color]

    def get_dominant_color(self):
        image = cv2.resize(self.cropped_image[:, :10, :], (1, 1), cv2.INTER_AREA)
        return tuple(image[0, 0])


class NpEncoder(json.JSONEncoder):
    """Class Encodes the Final output to convert to Json"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)