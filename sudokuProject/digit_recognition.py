import cv2
import numpy as np
import os
from skimage.feature import hog


def load_templates(template_dir="templates"):
    """
    Load multiple templates per digit and compute HOG features
    """
    templates = {}

    for filename in os.listdir(template_dir):
        if not filename.endswith(".png"):
            continue

        # Remove extension and extract digit
        name = os.path.splitext(filename)[0]   # e.g. "5_2" or "8"
        digit = int(name.split("_")[0])

        path = os.path.join(template_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        if digit not in templates:
            templates[digit] = []

        templates[digit].append(features)

    return templates



def recognize_digit(digit_img, templates):
    """
    Recognize a digit using minimum HOG distance across templates
    """
    features = hog(
        digit_img,
        orientations=9,
        pixels_per_cell=(7, 7),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    best_digit = None
    min_distance = float("inf")

    for digit, feature_list in templates.items():
        for tmpl_features in feature_list:
            dist = np.linalg.norm(features - tmpl_features)
            if dist < min_distance:
                min_distance = dist
                best_digit = digit

    return best_digit
