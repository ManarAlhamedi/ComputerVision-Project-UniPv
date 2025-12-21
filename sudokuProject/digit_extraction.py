import cv2
import numpy as np

def extract_digit(cell, min_area_ratio=0.02):
    """
    Extract digit from a Sudoku cell.
    Returns None if the cell is empty.
    """
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Remove borders (important!)
    h, w = thresh.shape
    margin = int(min(h, w) * 0.1)
    thresh[:margin, :] = 0
    thresh[-margin:, :] = 0
    thresh[:, :margin] = 0
    thresh[:, -margin:] = 0

    # Find contours inside the cell
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Take largest contour (likely digit)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # Reject small noise
    if area < min_area_ratio * h * w:
        return None

    # Create mask for digit
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [cnt], -1, 255, -1)

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Crop bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    digit = digit[y:y+h, x:x+w]

    # Resize to standard size
    digit = cv2.resize(digit, (28, 28))

    return digit
