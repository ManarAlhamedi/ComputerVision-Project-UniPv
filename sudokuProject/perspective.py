import cv2
import numpy as np

def order_points(pts):
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[s.argmin()]  # top-left
    rect[2] = pts[s.argmax()]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[diff.argmin()]  # top-right
    rect[3] = pts[diff.argmax()]  # bottom-left

    return rect


def warp_sudoku(image, contour, size=450):
    """
    Apply perspective transform to get top-down Sudoku grid
    """
    rect = order_points(contour)

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (size, size))

    return warped
