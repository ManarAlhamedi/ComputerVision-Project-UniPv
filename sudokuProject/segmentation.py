import cv2
import numpy as np

def split_into_cells(warped, grid_size=9):
    """
    Split the warped Sudoku grid into 81 cells
    """
    cells = []
    h, w = warped.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    for row in range(grid_size):
        row_cells = []
        for col in range(grid_size):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            cell = warped[y1:y2, x1:x2]
            row_cells.append(cell)

        cells.append(row_cells)

    return cells
