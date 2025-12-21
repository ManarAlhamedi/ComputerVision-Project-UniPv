import cv2
from sudokuProject.preprocess import preprocess_image
from grid_detection import find_sudoku_contour
from sudokuProject.perspective import warp_sudoku
from sudokuProject.segmentation import split_into_cells
from digit_extraction import extract_digit
from digit_recognition import load_templates, recognize_digit

# -------------------------
# Load image
# -------------------------
image = cv2.imread("sudoku.jpg")
if image is None:
    print("Error: sudoku.jpg not found")
    exit()

original = image.copy()

# -------------------------
# Preprocess
# -------------------------
binary = preprocess_image(image)

# -------------------------
# Detect grid
# -------------------------
sudoku_contour = find_sudoku_contour(binary)
if sudoku_contour is None:
    print("Sudoku grid not detected")
    exit()

# Draw detected grid on original image
cv2.drawContours(image, [sudoku_contour], -1, (0, 255, 0), 3)

# -------------------------
# Warp grid
# -------------------------
warped = warp_sudoku(original, sudoku_contour)

# -------------------------
# Split into cells
# -------------------------
cells = split_into_cells(warped)

# -------------------------
# Load digit templates
# -------------------------
templates = load_templates("templates/blurred")

# -------------------------
# Recognize digits
# -------------------------
sudoku_result = []

for r in range(9):
    row = []
    for c in range(9):
        digit_img = extract_digit(cells[r][c])
        if digit_img is None:
            row.append(0)
        else:
            digit = recognize_digit(digit_img, templates)
            row.append(digit)
    sudoku_result.append(row)

# -------------------------
# Print result
# -------------------------
print("Recognized Sudoku:")
for row in sudoku_result:
    print(row)

# -------------------------
# VISUALIZATION (IMPORTANT)
# -------------------------

# Show original image with detected grid
cv2.imshow("Detected Sudoku Grid", image)

# Show warped (top-down) grid
cv2.imshow("Warped Sudoku Grid", warped)

# Show a few cells (first row)
for i in range(5):
    cv2.imshow(f"Cell (0,{i})", cells[0][i])

# Show a few extracted digits
shown = 0
for r in range(9):
    for c in range(9):
        digit_img = extract_digit(cells[r][c])
        if digit_img is not None:
            cv2.imshow(f"Extracted Digit ({r},{c})", digit_img)
            shown += 1
        if shown == 5:
            break
    if shown == 5:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
