"""
This project implements Binary morphology (erosion/dilation/opening/closing) from scratch.

Example:
  python morphology.py --op erode --in input.png --out out.png --ksize 5 --shape square --iters 2
  python morphology.py --op dilate --in input.png --out out.png --ksize 7 --shape disk --pad constant --pad-value 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Literal, Dict, Callable, List, Tuple, Optional


KernelShape = Literal["square", "cross", "disk"]
PadMode = Literal["constant", "edge"]


def make_structuring_element(ksize: int, shape: KernelShape) -> np.ndarray:
    """
    Create a boolean structuring element (SE) of size ksize x ksize.

    Parameters
    ----------
    ksize : int
        Odd kernel size (3, 5, 7). Must be >= 1 and odd.
    shape : {"square", "cross", "disk"}
        Shape of the structuring element:
        - "square": all ones in the ksize x ksize window
        - "cross": ones on the center row and center column
        - "disk": approximate filled circle inscribed in the window

    Returns
    -------
    se : (ksize, ksize) np.ndarray of bool
        Boolean mask indicating which neighbors are considered.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 1")

    if shape == "square":
        return np.ones((ksize, ksize), dtype=bool)

    c = ksize // 2

    if shape == "cross":
        se = np.zeros((ksize, ksize), dtype=bool)
        se[c, :] = True
        se[:, c] = True
        return se

    if shape == "disk":
        yy, xx = np.ogrid[:ksize, :ksize]
        r = c
        return (yy - c) ** 2 + (xx - c) ** 2 <= r * r

    raise ValueError(f"Unknown shape: {shape}")


def _pad_binary(
    img: np.ndarray,
    pad: int,
    mode: PadMode,
    constant_value: int = 0,
) -> np.ndarray:
    if pad == 0:
        return img

    if mode == "constant":
        return np.pad(img, pad_width=pad, mode="constant", constant_values=constant_value)
    if mode == "edge":
        return np.pad(img, pad_width=pad, mode="edge")

    raise ValueError(f"Unknown pad mode: {mode}")


def binary_erosion(
    image: np.ndarray,
    se: np.ndarray,
    iterations: int = 1,
    pad_mode: PadMode = "constant",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Perform binary erosion on a 2D binary image using a given structuring element.

    Definition
    ----------
    Erosion shrinks foreground regions. A pixel in the output is 1 (True) if and only if
    *all* pixels under the structuring element (where se==True) are 1 in the input.

    Parameters
    ----------
    image : (H, W) np.ndarray
        Input binary image. Accepted dtypes: bool, uint8, int.
        Nonzero values are treated as foreground (1).
    se : (K, K) np.ndarray of bool
        Structuring element mask. True entries define which neighbors must be foreground
        for the output pixel to be foreground.
        K should be odd in both dimensions for a centered SE.
    iterations : int, default=1
        Number of times to apply erosion repeatedly.
    pad_mode : {"constant", "edge"}, default="constant"
        How to pad the image borders.
        - "constant": pads with pad_value (often 0 for erosion)
        - "edge": replicates border pixels
    pad_value : int, default=0
        Used only with pad_mode="constant". For erosion, 0 is typical.

    Returns
    -------
    out : (H, W) np.ndarray of bool
        Eroded binary image.

    Notes
    -----
    - This implementation is from scratch.
    - Runtime is O(H * W * nnz(se)) per iteration.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if se.ndim != 2:
        raise ValueError("se must be a 2D array")

    img = (image != 0)  # convert to bool foreground mask
    se = (se != 0)

    kh, kw = se.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Structuring element dimensions must be odd")

    pad = kh // 2
    se_coords = np.argwhere(se)  # list of (dy, dx) positions where se is True
    # Convert to offsets relative to center:
    cy, cx = pad, pad
    offsets = [(int(y - cy), int(x - cx)) for y, x in se_coords]

    out = img
    for _ in range(iterations):
        padded = _pad_binary(out.astype(np.uint8), pad=pad, mode=pad_mode, constant_value=pad_value).astype(bool)
        H, W = out.shape
        eroded = np.ones((H, W), dtype=bool)

        # For erosion: output is AND over all selected neighbors
        for dy, dx in offsets:
            eroded &= padded[pad + dy : pad + dy + H, pad + dx : pad + dx + W]

            # Early exit: if everything already False, can break
            if not eroded.any():
                break

        out = eroded

    return out


def binary_dilation(
    image: np.ndarray,
    se: np.ndarray,
    iterations: int = 1,
    pad_mode: PadMode = "constant",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Perform binary dilation on a 2D binary image using a given structuring element.

    Definition
    ----------
    Dilation grows foreground regions. A pixel in the output is 1 (True) if and only if
    *any* pixel under the structuring element (where se==True) is 1 in the input.

    Parameters
    ----------
    image : (H, W) np.ndarray
        Input binary image. Accepted dtypes: bool, uint8, int.
        Nonzero values are treated as foreground (1).
    se : (K, K) np.ndarray of bool
        Structuring element mask. True entries define which neighbors can activate
        the output pixel.
        K should be odd in both dimensions for a centered SE.
    iterations : int, default=1
        Number of times to apply dilation repeatedly.
    pad_mode : {"constant", "edge"}, default="constant"
        How to pad the image borders.
        - "constant": pads with pad_value (often 0 for dilation)
        - "edge": replicates border pixels
    pad_value : int, default=0
        Used only with pad_mode="constant". For dilation, 0 is typical.

    Returns
    -------
    out : (H, W) np.ndarray of bool
        Dilated binary image.

    Notes
    -----
    - This implementation is from scratch and does not use cv2 or scipy.
    - Runtime is O(H * W * nnz(se)) per iteration.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")
    if se.ndim != 2:
        raise ValueError("se must be a 2D array")

    img = (image != 0)
    se = (se != 0)

    kh, kw = se.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Structuring element dimensions must be odd")

    pad = kh // 2
    se_coords = np.argwhere(se)
    cy, cx = pad, pad
    offsets = [(int(y - cy), int(x - cx)) for y, x in se_coords]

    out = img
    for _ in range(iterations):
        padded = _pad_binary(out.astype(np.uint8), pad=pad, mode=pad_mode, constant_value=pad_value).astype(bool)
        H, W = out.shape
        dilated = np.zeros((H, W), dtype=bool)

        # For dilation: output is OR over all selected neighbors
        for dy, dx in offsets:
            dilated |= padded[pad + dy : pad + dy + H, pad + dx : pad + dx + W]

            # Early exit: if everything already True, can break
            if dilated.all():
                break

        out = dilated

    return out

def binary_opening(
    image: np.ndarray,
    se: np.ndarray,
    iterations: int = 1,
    pad_mode: PadMode = "constant",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Perform binary opening on a 2D binary image using a given structuring element.

    Definition
    ----------
    Opening is defined as an erosion followed by a dilation using the same structuring
    element:

        opening(image) = dilation( erosion(image, se), se )

    Effect
    ------------------
    - Removes small foreground noise (small bright specks).
    - Preserves the overall shape of larger objects better than erosion alone.

    Parameters
    ----------
    image : (H, W) np.ndarray
        Input binary image. Nonzero values are treated as foreground (1).
    se : (K, K) np.ndarray of bool
        Structuring element mask (True indicates included neighbors).
        K should be odd for a centered element.
    iterations : int, default=1
        Number of times to apply the opening operator.
        Each opening iteration is (erosion -> dilation).
    pad_mode : {"constant", "edge"}, default="constant"
        How to pad the image borders for the internal erosion/dilation steps.
    pad_value : int, default=0
        Constant padding value used when pad_mode="constant".
        Usually 0 is appropriate for typical binary morphology.

    Returns
    -------
    out : (H, W) np.ndarray of bool
        Opened (binary) image.

    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    out = (image != 0)
    for _ in range(iterations):
        out = binary_erosion(out, se, iterations=1, pad_mode=pad_mode, pad_value=pad_value)
        out = binary_dilation(out, se, iterations=1, pad_mode=pad_mode, pad_value=pad_value)
    return out


def binary_closing(
    image: np.ndarray,
    se: np.ndarray,
    iterations: int = 1,
    pad_mode: PadMode = "constant",
    pad_value: int = 0,
) -> np.ndarray:
    """
    Perform binary closing on a 2D binary image using a given structuring element.

    Definition
    ----------
    Closing is defined as a dilation followed by an erosion using the same structuring
    element:

        closing(image) = erosion( dilation(image, se), se )

    Effect
    ------------------
    - Fills small holes in foreground objects (small dark gaps inside bright regions).
    - Connects nearby components and bridges small breaks.
    - Smooths contours by closing narrow indentations.

    Parameters
    ----------
    image : (H, W) np.ndarray
        Input binary image. Nonzero values are treated as foreground (1).
    se : (K, K) np.ndarray of bool
        Structuring element mask (True indicates included neighbors).
        K should be odd for a centered element.
    iterations : int, default=1
        Number of times to apply the closing operator.
        Each closing iteration is (dilation -> erosion).
    pad_mode : {"constant", "edge"}, default="constant"
        How to pad the image borders for the internal dilation/erosion steps.
    pad_value : int, default=0
        Constant padding value used when pad_mode="constant".
        Usually 0 is appropriate for typical binary morphology.

    Returns
    -------
    out : (H, W) np.ndarray of bool
        Closed (binary) image.

    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    out = (image != 0)
    for _ in range(iterations):
        out = binary_dilation(out, se, iterations=1, pad_mode=pad_mode, pad_value=pad_value)
        out = binary_erosion(out, se, iterations=1, pad_mode=pad_mode, pad_value=pad_value)
    return out



def load_image_as_gray(path: str) -> np.ndarray:
    """Load an image and return grayscale uint8 array (H, W)."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_binary_image(path: str, binary: np.ndarray) -> None:
    """Save boolean/binary array as 8-bit image (0 or 255)."""
    out = (binary.astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(path)


# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Binary morphology (erosion/dilation) from scratch.")
#     p.add_argument("--op", choices=["erode", "dilate"], required=True, help="Operation to apply.")
#     p.add_argument("--in", dest="inp", required=True, help="Input image path.")
#     p.add_argument("--out", dest="out", required=True, help="Output image path.")
#     p.add_argument("--threshold", type=int, default=128, help="Binarization threshold for grayscale input.")
#     p.add_argument("--invert", action="store_true", help="Invert binary image after thresholding.")
#     p.add_argument("--ksize", type=int, default=3, help="Odd kernel size (3,5,7).")
#     p.add_argument("--shape", choices=["square", "cross", "disk"], default="square", help="Structuring element shape.")
#     p.add_argument("--iters", type=int, default=1, help="Number of iterations.")
#     p.add_argument("--pad", choices=["constant", "edge"], default="constant", help="Padding mode.")
#     p.add_argument("--pad-value", type=int, default=0, help="Constant padding value (0 or 1), used when pad=constant.")
#     return p.parse_args()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary morphology (erosion/dilation/opening/closing) from scratch.")
    p.add_argument("--op", choices=["erode", "dilate", "open", "close"], required=True, help="Operation to apply.")
    p.add_argument("--in", dest="inp", required=True, help="Input image path.")
    p.add_argument("--out", dest="out", required=True, help="Output image path.")
    p.add_argument("--threshold", type=int, default=128, help="Binarization threshold for grayscale input.")
    p.add_argument("--invert", action="store_true", help="Invert binary image after thresholding.")
    p.add_argument("--ksize", type=int, default=3, help="Odd kernel size (3,5,7).")
    p.add_argument("--shape", choices=["square", "cross", "disk"], default="square", help="Structuring element shape.")
    p.add_argument("--iters", type=int, default=1, help="Number of iterations.")
    p.add_argument("--pad", choices=["constant", "edge"], default="constant", help="Padding mode.")
    p.add_argument("--pad-value", type=int, default=0, help="Constant padding value (0 or 1), used when pad=constant.")
    return p.parse_args()



def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _binarize(gray: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    binary = (gray >= threshold)
    return (~binary) if invert else binary

def run_cli(args: argparse.Namespace) -> None:
    gray = load_image_as_gray(args.inp)
    binary = _binarize(gray, threshold=args.threshold, invert=args.invert)

    se = make_structuring_element(args.ksize, args.shape)

    ops: Dict[str, Callable[..., np.ndarray]] = {
        "erode": binary_erosion,
        "dilate": binary_dilation,
        "open": binary_opening,
        "close": binary_closing,
    }

    out = ops[args.op](
        binary,
        se,
        iterations=args.iters,
        pad_mode=args.pad,
        pad_value=args.pad_value,
    )

    save_binary_image(args.out, out)
    print(f"Saved: {Path(args.out).resolve()}")


def run_batch_grid() -> None:
    """
    Test binary morphology.

    What it does
    ------------
    - Reads all images from an input folder
    - Thresholds them to binary
    - Runs erosion/dilation/opening/closing over a grid of:
        * ksize
        * structuring element shape
        * iterations
        * padding mode (+ pad value)
    - Saves results to an output folder with descriptive filenames for your report.

    How to use
    ----------
    1) Put test images into:   ./test_images/
    2) Run:                   python morphology.py
    3) Check outputs in:      ./report_outputs/
    """
    # --------- user-editable settings ----------
    input_dir = Path("test_images")
    output_dir = Path("report_outputs")
    threshold = 128     # adjust per image if needed
    invert = False      # set True if your foreground is dark instead of bright

    # Grid of cases to run
    ksizes = [3, 7]
    shapes = ["square", "cross"]  
    iterations_list = [1, 3]

    # Padding cases:
    # - constant with pad_value 0 is common (assume background outside image)
    # - edge can reduce border artifacts for some images
    pad_cases: List[Tuple[str, int]] = [
        ("constant", 0),
        # You *can* test ("constant", 1) too, but it often produces strong border effects.
        # ("constant", 1),
    ]

    # Morphology ops to test
    ops: Dict[str, Callable[..., np.ndarray]] = {
        "erode": binary_erosion,
        "dilate": binary_dilation,
        "open": binary_opening,
        "close": binary_closing,
    }
    # -----------------------------------------

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir.resolve()}\n"
            f"Create it and add images, e.g. PNG/JPG/PGM."
        )

    _ensure_dir(output_dir)

    # Collect images
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}
    image_paths = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in exts]
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {input_dir.resolve()}.\n"
            f"Supported extensions: {sorted(exts)}"
        )

    # Run
    for img_path in image_paths:
        gray = load_image_as_gray(str(img_path))
        binary = _binarize(gray, threshold=threshold, invert=invert)

        base = img_path.stem
        # Save the binarized input too (useful for the report)
        save_binary_image(str(output_dir / f"{base}__BIN_thr{threshold}_inv{int(invert)}.png"), binary)

        for k in ksizes:
            for shape in shapes:
                se = make_structuring_element(k, shape)

                for iters in iterations_list:
                    for pad_mode, pad_value in pad_cases:
                        for op_name, op_fn in ops.items():
                            out = op_fn(
                                binary,
                                se,
                                iterations=iters,
                                pad_mode=pad_mode,
                                pad_value=pad_value,
                            )

                            # Report-friendly filename
                            fname = (
                                f"{base}"
                                f"__{op_name.upper()}"
                                f"__k{k}_{shape}"
                                f"__it{iters}"
                                f".png"
                            )
                            save_binary_image(str(output_dir / fname), out)

    print(f"Done. Results saved to: {output_dir.resolve()}")




# if __name__ == "__main__":
#     run_batch_grid()


if __name__ == "__main__":
    args = parse_args()
    run_cli(args)

# Example usage:
# python morphology.py --op erode --in input.png --out out.png --ksize 5 --shape square --iters 2
