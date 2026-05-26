"""
features.py — Pipeline Stage 2: Feature Extraction
====================================================
Responsibility: transform preprocessed images into compact numeric vectors
suitable for classical ML classifiers (SVM, Random Forest).

Design principles:
  - Each extractor is a pure function: same input → same output, no side effects.
  - Parameters are named constants with documented rationale, not magic numbers.
  - Only HOG is used in the primary pipeline. SIFT is implemented for reference
    but excluded from training because it produces variable-length descriptors
    (number of keypoints varies per image), which requires an additional
    aggregation step (e.g., Bag of Words) to use with SVM.

Why HOG for industrial defect detection:
  HOG captures the distribution of edge directions across local image regions.
  Surface defects (scratches, dents, bends) alter the local gradient structure
  in predictable ways. HOG is robust to small illumination changes because it
  works on gradient directions, not absolute intensities.
"""

import cv2
import numpy as np
from skimage.feature import hog


# ---------------------------------------------------------------------------
# HOG constants — each value is documented with the reason it was chosen
# ---------------------------------------------------------------------------

# 9 orientation bins → 40° resolution per bin.
# Standard value in the HOG literature (Dalal & Triggs, 2005).
# Fewer bins (e.g., 6) lose angular precision; more bins (e.g., 18) add noise.
HOG_ORIENTATIONS: int = 9

# 8×8 pixels per cell balances spatial resolution vs descriptor length.
# On 224×224 images: 224/8 = 28 cells per side → 28×28 = 784 cells.
# A 4×4 cell would quadruple the vector length without meaningful gain on
# defects that span multiple cells.
HOG_PIXELS_PER_CELL: tuple = (8, 8)

# 2×2 cells per block. Each block is L2-normalized independently, which
# provides local contrast normalization — critical for images where the
# part occupies only the center of the frame and the border is dark.
HOG_CELLS_PER_BLOCK: tuple = (2, 2)


# ---------------------------------------------------------------------------
# HOG extraction
# ---------------------------------------------------------------------------

def extract_hog(
    image: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: tuple = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple = HOG_CELLS_PER_BLOCK,
) -> np.ndarray:
    """Extract a HOG feature vector from a preprocessed image.

    Converts to grayscale before HOG computation. Color channels are not used
    because HOG on grayscale captures the structural gradient information that
    distinguishes defects, while color adds noise for metallic surfaces where
    the dominant cue is texture, not hue.

    Args:
        image: float32 RGB array (H, W, 3) in [0, 1] — output of preprocess().
        orientations: Number of gradient orientation bins.
        pixels_per_cell: Size (h, w) of each HOG cell in pixels.
        cells_per_block: Number of (row, col) cells per normalization block.

    Returns:
        1D float64 HOG feature vector. Length depends on image size and params:
        for 224×224 with defaults → 34596-dimensional vector.
    """
    # HOG operates on grayscale. Convert from float32 [0,1] → uint8 [0,255]
    # because skimage HOG internally normalizes by pixel range.
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return features


# ---------------------------------------------------------------------------
# SIFT extraction (reference implementation — not used in primary pipeline)
# ---------------------------------------------------------------------------

def extract_sift_keypoints(
    image: np.ndarray,
    n_features: int = 128,
):
    """Extract SIFT keypoints and descriptors.

    Not used in the primary SVM pipeline because SIFT produces a variable
    number of keypoints per image. To use SIFT with SVM, descriptors must
    first be aggregated into a fixed-length vector (e.g., Bag of Visual Words),
    which adds significant complexity for marginal gain over HOG on this task.

    Args:
        image: uint8 or float32 RGB array.
        n_features: Maximum number of keypoints to retain (0 = unlimited).

    Returns:
        Tuple of (keypoints, descriptors). descriptors is None if no keypoints
        are detected.
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    img_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
