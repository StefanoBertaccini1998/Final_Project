"""
features.py — Feature Engineering/Representation Stage
Pipeline Stage 2: handcrafted features (HOG, SIFT) and deep feature extraction.
"""

import cv2
import numpy as np
from skimage.feature import hog


def extract_hog(image: np.ndarray, orientations: int = 9,
                pixels_per_cell: tuple = (8, 8),
                cells_per_block: tuple = (2, 2)) -> np.ndarray:
    """Extract HOG (Histogram of Oriented Gradients) features.

    Args:
        image: Grayscale or RGB input image.
        orientations: Number of gradient orientation bins.
        pixels_per_cell: Size of each cell.
        cells_per_block: Number of cells per block for normalization.

    Returns:
        1D HOG feature vector.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)

    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True
    )
    return features


def extract_sift_keypoints(image: np.ndarray, n_features: int = 128):
    """Extract SIFT keypoints and descriptors.

    Args:
        image: Input image (uint8).
        n_features: Max number of features to retain.

    Returns:
        Tuple of (keypoints, descriptors).
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
