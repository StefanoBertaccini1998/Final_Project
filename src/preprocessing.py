"""
preprocessing.py — Data Acquisition & Preprocessing Stage
Pipeline Stage 1: handles loading, resizing, normalization, and augmentation.
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """Load and resize an image from disk.

    Args:
        path: Path to the image file.
        target_size: (width, height) tuple for resizing.

    Returns:
        Preprocessed image as numpy array (H, W, C), uint8.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1].

    Args:
        image: Input image as uint8 numpy array.

    Returns:
        Normalized image as float32 numpy array.
    """
    return image.astype(np.float32) / 255.0


def denoise(image: np.ndarray, method: str = "gaussian", ksize: int = 3) -> np.ndarray:
    """Apply noise reduction to an image.

    Args:
        image: Input image.
        method: 'gaussian' or 'median'.
        ksize: Kernel size (must be odd).

    Returns:
        Denoised image.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def augment(image: np.ndarray, flip_h: bool = True, flip_v: bool = False) -> list:
    """Generate augmented versions of an image.

    Args:
        image: Input image.
        flip_h: Apply horizontal flip.
        flip_v: Apply vertical flip.

    Returns:
        List of augmented images including the original.
    """
    augmented = [image]
    if flip_h:
        augmented.append(cv2.flip(image, 1))
    if flip_v:
        augmented.append(cv2.flip(image, 0))
    return augmented
