"""
preprocessing.py — Pipeline Stage 1: Data Acquisition & Preprocessing
======================================================================
Responsibility: transform raw input images into a clean, consistent
representation suitable for feature extraction and model training.

Design principles applied here:
  - Single Responsibility: each function does exactly one thing.
  - Explicit over implicit: all defaults are documented with the reason
    they were chosen, not just their value.
  - Reproducibility: no random state is hidden inside functions;
    callers control augmentation explicitly.

Why preprocessing matters for this project:
  Industrial camera images can vary in lighting conditions, sensor noise,
  and distance. Normalizing these factors before feature extraction
  ensures the model learns defect patterns, not illumination artifacts.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 224x224 is the standard input size for ImageNet-pretrained models
# (ResNet, EfficientNet). Using it here ensures compatibility with
# transfer learning without any architectural changes.
DEFAULT_SIZE: tuple = (224, 224)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str, target_size: tuple = DEFAULT_SIZE) -> np.ndarray:
    """Load an image from disk, convert to RGB, and resize.

    OpenCV reads images in BGR channel order by default. We convert to RGB
    immediately so that all downstream code works in a single color space
    and matches the convention expected by torchvision transforms.

    Args:
        path: Absolute or relative path to the image file.
        target_size: (width, height) in pixels. Defaults to 224x224.

    Returns:
        uint8 numpy array of shape (H, W, 3) in RGB order.

    Raises:
        FileNotFoundError: If the file does not exist or cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. "
            "Check that the path exists and the file is a valid image."
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0].

    Why [0, 1] and not zero-mean/unit-variance?
      For visualization and for the classical HOG pipeline we keep [0, 1].
      The ImageNet mean/std normalization required by pretrained models is
      applied separately inside the PyTorch DataLoader transform, keeping
      this function reusable for both pipelines.

    Args:
        image: uint8 numpy array (H, W, C) or (H, W).

    Returns:
        float32 numpy array with the same shape, values in [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Noise reduction
# ---------------------------------------------------------------------------

def denoise(image: np.ndarray, method: str = "gaussian", ksize: int = 3) -> np.ndarray:
    """Reduce high-frequency noise while preserving edge information.

    Method selection rationale:
      - "gaussian": smooths uniform noise (e.g., sensor noise). Fast and
        standard. Kernel size 3 is the minimum that has a visible effect
        without blurring meaningful texture.
      - "median": better at removing salt-and-pepper noise (isolated bright/
        dark pixels) because it replaces each pixel with the median of its
        neighborhood rather than a weighted average.

    For MVTec industrial images, Gaussian with ksize=3 is preferred because
    the dominant noise source is sensor noise, not impulse noise.

    Args:
        image: Input image (uint8 or float32).
        method: 'gaussian' or 'median'.
        ksize: Kernel size. Must be a positive odd integer.

    Returns:
        Denoised image with the same dtype and shape as input.

    Raises:
        ValueError: If method is not recognized or ksize is even.
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize must be odd, got {ksize}.")
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
    elif method == "median":
        # medianBlur requires uint8 input
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return cv2.medianBlur(image, ksize)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'gaussian' or 'median'.")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(
    image: np.ndarray,
    flip_h: bool = True,
    flip_v: bool = False,
) -> list:
    """Generate augmented copies of an image for training data expansion.

    Why augmentation is necessary here:
      The MVTec dataset has a limited number of defective samples per
      category (typically 60–120 images). Without augmentation, the model
      sees too few examples of defects and tends to overfit or predict
      the majority class (good).

    Augmentation strategy rationale:
      - Horizontal flip (flip_h=True): valid for industrial parts that have
        no inherent left/right orientation (nuts, screws, tiles). Doubles
        effective dataset size at zero cost.
      - Vertical flip (flip_v=False by default): disabled because some parts
        (e.g., screws) have a clear top/bottom orientation. Enable only if
        the chosen category is orientation-invariant.
      - Rotation and color jitter are handled in the PyTorch DataLoader
        pipeline (see models/deep.py) where GPU acceleration is available.

    Args:
        image: Input image as numpy array.
        flip_h: If True, include a horizontally flipped copy.
        flip_v: If True, include a vertically flipped copy.

    Returns:
        List of numpy arrays: always includes the original as the first
        element, followed by augmented variants.
    """
    augmented = [image]  # original always included

    if flip_h:
        # cv2.flip(img, 1) flips around the vertical axis (left ↔ right)
        augmented.append(cv2.flip(image, 1))

    if flip_v:
        # cv2.flip(img, 0) flips around the horizontal axis (top ↔ bottom)
        augmented.append(cv2.flip(image, 0))

    return augmented
