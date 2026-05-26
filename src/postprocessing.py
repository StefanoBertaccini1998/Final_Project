"""
postprocessing.py — Pipeline Stage 4: Post-processing
=======================================================
Responsibility: refine model output and apply spatial constraints.

Two distinct concerns are handled here:

1. ROI Zone Check — spatial validation
   The camera is fixed above a workstation. The part must appear within a
   defined Region of Interest (ROI). A part outside the ROI is flagged as
   "out of position" regardless of the defect classifier's output.
   This is a hard gate: out-of-position parts must be rejected before
   the defect classifier runs, because a misaligned part produces a feature
   distribution the classifier was never trained on.

2. Mask refinement — morphological post-processing
   For future segmentation extension: clean up binary masks using standard
   morphological operations (close gaps, remove noise).

3. NMS — for detection extension
   Deduplication of overlapping bounding boxes, included for completeness
   if the pipeline is extended to object detection in the future.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# ROI Zone Check
# ---------------------------------------------------------------------------

# Default ROI margin: the part must occupy the central 60% of the image.
# Margins of 20% on each side exclude camera edges where parts are never
# expected to appear in a correctly positioned workstation.
DEFAULT_ROI_MARGIN: float = 0.20

# Minimum overlap ratio between part bounding box and ROI to pass the check.
# 0.80 means at least 80% of the part must be inside the ROI.
MIN_OVERLAP_RATIO: float = 0.80


def define_roi(
    image_shape: Tuple[int, int],
    margin_pct: float = DEFAULT_ROI_MARGIN,
) -> Tuple[int, int, int, int]:
    """Define a rectangular ROI as the central region of the image.

    The ROI excludes a border of `margin_pct` on each side. This represents
    the expected workstation zone where parts are placed by the operator.

    Args:
        image_shape: (height, width) of the image.
        margin_pct: Fraction of image size to exclude as border (each side).

    Returns:
        (x1, y1, x2, y2) pixel coordinates of the ROI rectangle.
    """
    h, w = image_shape[:2]
    x1 = int(w * margin_pct)
    y1 = int(h * margin_pct)
    x2 = int(w * (1 - margin_pct))
    y2 = int(h * (1 - margin_pct))
    return x1, y1, x2, y2


def find_part_bbox(
    image: np.ndarray,
    bg_threshold: int = 30,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate the part in the image by thresholding the dark background.

    MVTec images have a near-black background. Thresholding isolates the part
    as a bright foreground region. The bounding box of the largest contour is
    returned as the part's location.

    Why threshold instead of using the classifier's output?
      The ROI check is a spatial pre-filter that runs independently of the
      defect classifier. It only needs to know WHERE the part is, not
      whether it is defective.

    Args:
        image: uint8 RGB image (H, W, 3) or float32 [0,1].
        bg_threshold: Pixel intensity below this value is considered background.
                      30/255 works for MVTec's near-black background.

    Returns:
        (x1, y1, x2, y2) bounding box of the part, or None if not found.
    """
    if image.dtype != np.uint8:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.copy()

    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, bg_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, x + w, y + h


def check_roi(
    part_bbox: Tuple[int, int, int, int],
    roi: Tuple[int, int, int, int],
    min_overlap: float = MIN_OVERLAP_RATIO,
) -> Tuple[bool, float]:
    """Check whether a part bounding box is sufficiently inside the ROI.

    Computes the overlap ratio: intersection area / part area.
    A part that extends slightly outside the ROI may still pass if
    MIN_OVERLAP_RATIO of it is inside.

    Args:
        part_bbox: (x1, y1, x2, y2) of the detected part.
        roi: (x1, y1, x2, y2) of the ROI rectangle.
        min_overlap: Minimum fraction of part that must be inside ROI.

    Returns:
        Tuple of (passed: bool, overlap_ratio: float).
    """
    px1, py1, px2, py2 = part_bbox
    rx1, ry1, rx2, ry2 = roi

    inter_x1 = max(px1, rx1)
    inter_y1 = max(py1, ry1)
    inter_x2 = min(px2, rx2)
    inter_y2 = min(py2, ry2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    part_area = (px2 - px1) * (py2 - py1)
    if part_area == 0:
        return False, 0.0

    overlap = inter_area / part_area
    return overlap >= min_overlap, round(overlap, 4)


def draw_overlay(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    part_bbox: Optional[Tuple[int, int, int, int]],
    in_roi: bool,
    is_defective: bool,
    defect_prob: float = 0.0,
) -> np.ndarray:
    """Draw ROI, part bounding box, and classification result on the image.

    Color coding:
      - ROI rectangle: yellow
      - Part bbox — in ROI + good:      green
      - Part bbox — in ROI + defective: red
      - Part bbox — out of ROI:         magenta

    Args:
        image: float32 [0,1] or uint8 RGB image.
        roi: ROI rectangle (x1, y1, x2, y2).
        part_bbox: Part bounding box, or None if not detected.
        in_roi: Whether the part passed the ROI check.
        is_defective: Whether the defect classifier flagged the part.
        defect_prob: Defect probability for display (optional).

    Returns:
        uint8 RGB image with overlay drawn.
    """
    if image.dtype != np.uint8:
        vis = (image * 255).astype(np.uint8).copy()
    else:
        vis = image.copy()

    # Draw ROI in yellow
    rx1, ry1, rx2, ry2 = roi
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 215, 0), 2)
    cv2.putText(vis, 'ROI', (rx1 + 4, ry1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)

    if part_bbox is not None:
        px1, py1, px2, py2 = part_bbox
        if not in_roi:
            color = (255, 0, 255)   # magenta — out of position
            label = 'OUT OF POSITION'
        elif is_defective:
            color = (255, 60, 60)   # red — defective
            label = f'DEFECTIVE ({defect_prob:.0%})'
        else:
            color = (60, 200, 60)   # green — good
            label = 'GOOD'

        cv2.rectangle(vis, (px1, py1), (px2, py2), color, 2)
        cv2.putText(vis, label, (px1, py1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis


# ---------------------------------------------------------------------------
# Mask refinement (for future segmentation extension)
# ---------------------------------------------------------------------------

def refine_mask(
    binary_mask: np.ndarray,
    operation: str = "close",
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply morphological operations to refine a segmentation mask.

    'close' (dilate then erode) fills small holes inside detected regions.
    'open'  (erode then dilate) removes small isolated noise blobs.

    Args:
        binary_mask: uint8 binary mask (0/255).
        operation: 'open', 'close', 'erode', or 'dilate'.
        kernel_size: Size of the elliptical structuring element.

    Returns:
        Refined uint8 binary mask.

    Raises:
        ValueError: If operation is not recognized.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ops = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
           "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE}
    if operation not in ops:
        raise ValueError(f"Unknown operation '{operation}'. Choose from: {list(ops)}")
    return cv2.morphologyEx(binary_mask, ops[operation], kernel)


# ---------------------------------------------------------------------------
# NMS (for future detection extension)
# ---------------------------------------------------------------------------

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> list:
    """Non-Maximum Suppression: remove duplicate overlapping bounding boxes.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2] boxes.
        scores: (N,) confidence scores.
        iou_threshold: Boxes with IoU above this are suppressed.

    Returns:
        List of indices of kept boxes, sorted by descending score.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])
        inter_w  = np.maximum(0.0, inter_x2 - inter_x1 + 1)
        inter_h  = np.maximum(0.0, inter_y2 - inter_y1 + 1)
        iou      = (inter_w * inter_h) / (areas[i] + areas[order[1:]] - inter_w * inter_h)
        order    = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep
