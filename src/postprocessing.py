"""
postprocessing.py — Post-processing Stage
Pipeline Stage 4: refines raw model output (NMS for detection, morphological ops for segmentation).
"""

import cv2
import numpy as np


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                         iou_threshold: float = 0.5) -> list:
    """Apply Non-Maximum Suppression (NMS) to detection bounding boxes.

    Args:
        boxes: Array of shape (N, 4) with [x1, y1, x2, y2] format.
        scores: Array of shape (N,) with confidence scores.
        iou_threshold: IoU threshold above which overlapping boxes are suppressed.

    Returns:
        List of indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
        inter_area = inter_w * inter_h

        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        idx = np.where(iou <= iou_threshold)[0]
        order = order[idx + 1]

    return keep


def refine_mask(binary_mask: np.ndarray, operation: str = "close",
                kernel_size: int = 5) -> np.ndarray:
    """Apply morphological operations to refine a segmentation mask.

    Args:
        binary_mask: Binary mask (0/255, uint8).
        operation: 'open', 'close', 'erode', or 'dilate'.
        kernel_size: Size of the structuring element.

    Returns:
        Refined binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE
    }
    if operation not in ops:
        raise ValueError(f"Unknown operation: {operation}")

    return cv2.morphologyEx(binary_mask, ops[operation], kernel)
