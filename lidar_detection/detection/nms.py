"""3D / BEV Non-Maximum Suppression."""
import numpy as np


def iou_bev(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute BEV IoU between one box and N boxes (axis-aligned approximation via
    min-max on rotated bounding rect — fast and sufficient for NMS).

    Args:
        box:   (7,) — x,y,z,l,w,h,yaw
        boxes: (N,7)
    Returns:
        iou:   (N,) float32
    """
    def _bbox(b):
        x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
        cos_a, sin_a = np.cos(yaw), np.sin(yaw)
        hl, hw = l / 2, w / 2
        lx = hl * abs(cos_a) + hw * abs(sin_a)
        ly = hl * abs(sin_a) + hw * abs(cos_a)
        return x - lx, y - ly, x + lx, y + ly

    ax1, ay1, ax2, ay2 = _bbox(box)
    area_a = (ax2 - ax1) * (ay2 - ay1)

    # Vectorised for boxes
    x, y, l, w, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2, w / 2
    lx = hl * np.abs(cos_a) + hw * np.abs(sin_a)
    ly = hl * np.abs(sin_a) + hw * np.abs(cos_a)
    bx1, by1, bx2, by2 = x - lx, y - ly, x + lx, y + ly
    area_b = (bx2 - bx1) * (by2 - by1)

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    union = area_a + area_b - inter + 1e-7
    return inter / union


def nms_3d(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Args:
        boxes:  (N, 7) — x,y,z,l,w,h,yaw
        scores: (N,)
        iou_threshold: float
    Returns:
        keep: indices of surviving boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        iou = iou_bev(boxes[i], boxes[rest])
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)
