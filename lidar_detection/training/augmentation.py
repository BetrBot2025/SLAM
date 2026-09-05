"""
Lightweight data augmentation for LiDAR point clouds + 3D boxes.
All transforms return new copies — originals are never modified.

Includes GT-sampling (copy-paste) augmentation:
  - Build the database once with:  python scripts/build_gt_database.py
  - Pass gt_database= to augment() to enable it.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def _copy_boxes(gt_boxes: List[Dict]) -> List[Dict]:
    return [
        {
            "center":     list(b["center"]),
            "size":       list(b["size"]),
            "yaw":        b["yaw"],
            "class_id":   b["class_id"],
            "class_name": b.get("class_name", ""),
        }
        for b in gt_boxes
    ]


def random_flip(
    points: np.ndarray,
    gt_boxes: List[Dict],
    prob: float = 0.5,
) -> Tuple[np.ndarray, List[Dict]]:
    """Flip along y-axis (left-right in BEV)."""
    if np.random.random() >= prob:
        return points, gt_boxes
    gt_boxes = _copy_boxes(gt_boxes)
    pts = points.copy()
    pts[:, 1] = -pts[:, 1]
    for b in gt_boxes:
        b["center"][1] = -b["center"][1]
        b["yaw"]       = -b["yaw"]
    return pts, gt_boxes


def global_rotation(
    points: np.ndarray,
    gt_boxes: List[Dict],
    max_angle: float = np.pi / 4,
) -> Tuple[np.ndarray, List[Dict]]:
    """Rotate entire scene around the Z-axis by a uniform random angle."""
    angle = np.random.uniform(-max_angle, max_angle)
    c, s  = np.cos(angle), np.sin(angle)
    gt_boxes = _copy_boxes(gt_boxes)
    pts = points.copy()
    x, y      = points[:, 0].copy(), points[:, 1].copy()
    pts[:, 0] = c * x - s * y
    pts[:, 1] = s * x + c * y
    for b in gt_boxes:
        cx, cy          = b["center"][0], b["center"][1]
        b["center"][0]  = c * cx - s * cy
        b["center"][1]  = s * cx + c * cy
        b["yaw"]       += angle
    return pts, gt_boxes


def global_scale(
    points: np.ndarray,
    gt_boxes: List[Dict],
    min_scale: float = 0.95,
    max_scale: float = 1.05,
) -> Tuple[np.ndarray, List[Dict]]:
    """Uniform scale of entire scene (xyz + box dims)."""
    scale = np.random.uniform(min_scale, max_scale)
    gt_boxes = _copy_boxes(gt_boxes)
    pts = points.copy()
    pts[:, :3] *= scale
    for b in gt_boxes:
        b["center"] = [v * scale for v in b["center"]]
        b["size"]   = [v * scale for v in b["size"]]
    return pts, gt_boxes


# ── GT-sampling (copy-paste) augmentation ─────────────────────────────────────

def _bev_iou_1v1(b1: List[float], b2: List[float]) -> float:
    """Axis-aligned BEV IoU between two boxes [cx,cy,l,w] each."""
    cx1, cy1, l1, w1 = b1
    cx2, cy2, l2, w2 = b2
    ix = max(0.0, min(cx1+l1/2, cx2+l2/2) - max(cx1-l1/2, cx2-l2/2))
    iy = max(0.0, min(cy1+w1/2, cy2+w2/2) - max(cy1-w1/2, cy2-w2/2))
    inter = ix * iy
    union = l1*w1 + l2*w2 - inter + 1e-7
    return inter / union


def _points_in_box(points: np.ndarray, box: Dict) -> np.ndarray:
    """Return boolean mask: True for points inside the rotated 3D box."""
    cx, cy, cz = box["center"]
    l, w, h    = box["size"]
    yaw        = box["yaw"]

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dz = points[:, 2] - cz

    c, s  = np.cos(-yaw), np.sin(-yaw)
    lx    = c * dx - s * dy
    ly    = s * dx + c * dy

    return (
        (np.abs(lx) <= l / 2) &
        (np.abs(ly) <= w / 2) &
        (np.abs(dz) <= h / 2)
    )


def gt_sampling(
    points:     np.ndarray,
    gt_boxes:   List[Dict],
    gt_database: Dict[str, List[Dict]],   # {class_name: [{box, rel_pts}, ...]}
    max_per_class: Dict[str, int] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Copy-paste augmentation: randomly sample GT boxes from the pre-built
    database and paste them into the current scene.

    gt_database[class_name] = list of entries, each:
        {"box": {...gt box dict...}, "rel_pts": np.ndarray (N,4)}
        where rel_pts are points in the box-local frame (box center at origin,
        yaw-aligned).

    Boxes are rejected if they collide (BEV IoU > 0) with any existing GT.
    """
    if max_per_class is None:
        max_per_class = {"Car": 15, "Pedestrian": 10, "Cyclist": 10}

    # Existing GT footprints for collision checking
    existing = [
        [b["center"][0], b["center"][1], b["size"][0], b["size"][1]]
        for b in gt_boxes
    ]

    added_pts   = []
    added_boxes = _copy_boxes(gt_boxes)

    for cls_name, db_entries in gt_database.items():
        if not db_entries:
            continue
        n_sample = max_per_class.get(cls_name, 10)
        sampled  = np.random.choice(len(db_entries), size=min(n_sample * 3, len(db_entries)),
                                    replace=False)

        placed = 0
        for i in sampled:
            if placed >= n_sample:
                break
            entry = db_entries[i]
            box   = entry["box"]

            # Collision check against all existing and already-placed boxes
            footprint = [box["center"][0], box["center"][1], box["size"][0], box["size"][1]]
            collision = any(_bev_iou_1v1(footprint, e) > 0 for e in existing)
            if collision:
                continue

            # Transform points from box-local frame → LiDAR frame
            rel_pts = entry["rel_pts"].copy()    # (N, 4) in box-local
            cx, cy, cz = box["center"]
            yaw        = box["yaw"]
            c_y, s_y   = np.cos(yaw), np.sin(yaw)
            lx, ly     = rel_pts[:, 0], rel_pts[:, 1]
            world_pts  = rel_pts.copy()
            world_pts[:, 0] = c_y * lx - s_y * ly + cx
            world_pts[:, 1] = s_y * lx + c_y * ly + cy
            world_pts[:, 2] = rel_pts[:, 2] + cz

            added_pts.append(world_pts)
            added_boxes.append({**box, "class_name": cls_name})
            existing.append(footprint)
            placed += 1

    if added_pts:
        points = np.concatenate([points] + added_pts, axis=0)

    return points, added_boxes


# ── Full augmentation pipeline ─────────────────────────────────────────────────

def augment(
    points:      np.ndarray,
    gt_boxes:    List[Dict],
    gt_database: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Full pipeline:
      [gt_sampling →] random flip → global rotation → global scale
    """
    if gt_database:
        points, gt_boxes = gt_sampling(points, gt_boxes, gt_database)
    points, gt_boxes = random_flip(points, gt_boxes)
    points, gt_boxes = global_rotation(points, gt_boxes)
    points, gt_boxes = global_scale(points, gt_boxes)
    return points, gt_boxes
