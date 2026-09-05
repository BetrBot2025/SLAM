"""Synthetic LiDAR point cloud generator for testing."""
import numpy as np
from typing import List, Tuple, Dict


def _rotation_matrix_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _generate_object_points(
    center: np.ndarray,
    size: np.ndarray,  # [l, w, h]
    yaw: float,
    n_points: int,
    noise: float = 0.02,
) -> np.ndarray:
    """Generate surface points for a 3D box."""
    l, w, h = size
    pts = []
    pts_per_face = n_points // 6

    # Six faces of the box
    faces = [
        (np.array([l / 2, 0, 0]), np.array([0, w, h])),
        (np.array([-l / 2, 0, 0]), np.array([0, w, h])),
        (np.array([0, w / 2, 0]), np.array([l, 0, h])),
        (np.array([0, -w / 2, 0]), np.array([l, 0, h])),
        (np.array([0, 0, h / 2]), np.array([l, w, 0])),
        (np.array([0, 0, -h / 2]), np.array([l, w, 0])),
    ]
    for face_center, extents in faces:
        n = pts_per_face
        offsets = (np.random.rand(n, 3) - 0.5) * extents
        face_pts = face_center + offsets
        pts.append(face_pts)

    pts = np.concatenate(pts, axis=0)
    rot = _rotation_matrix_z(yaw)
    pts = (rot @ pts.T).T + center
    pts += np.random.randn(*pts.shape).astype(np.float32) * noise
    return pts.astype(np.float32)


def generate_scene(
    n_cars: int = 5,
    n_pedestrians: int = 3,
    n_cyclists: int = 2,
    n_background: int = 10000,
    pc_range: List[float] = None,
    seed: int = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Returns:
        points: (N, 4) float32 — x, y, z, intensity
        gt_boxes: list of dicts with keys: center, size, yaw, class_id, class_name
    """
    if seed is not None:
        np.random.seed(seed)
    if pc_range is None:
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]

    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    gt_boxes = []
    object_points = []

    object_specs = [
        ("Car", 0, n_cars, [3.9, 1.6, 1.56]),
        ("Pedestrian", 1, n_pedestrians, [0.8, 0.6, 1.73]),
        ("Cyclist", 2, n_cyclists, [1.76, 0.6, 1.73]),
    ]

    placed_centers = []

    for class_name, class_id, count, base_size in object_specs:
        for _ in range(count):
            # Attempt placement without major overlap
            for attempt in range(50):
                cx = np.random.uniform(xmin + 5, xmax - 5)
                cy = np.random.uniform(ymin + 3, ymax - 3)
                cz = zmin + base_size[2] / 2
                center = np.array([cx, cy, cz], dtype=np.float32)

                # Simple overlap check
                ok = True
                for pc in placed_centers:
                    if np.linalg.norm(center[:2] - pc[:2]) < 3.0:
                        ok = False
                        break
                if ok:
                    break

            yaw = np.random.uniform(-np.pi, np.pi)
            size = np.array(base_size, dtype=np.float32)
            size += np.random.randn(3).astype(np.float32) * 0.05

            pts = _generate_object_points(center, size, yaw, n_points=120)
            intensity = np.random.uniform(0.3, 0.9, (len(pts), 1)).astype(np.float32)
            pts_with_int = np.concatenate([pts, intensity], axis=1)
            object_points.append(pts_with_int)

            gt_boxes.append({
                "center": center.tolist(),
                "size": size.tolist(),
                "yaw": float(yaw),
                "class_id": class_id,
                "class_name": class_name,
            })
            placed_centers.append(center)

    # Background ground plane + random clutter
    bg_x = np.random.uniform(xmin, xmax, n_background).astype(np.float32)
    bg_y = np.random.uniform(ymin, ymax, n_background).astype(np.float32)
    bg_z = np.random.uniform(zmin, zmin + 0.3, n_background).astype(np.float32)
    bg_i = np.random.uniform(0.0, 0.2, (n_background, 1)).astype(np.float32)
    bg = np.stack([bg_x, bg_y, bg_z], axis=1)
    bg = np.concatenate([bg, bg_i], axis=1)

    all_points = np.concatenate([bg] + object_points, axis=0)
    # Shuffle
    idx = np.random.permutation(len(all_points))
    all_points = all_points[idx]

    return all_points, gt_boxes
