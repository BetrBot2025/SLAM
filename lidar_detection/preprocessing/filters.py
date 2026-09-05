"""Point cloud filters: ground removal and ROI crop."""
import numpy as np


def remove_ground_ransac(
    points: np.ndarray,
    z_threshold: float = -1.5,
    ransac_threshold: float = 0.2,
    n_iter: int = 100,
    min_inliers: int = 500,
) -> np.ndarray:
    """Remove ground plane using RANSAC on low points."""
    low_mask = points[:, 2] < z_threshold + 0.5
    low_pts = points[low_mask]

    if len(low_pts) < 50:
        return points[points[:, 2] > z_threshold]

    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(low_pts), 3, replace=False)
        p1, p2, p3 = low_pts[idx, :3]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -np.dot(normal, p1)
        dist = np.abs(low_pts[:, :3] @ normal + d)
        inliers = dist < ransac_threshold
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return points[points[:, 2] > z_threshold]

    # Fit final plane on all inliers
    ground_pts = low_pts[best_inliers, :3]
    centroid = ground_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(ground_pts - centroid)
    normal = Vt[-1]
    d = -np.dot(normal, centroid)

    # Remove all points close to the fitted plane AND below a height threshold
    dist_all = np.abs(points[:, :3] @ normal + d)
    ground_mask = (dist_all < ransac_threshold) & (points[:, 2] < z_threshold + 0.5)
    return points[~ground_mask]


def crop_to_range(points: np.ndarray, pc_range: list) -> np.ndarray:
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    mask = (
        (points[:, 0] >= xmin) & (points[:, 0] < xmax) &
        (points[:, 1] >= ymin) & (points[:, 1] < ymax) &
        (points[:, 2] >= zmin) & (points[:, 2] < zmax)
    )
    return points[mask]
