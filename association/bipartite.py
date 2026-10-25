"""WP4: global greedy 2D-3D bipartite association (Algorithm 1 Stage 1,
Eqs. 1-3 of the DSP-SLAM++ paper).

Projection (3D points -> image pixels, and fisheye mask rectification) is
deliberately NOT done here -- that's WP2/WP3 geometry (slam_frontend/rgbl.py)
and depends on calibration. This module only implements the matching
algorithm, taking already-projected pixel coordinates per 3D detection.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

from detection2d.interface import Detection2D


@dataclass
class Detection3D:
    class_id: int
    class_name: str
    score: float
    box: List[float]                # [x,y,z,l,w,h,yaw], object frame = sensor frame
    projected_points: np.ndarray     # (N, 2) pixel coords of this object's LiDAR
                                      # points, in the same rectified image space
                                      # as Detection2D.mask


def _inlier_ratio(points_px: np.ndarray, mask: np.ndarray) -> float:
    """|P3D,i cap M2D,j| / |P3D,i| -- fraction of projected points falling
    inside the mask boundary (Sec. IV-A)."""
    n = points_px.shape[0]
    if n == 0:
        return 0.0
    H, W = mask.shape
    xi = np.round(points_px[:, 0]).astype(int)
    yi = np.round(points_px[:, 1]).astype(int)
    valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if not np.any(valid):
        return 0.0
    inside = int(mask[yi[valid], xi[valid]].sum())
    return inside / n


def build_score_matrix(
    det3d: List[Detection3D],
    det2d: List[Detection2D],
) -> np.ndarray:
    """S in R^{n3d x n2d} (Eq. 1). -1 for cross-class pairs; else the inlier
    ratio in [0, 1]. (The paper's S in Z notation is a simplification of this
    ratio-valued matrix -- the text explicitly defines the entries as a
    fraction, which is what's implemented here.)"""
    n3d, n2d = len(det3d), len(det2d)
    S = np.full((n3d, n2d), -1.0, dtype=np.float64)
    for i, d3 in enumerate(det3d):
        for j, d2 in enumerate(det2d):
            if d3.class_id != d2.class_id:
                continue
            S[i, j] = _inlier_ratio(d3.projected_points, d2.mask)
    return S


def global_greedy_assignment(
    S: np.ndarray,
    det3d: List[Detection3D],
    min_ratio: Union[float, Dict[str, float]] = 0.1,
) -> List[Tuple[int, int]]:
    """Iterative arg-max assignment with row/column invalidation until all
    valid scores are exhausted (Eqs. 2-3, Algorithm 1 line 3).

    Args:
        min_ratio: class-specific point-ratio validation threshold; either a
            single float for all classes or a {class_name: float} dict.

    Returns list of (i3d, j2d) index pairs, each used at most once.
    """
    S = S.copy()
    n3d, n2d = S.shape
    if n3d == 0 or n2d == 0:
        return []

    pairs: List[Tuple[int, int]] = []
    for _ in range(min(n3d, n2d)):
        i, j = np.unravel_index(np.argmax(S), S.shape)
        best = S[i, j]
        thr = min_ratio.get(det3d[i].class_name, 0.1) if isinstance(min_ratio, dict) else min_ratio
        if best < thr:
            break
        pairs.append((int(i), int(j)))
        S[i, :] = -1.0
        S[:, j] = -1.0
    return pairs


def _selftest():
    """Runnable directly: python -m association.bipartite"""
    class _M:
        def __init__(self, arr):
            self.mask = arr

    mask_car = np.zeros((10, 10), dtype=bool)
    mask_car[2:6, 2:6] = True
    mask_ped = np.zeros((10, 10), dtype=bool)
    mask_ped[7:9, 7:9] = True

    det2d = [
        Detection2D(mask=mask_car, class_id=0, class_name="Car", score=0.9),
        Detection2D(mask=mask_ped, class_id=3, class_name="Pedestrian", score=0.8),
    ]

    car_pts = np.array([[3, 3], [4, 4], [3, 4], [50, 50]], dtype=float)   # 3/4 inside mask_car
    ped_pts = np.array([[7, 7], [8, 8]], dtype=float)                     # 2/2 inside mask_ped

    det3d = [
        Detection3D(class_id=0, class_name="Car", score=0.85, box=[0]*7, projected_points=car_pts),
        Detection3D(class_id=3, class_name="Pedestrian", score=0.7, box=[0]*7, projected_points=ped_pts),
    ]

    S = build_score_matrix(det3d, det2d)
    assert S[0, 1] == -1.0 and S[1, 0] == -1.0, "cross-class pairs must be invalidated"
    assert abs(S[0, 0] - 0.75) < 1e-9, f"expected 0.75 car inlier ratio, got {S[0,0]}"
    assert abs(S[1, 1] - 1.0) < 1e-9, f"expected 1.0 pedestrian inlier ratio, got {S[1,1]}"

    pairs = global_greedy_assignment(S, det3d, min_ratio=0.5)
    assert set(pairs) == {(0, 0), (1, 1)}, f"expected both matched, got {pairs}"

    print("association.bipartite self-test OK:", pairs)


if __name__ == "__main__":
    _selftest()
