"""Voxelization: converts raw point cloud into pillar features."""
import numpy as np
from typing import Tuple, List
from ..config import VoxelConfig


def voxelize(
    points: np.ndarray,  # (N, 4+) — x, y, z, intensity, ...
    cfg: VoxelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        voxels:      (V, max_pts, C)  — padded point features per voxel
        coordinates: (V, 3)           — (z_idx, y_idx, x_idx)
        num_points:  (V,)             — actual point count per voxel
    """
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    vsize = np.array(cfg.voxel_size, dtype=np.float32)
    grid_size = np.array(cfg.grid_size, dtype=np.int32)  # [X, Y, Z]
    max_pts = cfg.max_points_per_voxel
    max_vox = cfg.max_voxels
    C = points.shape[1]

    # Filter to PC range
    mask = (
        (points[:, 0] >= pc_range[0]) & (points[:, 0] < pc_range[3]) &
        (points[:, 1] >= pc_range[1]) & (points[:, 1] < pc_range[4]) &
        (points[:, 2] >= pc_range[2]) & (points[:, 2] < pc_range[5])
    )
    points = points[mask]

    # Compute voxel indices for each point
    coords_float = (points[:, :3] - pc_range[:3]) / vsize  # (N, 3) in [X, Y, Z]
    coords_int = coords_float.astype(np.int32)
    coords_int = np.clip(coords_int, 0, grid_size - 1)

    # Encode (x, y, z) -> single integer key for fast grouping
    # Store as (z, y, x) for later indexing
    z_idx = coords_int[:, 2]
    y_idx = coords_int[:, 1]
    x_idx = coords_int[:, 0]
    voxel_key = z_idx * (grid_size[1] * grid_size[0]) + y_idx * grid_size[0] + x_idx

    # Sort by key, then group
    sort_order = np.argsort(voxel_key)
    voxel_key = voxel_key[sort_order]
    points_sorted = points[sort_order]
    z_sorted = z_idx[sort_order]
    y_sorted = y_idx[sort_order]
    x_sorted = x_idx[sort_order]

    unique_keys, first_occ, counts = np.unique(voxel_key, return_index=True, return_counts=True)

    # Limit voxel count
    if len(unique_keys) > max_vox:
        # Keep voxels with most points first
        keep = np.argsort(-counts)[:max_vox]
        unique_keys = unique_keys[keep]
        first_occ = first_occ[keep]
        counts = counts[keep]

    V = len(unique_keys)
    voxels = np.zeros((V, max_pts, C), dtype=np.float32)
    coordinates = np.zeros((V, 3), dtype=np.int32)
    num_points = np.zeros(V, dtype=np.int32)

    for i, (fi, cnt) in enumerate(zip(first_occ, counts)):
        n = min(int(cnt), max_pts)
        voxels[i, :n] = points_sorted[fi: fi + n]
        coordinates[i] = [z_sorted[fi], y_sorted[fi], x_sorted[fi]]
        num_points[i] = n

    return voxels, coordinates, num_points


def compute_pillar_features(
    voxels: np.ndarray,     # (V, max_pts, C)
    coordinates: np.ndarray,  # (V, 3) z,y,x
    num_points: np.ndarray,   # (V,)
    cfg: VoxelConfig,
) -> np.ndarray:
    """
    Augment each point with the 5 extra features from the paper (D=9 total):
      xp, yp  — offset from pillar x,y centre
      xc, yc, zc — offset from arithmetic mean of all points in the pillar
    Returns (V, max_pts, C+5) features.
    """
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    vsize = np.array(cfg.voxel_size, dtype=np.float32)

    V, P, C = voxels.shape
    # Pillar center (x, y) from coordinate
    pillar_cx = (coordinates[:, 2].astype(np.float32) + 0.5) * vsize[0] + pc_range[0]
    pillar_cy = (coordinates[:, 1].astype(np.float32) + 0.5) * vsize[1] + pc_range[1]

    # Mean of non-zero points per pillar
    mask = np.zeros((V, P), dtype=bool)
    for i, n in enumerate(num_points):
        mask[i, :n] = True

    # Sum only valid (non-padded) points before dividing
    masked_pts = voxels[:, :, :3] * mask[:, :, None]   # zero out padding
    sum_pts    = masked_pts.sum(axis=1)                 # (V, 3)
    n_safe     = np.maximum(num_points, 1)[:, None]
    mean_pts   = sum_pts / n_safe                       # (V, 3)

    # Offset from center
    dx = voxels[:, :, 0] - pillar_cx[:, None]  # (V, P)
    dy = voxels[:, :, 1] - pillar_cy[:, None]

    # Offset from mean
    fx = voxels[:, :, 0] - mean_pts[:, 0:1]
    fy = voxels[:, :, 1] - mean_pts[:, 1:2]
    fz = voxels[:, :, 2] - mean_pts[:, 2:3]

    augmented = np.concatenate([
        voxels,
        dx[:, :, None], dy[:, :, None],
        fx[:, :, None], fy[:, :, None], fz[:, :, None],
    ], axis=2)  # (V, P, C+5)

    # Zero out padded positions
    augmented[~mask] = 0.0
    return augmented.astype(np.float32)
