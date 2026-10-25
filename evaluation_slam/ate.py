"""WP9: Absolute Trajectory Error (ATE trans [m] / ATE rot [deg]) -- Table I.

Umeyama/Horn rigid (optionally similarity) alignment of the estimated
trajectory onto ground truth before computing per-pose error, since SLAM
trajectories are only defined up to the world frame the system happened to
start in (and, for monocular-only configs, up to scale -- hence with_scale).
"""
from typing import List, Tuple
import numpy as np


def align_trajectories(
    est_pos: np.ndarray,
    gt_pos: np.ndarray,
    with_scale: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Umeyama alignment: finds (R, t, s) minimizing ||gt - (s*R@est + t)||^2.
    Returns (R (3,3), t (3,), s)."""
    mu_est = est_pos.mean(axis=0)
    mu_gt = gt_pos.mean(axis=0)
    est_c = est_pos - mu_est
    gt_c = gt_pos - mu_gt

    Sigma = (gt_c.T @ est_c) / est_pos.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_est = (est_c ** 2).sum(axis=1).mean()
        s = float(np.trace(np.diag(D) @ S) / var_est) if var_est > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_gt - s * R @ mu_est
    return R, t, s


def compute_ate(
    est_poses: List[np.ndarray],
    gt_poses: List[np.ndarray],
    align: bool = True,
    with_scale: bool = False,
) -> Tuple[float, float]:
    """
    est_poses, gt_poses: lists of (4, 4) poses, same length, same order.
    Returns (ate_trans_rmse [m], ate_rot_rmse [deg]) -- matches Table I's
    "ATE Trans [m] / ATE Rot [deg]" columns.
    """
    assert len(est_poses) == len(gt_poses) and len(est_poses) > 0
    est_pos = np.array([p[:3, 3] for p in est_poses])
    gt_pos = np.array([p[:3, 3] for p in gt_poses])
    est_rot = [p[:3, :3] for p in est_poses]
    gt_rot = [p[:3, :3] for p in gt_poses]

    if align:
        R, t, s = align_trajectories(est_pos, gt_pos, with_scale=with_scale)
        aligned_pos = (s * (R @ est_pos.T)).T + t
        aligned_rot = [R @ Re for Re in est_rot]
    else:
        aligned_pos = est_pos
        aligned_rot = est_rot

    trans_err = np.linalg.norm(aligned_pos - gt_pos, axis=1)
    ate_trans_rmse = float(np.sqrt(np.mean(trans_err ** 2)))

    rot_err_deg = []
    for Re, Rg in zip(aligned_rot, gt_rot):
        dR = Re.T @ Rg
        cos_angle = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg.append(np.degrees(np.arccos(cos_angle)))
    ate_rot_rmse = float(np.sqrt(np.mean(np.array(rot_err_deg) ** 2)))

    return ate_trans_rmse, ate_rot_rmse


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _selftest():
    """Runnable directly: python -m evaluation_slam.ate"""
    n = 20
    gt_poses = []
    for i in range(n):
        M = np.eye(4)
        M[:3, :3] = _rot_z(0.05 * i)
        M[:3, 3] = [float(i), 0.1 * np.sin(i), 0.0]
        gt_poses.append(M)

    # est = gt transformed by a fixed, arbitrary rigid offset (simulates a
    # SLAM system that starts in a different world frame but tracks perfectly)
    R_off = _rot_z(0.7)
    t_off = np.array([5.0, -3.0, 1.0])
    est_poses = []
    for M in gt_poses:
        E = np.eye(4)
        E[:3, :3] = R_off @ M[:3, :3]
        E[:3, 3] = R_off @ M[:3, 3] + t_off
        est_poses.append(E)

    ate_trans, ate_rot = compute_ate(est_poses, gt_poses, align=True)
    assert ate_trans < 1e-6, f"expected ~0 ATE trans after alignment, got {ate_trans}"
    assert ate_rot < 1e-4, f"expected ~0 ATE rot after alignment, got {ate_rot}"

    ate_trans_unaligned, _ = compute_ate(est_poses, gt_poses, align=False)
    assert ate_trans_unaligned > 1.0, "unaligned ATE should be large given the offset"

    print(f"evaluation_slam.ate self-test OK: aligned ATE trans={ate_trans:.2e}m rot={ate_rot:.2e}deg "
          f"(unaligned trans={ate_trans_unaligned:.2f}m)")


if __name__ == "__main__":
    _selftest()
