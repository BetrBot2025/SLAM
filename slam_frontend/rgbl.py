"""WP3: RGB-L metric depth fusion, feature-rectification strategy (Sec. VI,
Fig. 5b of the DSP-SLAM++ paper).

Rather than distorting the dense LiDAR point cloud into raw fisheye space,
sparse 2D features are undistorted into a virtual pinhole space (Kannala-
Brandt model), and the LiDAR cloud is projected into that same space to
associate metric depth -- cheaper since len(features) << len(lidar points).

This module is real, working geometry code, independently testable of WP2
(ORB-SLAM3) -- it only needs feature pixel coordinates and a LiDAR point
cloud in the camera frame, both of which can be synthesized for testing.
"""
from typing import Optional
import numpy as np

from datasets.calibration import FisheyeIntrinsics

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def kannala_brandt_undistort_points(
    pixels: np.ndarray,
    intr: FisheyeIntrinsics,
    virtual_K: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Undistort fisheye pixel coordinates into a virtual pinhole image
    using OpenCV's fisheye model (which implements Kannala-Brandt).

    Args:
        pixels: (N, 2) raw distorted pixel coordinates.
        virtual_K: pinhole intrinsics of the target rectified space; defaults
            to the fisheye camera's own K (same focal length/principal point).

    Returns: (N, 2) rectified pixel coordinates.
    """
    if not _HAS_CV2:
        raise ImportError("kannala_brandt_undistort_points needs opencv-python: pip install opencv-python")
    if virtual_K is None:
        virtual_K = intr.K()

    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    rectified = cv2.fisheye.undistortPoints(pts, intr.K(), intr.D(), P=virtual_K)
    return rectified.reshape(-1, 2)


def project_pinhole(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(N, 3) points in camera frame -> (N, 2) pixel coords via a pinhole K.
    Points with z <= 0 (behind the camera) are returned as NaN."""
    z = points_cam[:, 2]
    px = np.full((points_cam.shape[0], 2), np.nan)
    valid = z > 1e-6
    uv = (K[:2, :2] @ (points_cam[valid, :2] / z[valid, None]).T).T + K[:2, 2]
    px[valid] = uv
    return px


def associate_lidar_depth(
    rectified_feature_px: np.ndarray,   # (N, 2), output of kannala_brandt_undistort_points
    lidar_points_cam: np.ndarray,        # (M, 3), LiDAR points already in camera frame
    virtual_K: np.ndarray,
    search_radius_px: float = 3.0,
) -> np.ndarray:
    """For each rectified feature, find the nearest projected LiDAR point
    within search_radius_px and return its camera-frame depth (z). NaN where
    no LiDAR point is close enough (Sec. VI: sparse feature set, dense LiDAR
    projected into the same virtual pinhole space)."""
    lidar_px = project_pinhole(lidar_points_cam, virtual_K)
    valid = ~np.isnan(lidar_px[:, 0])
    lidar_px_valid = lidar_px[valid]
    depths_valid = lidar_points_cam[valid, 2]

    n = rectified_feature_px.shape[0]
    out_depth = np.full(n, np.nan)
    if lidar_px_valid.shape[0] == 0:
        return out_depth

    if _HAS_SCIPY:
        tree = cKDTree(lidar_px_valid)
        dist, idx = tree.query(rectified_feature_px, distance_upper_bound=search_radius_px)
        found = np.isfinite(dist)
        out_depth[found] = depths_valid[idx[found]]
    else:
        # Brute-force fallback (fine for the modest feature counts this is
        # designed for; swap in scipy.spatial.cKDTree for large-scale use).
        for i in range(n):
            d = np.hypot(lidar_px_valid[:, 0] - rectified_feature_px[i, 0],
                         lidar_px_valid[:, 1] - rectified_feature_px[i, 1])
            j = np.argmin(d)
            if d[j] <= search_radius_px:
                out_depth[i] = depths_valid[j]
    return out_depth


def _selftest():
    """Runnable directly: python -m slam_frontend.rgbl"""
    intr = FisheyeIntrinsics(fx=400, fy=400, cx=320, cy=240, k1=-0.02, k2=0.001, k3=0.0, k4=0.0,
                              width=640, height=480)

    if _HAS_CV2:
        # Round-trip: distort a known pinhole point with cv2.fisheye, then
        # undistort it back and check we recover the original pixel.
        obj_pt = np.array([[[0.1, 0.05, 1.0]]], dtype=np.float64)   # (1,1,3) camera-frame point
        distorted, _ = cv2.fisheye.projectPoints(
            obj_pt, np.zeros(3), np.zeros(3), intr.K(), intr.D()
        )
        distorted_px = distorted.reshape(1, 2)
        rectified_px = kannala_brandt_undistort_points(distorted_px, intr)
        expected_px = project_pinhole(obj_pt.reshape(1, 3), intr.K())
        err = np.linalg.norm(rectified_px - expected_px)
        assert err < 1.0, f"KB round-trip pixel error too large: {err:.3f}px"
        print(f"slam_frontend.rgbl self-test OK: KB round-trip error {err:.4f}px")
    else:
        print("slam_frontend.rgbl self-test SKIPPED: opencv-python not installed")

    # associate_lidar_depth, independent of cv2
    K = intr.K()
    lidar_cam = np.array([[0.1, 0.05, 5.0], [2.0, 2.0, 8.0]])
    feature_px = project_pinhole(lidar_cam[:1], K)   # exactly on top of the first LiDAR point
    depth = associate_lidar_depth(feature_px, lidar_cam, K, search_radius_px=2.0)
    assert abs(depth[0] - 5.0) < 1e-6, f"expected depth 5.0, got {depth[0]}"
    print("slam_frontend.rgbl depth association self-test OK:", depth)


if __name__ == "__main__":
    _selftest()
