"""Calibration containers: camera intrinsics (pinhole / Kannala-Brandt fisheye)
and sensor extrinsics. Static per-rig data — per-frame dynamic pose lives on
SyncedFrame, not here.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)


@dataclass
class FisheyeIntrinsics:
    """Kannala-Brandt (equidistant) fisheye model — matches OpenCV's
    cv2.fisheye convention: r(theta) = theta + k1*theta^3 + k2*theta^5
    + k3*theta^7 + k4*theta^9, theta = angle from the optical axis.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    width: int
    height: int

    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.k3, self.k4], dtype=np.float64)


@dataclass
class Extrinsics:
    """Rigid transform: p_dst = R @ p_src + t."""
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """points: (N, 3) in src frame -> (N, 3) in dst frame."""
        return (self.R @ points.T).T + self.t

    def as_matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def inverse(self) -> "Extrinsics":
        R_inv = self.R.T
        return Extrinsics(R=R_inv, t=-R_inv @ self.t)


@dataclass
class CalibrationSet:
    """Static per-rig calibration bundle for one dataset/sensor suite."""
    camera: Optional[object] = None            # PinholeIntrinsics | FisheyeIntrinsics | None
    lidar_to_camera: Optional[Extrinsics] = None
    is_fisheye: bool = False
