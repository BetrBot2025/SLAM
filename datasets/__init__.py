"""WP0: streaming (image, LiDAR, pose) dataset adapters + calibration containers.

Separate from lidar_detection/data/, which is training-time (batched, shuffled)
loading. This package yields sequential, per-frame SyncedFrame objects suitable
for SLAM playback.
"""
from .frame import SyncedFrame
from .calibration import PinholeIntrinsics, FisheyeIntrinsics, Extrinsics, CalibrationSet

__all__ = [
    "SyncedFrame",
    "PinholeIntrinsics", "FisheyeIntrinsics", "Extrinsics", "CalibrationSet",
]
