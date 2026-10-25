"""WP0: sequential, per-frame nuScenes adapter for SLAM playback.

Wraps lidar_detection.data.nuscenes_loader.NuScenesDataset (training-time,
random-access) in scene order, yielding SyncedFrame with GT global pose
(for ATE evaluation) attached.

CAM_FRONT images are loaded when the dataset root has them indexed (requires
opencv-python) -- SyncedFrame.image is None only when a sample truly has no
camera keyframe or cv2 isn't installed, not unconditionally as before.
camera/lidar_to_camera calibration is still not wired (no CAM_FRONT
intrinsics/extrinsics parsing yet) -- needed before a real fisheye/pinhole
tracker can consume these frames for anything beyond a raw image.
"""
from typing import Iterator, Optional
import numpy as np

from lidar_detection.data.nuscenes_loader import NuScenesDataset
from .calibration import CalibrationSet
from .frame import SyncedFrame


class NuScenesStream:
    def __init__(self, dataset: NuScenesDataset):
        """Wrap an already-constructed NuScenesDataset (any split/version)."""
        self.dataset = dataset
        # Camera intrinsics/extrinsics not parsed yet -- see module docstring.
        self.calib = CalibrationSet(camera=None, lidar_to_camera=None, is_fisheye=False)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> SyncedFrame:
        points, gt_boxes = self.dataset[idx]
        R, t = self.dataset.get_ego_pose(idx)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return SyncedFrame(
            frame_id=idx,
            timestamp=self.dataset.get_timestamp(idx),
            points=points,
            calib=self.calib,
            image=self.dataset.get_image(idx),
            pose=pose,
            gt_boxes=gt_boxes,
            token=self.dataset.get_token(idx),
        )

    def __iter__(self) -> Iterator[SyncedFrame]:
        for i in range(len(self)):
            yield self[i]
