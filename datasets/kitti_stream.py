"""WP0: sequential per-frame KITTI odometry adapter for SLAM playback.

Expects the standard KITTI odometry layout:
    <root>/sequences/<seq>/velodyne/000000.bin   (N,4) float32 x,y,z,reflectance
    <root>/sequences/<seq>/image_2/000000.png     (optional; left color camera)
    <root>/sequences/<seq>/calib.txt              P0..P3, Tr (velo->cam0)
    <root>/poses/<seq>.txt                        (optional) 3x4 GT pose per line, cam0 frame

Tolerant of missing pieces: no poses.txt -> pose=None; no image_2 -> image=None.
"""
import os
from typing import Iterator, List, Optional
import numpy as np

from .calibration import PinholeIntrinsics, Extrinsics, CalibrationSet
from .frame import SyncedFrame

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _parse_calib(path: str) -> CalibrationSet:
    values = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            values[key.strip()] = np.array([float(x) for x in rest.split()])

    camera = None
    if "P2" in values:
        P2 = values["P2"].reshape(3, 4)
        camera = PinholeIntrinsics(
            fx=float(P2[0, 0]), fy=float(P2[1, 1]),
            cx=float(P2[0, 2]), cy=float(P2[1, 2]),
            width=1242, height=375,   # standard KITTI image_2 size; informational only
        )

    lidar_to_camera = None
    if "Tr" in values:
        Tr = values["Tr"].reshape(3, 4)
        lidar_to_camera = Extrinsics(R=Tr[:, :3], t=Tr[:, 3])

    return CalibrationSet(camera=camera, lidar_to_camera=lidar_to_camera, is_fisheye=False)


def _parse_poses(path: str) -> List[np.ndarray]:
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                continue
            M = np.eye(4)
            M[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(M)
    return poses


class KittiStream:
    def __init__(self, root: str, sequence: str = "00"):
        self.seq_dir = os.path.join(root, "sequences", sequence)
        self.velodyne_dir = os.path.join(self.seq_dir, "velodyne")
        self.image_dir = os.path.join(self.seq_dir, "image_2")

        self._frame_ids = sorted(
            f[:-4] for f in os.listdir(self.velodyne_dir) if f.endswith(".bin")
        ) if os.path.isdir(self.velodyne_dir) else []

        calib_path = os.path.join(self.seq_dir, "calib.txt")
        self.calib = _parse_calib(calib_path) if os.path.isfile(calib_path) else CalibrationSet()

        poses_path = os.path.join(root, "poses", f"{sequence}.txt")
        self._poses = _parse_poses(poses_path) if os.path.isfile(poses_path) else None

        if not self._frame_ids:
            print(f"KittiStream: no velodyne frames found under {self.velodyne_dir} "
                  f"-- point the `root` at a real KITTI odometry download to use this stream.")

    def __len__(self) -> int:
        return len(self._frame_ids)

    def __getitem__(self, idx: int) -> SyncedFrame:
        fid = self._frame_ids[idx]
        points = np.fromfile(
            os.path.join(self.velodyne_dir, f"{fid}.bin"), dtype=np.float32
        ).reshape(-1, 4)

        image = None
        img_path = os.path.join(self.image_dir, f"{fid}.png")
        if _HAS_CV2 and os.path.isfile(img_path):
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        pose = self._poses[idx] if self._poses is not None and idx < len(self._poses) else None

        return SyncedFrame(
            frame_id=idx,
            timestamp=float(idx) / 10.0,   # KITTI odometry LiDAR runs at ~10 Hz; no per-frame timestamps shipped
            points=points,
            calib=self.calib,
            image=image,
            pose=pose,
            gt_boxes=[],   # KITTI odometry has no 3D box annotations (that's KITTI detection, a different split)
            token=fid,
        )

    def __iter__(self) -> Iterator[SyncedFrame]:
        for i in range(len(self)):
            yield self[i]
