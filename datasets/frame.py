"""SyncedFrame: one synchronized (image, LiDAR, pose) sample for SLAM playback."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np

from .calibration import CalibrationSet


@dataclass
class SyncedFrame:
    frame_id: int
    timestamp: float                              # seconds
    points: np.ndarray                             # (N, 4) x,y,z,intensity, sensor frame
    calib: CalibrationSet
    image: Optional[np.ndarray] = None             # (H, W, 3) uint8, or None if unavailable
    pose: Optional[np.ndarray] = None               # (4,4) GT global pose of the LiDAR sensor, or None
    gt_boxes: List[Dict] = field(default_factory=list)
    token: Optional[str] = None                     # dataset-native id, for debugging/lookup
