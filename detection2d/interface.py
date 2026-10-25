"""Det2D interface: abstract 2D instance-segmentation detector.

Kept independent of any specific model so association/bipartite.py (WP4)
never has to know whether detections came from YOLO11x-seg, another
segmenter, or (during bring-up) a null stand-in.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Must match the class taxonomy used by the 3D detector
# (lidar_detection/config.py ModelConfig.class_names) so Eq. 1's class-
# consistency gate (c_3D == c_2D) is meaningful.
CLASS_NAMES = ["Car", "Bus", "Motorcycle", "Pedestrian"]


@dataclass
class Detection2D:
    mask: np.ndarray        # (H, W) bool — segmentation mask in image space
    class_id: int
    class_name: str
    score: float
    bbox: Optional[np.ndarray] = None   # (4,) x1,y1,x2,y2, informational only


class Detector2D(ABC):
    """Implementations return per-class-thresholded Detection2D instances
    for one image, matching D_2D at Algorithm 1 line 2."""

    @abstractmethod
    def run(self, image: np.ndarray) -> List[Detection2D]:
        ...
