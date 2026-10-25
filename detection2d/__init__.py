"""WP1 (remaining gap): 2D multi-class instance segmentation.

lidar_detection/pipeline.py already provides the 3D side (Det3D) with
use_center=True + per-class thresholds. This package provides Det2D:
per-keyframe instance masks with class labels, matching Algorithm 1 line 2
of the DSP-SLAM++ paper (D_2D).
"""
from .interface import Detection2D, Detector2D
from .yolo_seg import YoloSegDetector2D, NullDetector2D

__all__ = ["Detection2D", "Detector2D", "YoloSegDetector2D", "NullDetector2D"]
