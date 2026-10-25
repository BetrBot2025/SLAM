"""WP8: joint map-object pose-graph optimization (Sec. III).

Genuinely blocked on WP2: the paper extends ORB-SLAM3's own local bundle
adjustment (its g2o graph) with object pose/shape residuals, so there is no
meaningful implementation of this WP until a real ORB-SLAM3 binding
(slam_frontend/tracking_interface.py's ORBSlam3Tracker) exists to extend.

Two viable paths once WP2 lands (recorded here so the decision isn't lost):
  (a) Extend ORB-SLAM3's C++ g2o graph directly with object vertices/edges
      -- matches the paper exactly, but means writing C++ and extending the
      pybind11 binding's surface area.
  (b) Run a decoupled, Python-side pose graph (e.g. via `gtsam` or
      `g2opy`) that consumes ORB-SLAM3's camera poses/map points as fixed
      inputs each keyframe and only optimizes object pose/shape jointly with
      a local window of camera poses -- weaker coupling than the paper, but
      keeps all of WP8 in Python and avoids touching ORB-SLAM3 internals.
      Given this repo is pure Python/PyTorch elsewhere, (b) is the pragmatic
      default unless profiling shows it under-performs (a).

JointOptimizer below is the interface WP7's map objects and WP2's camera
poses should be handed through, whichever path is chosen.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class CameraPoseVertex:
    frame_id: int
    pose: np.ndarray   # (4, 4)


@dataclass
class ObjectVertex:
    object_id: int
    pose: np.ndarray        # (4, 4)
    latent_code: np.ndarray  # from WP6


class JointOptimizer(ABC):
    @abstractmethod
    def add_camera_pose(self, vertex: CameraPoseVertex) -> None: ...

    @abstractmethod
    def add_object(self, vertex: ObjectVertex) -> None: ...

    @abstractmethod
    def add_observation_edge(self, frame_id: int, object_id: int, observed_center: np.ndarray) -> None:
        """Residual tying a camera pose to an object observation at that keyframe."""
        ...

    @abstractmethod
    def optimize(self) -> None:
        """Run one local bundle-adjustment pass; updates vertices in place."""
        ...


class NotYetImplementedOptimizer(JointOptimizer):
    """Placeholder raising a clear error, so callers fail loudly instead of
    silently no-op-ing until WP2 (ORB-SLAM3) lands and a real path (a) or
    (b), per the module docstring, is picked and implemented."""

    def add_camera_pose(self, vertex: CameraPoseVertex) -> None:
        raise NotImplementedError("WP8 blocked on WP2 -- see module docstring")

    def add_object(self, vertex: ObjectVertex) -> None:
        raise NotImplementedError("WP8 blocked on WP2 -- see module docstring")

    def add_observation_edge(self, frame_id: int, object_id: int, observed_center: np.ndarray) -> None:
        raise NotImplementedError("WP8 blocked on WP2 -- see module docstring")

    def optimize(self) -> None:
        raise NotImplementedError("WP8 blocked on WP2 -- see module docstring")
