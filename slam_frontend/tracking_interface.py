"""WP2: fisheye visual tracking front-end interface.

Status: ORB-SLAM3 is built and verified (Docker image orbslam3:22.04, built
inside WSL2 -- see slam_frontend/orbslam3_build/), including a pybind11
binding (`orbslam3_python`) with a source patch adding the one thing vanilla
ORB-SLAM3 doesn't expose: whether the last processed frame became a
keyframe. Constructing System() and running track_monocular() end-to-end
against ORB-SLAM3's own EuRoC example config has been smoke-tested and
works. Caveat that matters for how you run this: `orbslam3_python` is a
native Linux (.so) extension built for the Linux/WSL2 Python it was compiled
against -- it is NOT importable from this repo's Windows venv. Anything
using ORBSlam3Tracker for real has to run inside that same Linux
environment, not the Windows-side Python.

There is no separate "TrackFisheye()" in ORB-SLAM3 -- fisheye vs. pinhole is
selected by the settings YAML's Camera.type field (e.g. "KannalaBrandt8"),
and both go through the same TrackMonocular() call. What this module gives you:

  1. The Pose/TrackingFrontend interface every downstream WP (WP3 metric
     fusion, WP4 association timing, WP8 joint optimization) codes against,
     so they don't need to change based on which tracker is wired in.
  2. DummyGTPoseTracker: a concrete stand-in that replays a dataset's own
     ground-truth pose as if it were "tracked". This is NOT SLAM -- it's a
     scaffold that lets WP4-WP10 be built, run, and smoke-tested end-to-end
     (see run_slam.py) independent of whether you're in a Linux environment.
  3. ORBSlam3Tracker: the real WP2 implementation, wrapping
     orbslam3_python.System.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from datasets.frame import SyncedFrame


@dataclass
class Pose:
    matrix: np.ndarray       # (4, 4) camera-to-world (or lidar-to-world if no camera)
    timestamp: float
    is_keyframe: bool = False


class TrackingFrontend(ABC):
    @abstractmethod
    def process_frame(self, frame: SyncedFrame) -> Pose:
        """Track one frame, return its estimated pose."""
        ...

    @abstractmethod
    def get_map_points(self) -> np.ndarray:
        """(N, 3) sparse map points accumulated so far, for WP8's joint BA."""
        ...


class DummyGTPoseTracker(TrackingFrontend):
    """Replays SyncedFrame.pose (dataset ground truth) as the tracked pose.
    Every frame with a pose is treated as a keyframe. Use only to exercise
    WP4-WP10 before a real tracker exists -- it trivially achieves ATE=0
    against its own source, which is not a meaningful tracking result."""

    def __init__(self):
        self._map_points = np.zeros((0, 3))

    def process_frame(self, frame: SyncedFrame) -> Pose:
        if frame.pose is None:
            raise ValueError(
                f"DummyGTPoseTracker requires SyncedFrame.pose (frame {frame.frame_id} has none)"
            )
        return Pose(matrix=frame.pose, timestamp=frame.timestamp, is_keyframe=True)

    def get_map_points(self) -> np.ndarray:
        return self._map_points


class ORBSlam3Tracker(TrackingFrontend):
    """Real WP2 implementation, wrapping the orbslam3_python pybind11 module
    built in slam_frontend/orbslam3_build/ (see that directory's README for
    how to build/rebuild it). Monocular only -- fisheye vs. pinhole is
    selected by `settings_path`'s Camera.type, not a constructor argument.

    settings_path needs a full ORB-SLAM3 YAML (camera intrinsics/distortion,
    ORB extractor params, fps, etc.) -- see Examples/*/*.yaml in the
    ORB-SLAM3 checkout for the format; there is no fisheye example shipped
    with ORB-SLAM3 itself, so a real fisheye rig needs its own YAML with
    Camera.type: "KannalaBrandt8" and k1-k4 filled in from calibration.
    """

    def __init__(self, vocab_path: str, settings_path: str, use_viewer: bool = False):
        try:
            import orbslam3_python as _orb
        except ImportError as e:
            raise ImportError(
                "ORBSlam3Tracker needs the `orbslam3_python` module built in "
                "slam_frontend/orbslam3_build/ (docker build, see that directory's "
                "README). It's a native Linux extension -- importable only from "
                "Python running inside that same Linux/WSL2 environment, never "
                "from this repo's Windows venv."
            ) from e
        self._system = _orb.System(vocab_path, settings_path, use_viewer=use_viewer)
        self._map_points = np.zeros((0, 3))

    def process_frame(self, frame: SyncedFrame) -> Pose:
        if frame.image is None:
            raise ValueError(f"ORBSlam3Tracker requires SyncedFrame.image (frame {frame.frame_id} has none)")
        Tcw = self._system.track_monocular(frame.image, frame.timestamp)   # (4,4): world -> camera
        Twc = np.linalg.inv(Tcw)                                            # camera -> world, matches Pose's convention
        is_kf = self._system.is_keyframe()
        self._map_points = self._system.get_map_points()
        return Pose(matrix=Twc, timestamp=frame.timestamp, is_keyframe=is_kf)

    def get_map_points(self) -> np.ndarray:
        return self._map_points

    def shutdown(self):
        self._system.shutdown()
