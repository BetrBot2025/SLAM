# ORB-SLAM3 build (WP2)

Verified working build of [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3),
containerized against `ubuntu:22.04` (not the ORB-SLAM3-era 20.04 -- see the
Dockerfile's top comment for why) via Docker running inside WSL2.

## Prerequisites

- WSL2 with an Ubuntu distro (`wsl -d Ubuntu -u root -- docker --version` to check)
- Docker inside that distro: `apt-get install -y docker.io && systemctl enable --now docker`

## Build

```bash
# from WSL2, inside this directory (or via the /mnt/c/... path from Windows)
docker build -t orbslam3:22.04 -f Dockerfile .
```

Takes a while (Pangolin + DBoW2 + g2o + Sophus + ORB-SLAM3 core, ~1.4GB
image). Known pitfall already hit and fixed here: the final `make` step
must NOT run at full `-j$(nproc)` on memory-constrained hosts (e.g. WSL2's
default ~7.6GB budget on a 16GB machine) -- `Tracking.cc`/`Optimizer.cc`/
`G2oTypes.cc` each pull in heavy Eigen+g2o template instantiation and can
use several GB per translation unit; too many in parallel gets `cc1plus`
OOM-killed. The Dockerfile pins that stage to `-j2`.

## Verify

```bash
docker run --rm orbslam3:22.04 bash -c "ls /opt/ORB_SLAM3/lib/libORB_SLAM3.so"
docker run --rm orbslam3:22.04 /opt/ORB_SLAM3/Examples/Monocular/mono_euroc
```

The second command should print ORB-SLAM3's own usage message (it takes
vocabulary/settings/dataset paths) -- that confirms the binary actually
links and runs, not just compiles.

## Python binding

`orbslam3_python.cc` (built into the image as an extra CMake target,
`lib/orbslam3_python.cpython-310-x86_64-linux-gnu.so`) exposes:

- `System(vocab_path, settings_path, use_viewer=False)`
- `.track_monocular(image, timestamp) -> (4,4) float32 Tcw` (world -> camera)
- `.get_map_points() -> (N,3) float32`
- `.is_keyframe() -> bool` -- relies on the source patch below, since vanilla
  ORB-SLAM3 has no public way to know this
- `.tracking_state() -> TrackingState` enum
- `.shutdown()`

`patch_orbslam3.py` adds `System::WasLastFrameAKeyFrame()` (and the
`Tracking`-level flag it forwards) to the cloned source before building --
Algorithm 1 of DSP-SLAM++ needs that signal to gate detection, and vanilla
ORB-SLAM3 doesn't expose it. This is the same kind of source-level
extension the original DSP-SLAM made to ORB-SLAM2/3 for the same reason.

Smoke-tested end-to-end (constructed `System` against ORB-SLAM3's own
`Examples/Monocular/EuRoC.yaml`, ran one frame through `track_monocular`,
confirmed shapes/types and a clean `shutdown()`) -- proves the binding's
plumbing is correct, not that tracking itself is accurate (that needs a
real image sequence, not one frame of random noise).

**Importable only from Linux/WSL2 Python** (the `.so` is a native extension
built against the container's Python 3.10) -- `slam_frontend/tracking_interface.py`'s
`ORBSlam3Tracker` (the Python-side wrapper this repo's pipeline actually
calls) cannot be imported from this repo's Windows venv.

## What's still needed for WP2

- A real fisheye YAML (Camera.type: "KannalaBrandt8" + k1-k4) for an actual
  rig -- no fisheye example ships with ORB-SLAM3 itself, only pinhole ones.
- Running the Python pipeline (run_slam.py et al.) inside this Linux
  environment when using `ORBSlam3Tracker`, since it can't cross into the
  Windows venv -- either mirror the venv inside WSL2, or run everything
  from within a container built on top of this image.
- WP3's real fisheye-LiDAR association wired through ORB-SLAM3's actual
  tracked features (slam_frontend/rgbl.py's math is ready; it just hasn't
  been connected to a live `ORBSlam3Tracker` run yet).

## Licensing

ORB-SLAM3 is GPLv3. Anything that statically or dynamically links against
it (including a pybind11 binding) inherits that obligation if distributed.
