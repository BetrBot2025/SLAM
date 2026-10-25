"""WP0-WP10 end-to-end wiring: ties every work package's module together into
one runnable loop, matching the DSP-SLAM++ thread architecture (Fig. 2) as
closely as today's real components allow.

What's REAL in this run: WP0 streaming, WP1 3D detection (your trained
checkpoint), WP4 association algorithm, WP5 map association, WP6 DeepSDF
optimization loop (untrained decoders), WP7 async backend, WP9 ATE/latency,
WP10 visualization.

What's a STAND-IN, clearly flagged at runtime: WP1's 2D half (NullDetector2D
-- no YOLO weights wired yet, so WP4's real bipartite matching is bypassed
in favor of a degraded passthrough), WP8 (no-op, depends on a real WP2 BA
graph to extend).

WP2 tracking defaults to DummyGTPoseTracker (replays ground truth -- only
meaningful when a real dataset with GT poses is given). Pass --tracker
orbslam3 for the real ORB-SLAM3 binding (slam_frontend/tracking_interface.py)
-- this only works from inside the Linux/WSL2 environment the
orbslam3_python module was built in (see slam_frontend/orbslam3_build/),
never from a plain Windows venv, and needs --orb-vocab/--orb-settings.

Usage:
    python run_slam.py                                   # synthetic data, 4-class checkpoint, dummy tracker
    python run_slam.py --nuscenes-root <path> --version v1.0-mini
    python run_slam.py --tracker orbslam3 \\
        --orb-vocab /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \\
        --orb-settings /opt/ORB_SLAM3/Examples/Monocular/EuRoC.yaml
"""
import argparse
import os
from typing import List, Optional

import numpy as np

from lidar_detection.config import PipelineConfig
from lidar_detection.pipeline import DetectionPipeline
from lidar_detection.data.generator import generate_scene
from lidar_detection.visualization.visualizer import draw_3d_map

from datasets.frame import SyncedFrame
from datasets.calibration import CalibrationSet
from detection2d.yolo_seg import NullDetector2D
from association.bipartite import Detection3D, build_score_matrix, global_greedy_assignment
from association.map_association import MapObject, associate_to_map
from async_backend.pipeline import AsyncReconstructionBackend, ReconstructionTask, ReconstructionResult
from reconstruction.multi_decoder import MultiDecoderReconstructor
from slam_frontend.tracking_interface import DummyGTPoseTracker
from evaluation_slam.ate import compute_ate
from evaluation_slam.latency import LatencyTracker


def _make_synthetic_canvas(size=(720, 1280), seed: int = 0) -> np.ndarray:
    """Deterministic textured grayscale canvas, numpy-only (no cv2 needed).
    A panning crop across this canvas (see _synthetic_image) gives a real
    tracker frame-to-frame structure to latch onto, unlike independent pure
    noise per frame -- enough to prove ORBSlam3Tracker's wiring runs
    end-to-end; not meant to produce an accurate trajectory."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    checker = (((xx // 24) + (yy // 24)) % 2) * 180
    noise = rng.integers(0, 60, size=(h, w))
    return np.clip(checker + noise, 0, 255).astype(np.uint8)


def _synthetic_image(canvas: np.ndarray, frame_idx: int, crop=(480, 640), pan_px: int = 6) -> np.ndarray:
    h, w = crop
    x0 = (frame_idx * pan_px) % (canvas.shape[1] - w)
    y0 = canvas.shape[0] // 2 - h // 2
    return canvas[y0:y0 + h, x0:x0 + w]


def synthetic_frame_stream(cfg: PipelineConfig, n_frames: int, seed: int):
    """Fallback stream when no real dataset root is given -- reuses the
    existing synthetic generator (same one demo.py uses) for points, plus a
    synthetic panning image (see _make_synthetic_canvas) so --tracker
    orbslam3 has something to run against even with no real dataset. Note:
    the point generator's own class taxonomy (Car/Pedestrian/Cyclist)
    predates the 4-class checkpoint; that's fine here since we only feed it
    raw points to the detector and never compare against its gt_boxes for
    class matching."""
    pose = np.eye(4)
    canvas = _make_synthetic_canvas(seed=seed)
    for i in range(n_frames):
        points, gt_boxes = generate_scene(seed=seed + i)
        pose = pose.copy()
        pose[:3, 3] += np.array([1.0, 0.0, 0.0])
        yield SyncedFrame(frame_id=i, timestamp=i * 0.1, points=points,
                           calib=CalibrationSet(), image=_synthetic_image(canvas, i), pose=pose.copy(),
                           gt_boxes=gt_boxes, token=f"synthetic-{i}")


def nuscenes_frame_stream(root: str, version: str):
    from lidar_detection.data.nuscenes_loader import NuScenesDataset
    from datasets.nuscenes_stream import NuScenesStream
    dataset = NuScenesDataset(root=root, version=version)
    return NuScenesStream(dataset)


def _crop_points_to_box(points: np.ndarray, box: List[float], margin: float = 0.3) -> np.ndarray:
    """Points within an axis-aligned box around the detection's centre,
    re-expressed in the object's local (centered) frame. A real system would
    do this properly (yaw-aligned crop, accumulated across frames per
    tracked object) -- this is a single-frame stand-in, enough to feed WP6."""
    cx, cy, cz, l, w, h, yaw = box
    d = points[:, :3] - np.array([cx, cy, cz])
    half = np.array([l / 2 + margin, w / 2 + margin, h / 2 + margin])
    mask = np.all(np.abs(d) <= half, axis=1)
    return d[mask].astype(np.float32)


def run(args):
    cfg = PipelineConfig()
    print(f"3D detector: CenterPoint, classes={cfg.model.class_names}, checkpoint={args.checkpoint}")
    detector3d = DetectionPipeline(cfg, checkpoint=args.checkpoint, use_center=True)
    detector2d = NullDetector2D()   # WP1's 2D half -- see module docstring

    if args.tracker == "orbslam3":
        from slam_frontend.tracking_interface import ORBSlam3Tracker
        if not (args.orb_vocab and args.orb_settings):
            raise SystemExit("--tracker orbslam3 needs --orb-vocab and --orb-settings")
        print(f"WP2: real ORBSlam3Tracker (vocab={args.orb_vocab}, settings={args.orb_settings})")
        tracker = ORBSlam3Tracker(args.orb_vocab, args.orb_settings, use_viewer=False)
    else:
        tracker = DummyGTPoseTracker()  # WP2 stand-in -- see module docstring
    reconstructor = MultiDecoderReconstructor(class_names=cfg.model.class_names, latent_dim=32, hidden_dim=64)
    latency = LatencyTracker()

    def _reconstruct_fn(task: ReconstructionTask) -> ReconstructionResult:
        out = reconstructor.reconstruct(task.class_name, task.point_cloud, num_iters=30, extract=True)
        return ReconstructionResult(object_id=task.object_id, latent_code=out.latent_code,
                                     pose=task.pose, mesh=out.mesh)

    backend = AsyncReconstructionBackend(_reconstruct_fn, num_workers=2)

    all_map_objects: List[MapObject] = []
    tracked_objects: List[MapObject] = []
    placeholders_by_id = {}
    next_object_id = 0

    est_trajectory, gt_trajectory = [], []

    if args.nuscenes_root:
        stream = nuscenes_frame_stream(args.nuscenes_root, args.version)
        if len(stream) == 0:
            print("No nuScenes frames found at that root -- falling back to synthetic data.")
            stream = list(synthetic_frame_stream(cfg, args.n_frames, args.seed))
    else:
        stream = list(synthetic_frame_stream(cfg, args.n_frames, args.seed))

    n_frames = min(len(stream), args.n_frames) if args.nuscenes_root else len(stream)
    warned_no_2d = False

    for i, frame in enumerate(stream):
        if i >= n_frames:
            break
        latency.tick_frame()

        pose = tracker.process_frame(frame)   # WP2 stand-in
        est_trajectory.append(pose.matrix[:3, 3])
        gt_trajectory.append(frame.pose[:3, 3] if frame.pose is not None else pose.matrix[:3, 3])

        with latency.measure("detect"):
            dets_3d_raw = detector3d.run(frame.points)
            dets_2d = detector2d.run(frame.image) if frame.image is not None else []

        det3d = [
            Detection3D(class_id=d["class_id"], class_name=d["class_name"], score=d["score"],
                        box=d["box"], projected_points=np.zeros((0, 2)))
            for d in dets_3d_raw
        ]

        if dets_2d:
            S = build_score_matrix(det3d, dets_2d)
            pairs = global_greedy_assignment(S, det3d)
            d_valid = [det3d[i3] for i3, _ in pairs]
        else:
            if not warned_no_2d and det3d:
                print("  [no 2D detector wired -- bypassing WP4's real 2D-3D association; "
                      "treating 3D detections directly as validated candidates for this smoke test]")
                warned_no_2d = True
            d_valid = det3d   # degraded fallback -- see run() docstring

        with latency.measure("object"):
            tracked_matches, reappear_matches, new_objects = associate_to_map(
                tracked_objects, all_map_objects, d_valid, frame_id=i, next_object_id=next_object_id,
            )
            next_object_id += len(new_objects)

            for obj in new_objects:
                cropped = _crop_points_to_box(frame.points, list(obj.center) + list(obj.size) + [obj.yaw])
                if cropped.shape[0] >= 10:
                    task = ReconstructionTask(object_id=obj.object_id, class_name=obj.class_name,
                                               point_cloud=cropped, pose=pose.matrix,
                                               box=list(obj.center) + list(obj.size) + [obj.yaw])
                    placeholders_by_id[obj.object_id] = backend.submit_task(task)

            all_map_objects.extend(new_objects)
            tracked_objects = all_map_objects   # single-camera-view demo: everything stays "tracked"

        for result in backend.drain_results():
            print(f"  frame {i}: reconstruction ready for object {result.object_id} "
                  f"(mesh={'yes' if result.mesh else 'no -- scikit-image missing or empty SDF'})")

        if i % max(1, n_frames // 5) == 0 or i == n_frames - 1:
            print(f"frame {i}/{n_frames-1}: {len(det3d)} 3D dets, {len(d_valid)} validated, "
                  f"{len(new_objects)} new objects, map size={len(all_map_objects)}, fps={latency.fps():.1f}")

    backend.shutdown()
    if hasattr(tracker, "shutdown"):
        tracker.shutdown()

    print("\n--- WP9: evaluation ---")
    latency.print_summary()
    if len(est_trajectory) > 1:
        est_poses = [np.eye(4) for _ in est_trajectory]
        gt_poses = [np.eye(4) for _ in gt_trajectory]
        for M, p in zip(est_poses, est_trajectory):
            M[:3, 3] = p
        for M, p in zip(gt_poses, gt_trajectory):
            M[:3, 3] = p
        ate_trans, ate_rot = compute_ate(est_poses, gt_poses)
        if args.tracker == "orbslam3":
            print(f"ATE trans={ate_trans:.4f}m rot={ate_rot:.4f}deg "
                  f"(real ORB-SLAM3 estimate vs. dataset GT -- meaningless against the synthetic "
                  f"panning-image fallback; only meaningful with a real image sequence + GT poses)")
        else:
            print(f"ATE trans={ate_trans:.4f}m rot={ate_rot:.4f}deg "
                  f"(DummyGTPoseTracker replays GT, so ~0 here is expected -- not a real tracking result)")

    print("\n--- WP10: visualization ---")
    viz_objects = [
        {"center": obj.center.tolist(), "size": obj.size.tolist(), "yaw": obj.yaw,
         "class_name": obj.class_name, "mesh": None}
        for obj in all_map_objects
    ]
    out_path = args.save or "slam_map.png"
    draw_3d_map(viz_objects, trajectory_est=np.array(est_trajectory),
                trajectory_gt=np.array(gt_trajectory), save_path=out_path)

    print(f"\nDone: {len(all_map_objects)} objects in the map. Map plot: {out_path}")


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoint_nuscenes_center_4class.pth")
    p.add_argument("--nuscenes-root", default=None, help="path to a real nuScenes root, if you have one")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--n-frames", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", default=None)
    p.add_argument("--tracker", choices=["dummy", "orbslam3"], default="dummy",
                    help="dummy: replay dataset GT pose. orbslam3: real ORB-SLAM3 "
                         "(needs --orb-vocab/--orb-settings, and running inside the "
                         "Linux env orbslam3_python was built in -- see slam_frontend/orbslam3_build/)")
    p.add_argument("--orb-vocab", default=None, help="path to ORBvoc.txt")
    p.add_argument("--orb-settings", default=None, help="path to an ORB-SLAM3 settings YAML")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found at {args.checkpoint} -- pass --checkpoint <path>. Continuing "
              f"with randomly-initialized weights (detections will be meaningless, but the pipeline still runs).")
        args.checkpoint = None
    run(args)
