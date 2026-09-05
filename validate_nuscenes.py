"""
Validate PointPillars on nuScenes held-out val split.

Reports official nuScenes detection metrics:
  - AP per class at D = {0.5, 1, 2, 4} m centre distance
  - mAP averaged over classes and distance thresholds
  - TP metrics: mATE (translation), mASE (scale), mAOE (orientation)
  - NDS (nuScenes Detection Score, partial — no velocity/attribute head)

Usage:
    python validate_nuscenes.py
    python validate_nuscenes.py --checkpoint checkpoint_nuscenes.pth --n 500
    python validate_nuscenes.py --blobs 1 2 --n 1000 --score-thr 0.05
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from lidar_detection.config import PipelineConfig
from lidar_detection.data.nuscenes_loader import NuScenesDataset
from lidar_detection.models.pointpillars import PointPillars, CenterPointPillars
from lidar_detection.evaluation.evaluator import Evaluator
from lidar_detection.evaluation.metrics import compute_nuscenes_metrics

DEFAULT_DATAROOT = "lidar_detection/data/nuscenes_full"
VERSION          = "v1.0-trainval"


def _print_diagnostics(all_dets, all_gts, cfg, score_thr):
    class_names = cfg.model.class_names
    pc_range    = cfg.voxel.point_cloud_range

    det_counts      = {c: 0 for c in class_names}
    all_scores      = []
    frames_with_det = 0
    for frame_dets in all_dets:
        if frame_dets:
            frames_with_det += 1
        for d in frame_dets:
            det_counts[class_names[d["class_id"]]] += 1
            all_scores.append(d["score"])

    gt_counts  = {c: 0 for c in class_names}
    gt_inrange = {c: 0 for c in class_names}
    for frame_gts in all_gts:
        for g in frame_gts:
            cn = class_names[g["class_id"]]
            gt_counts[cn] += 1
            bx, by = g["box"][0], g["box"][1]
            if (pc_range[0] <= bx <= pc_range[3] and
                    pc_range[1] <= by <= pc_range[4]):
                gt_inrange[cn] += 1

    n = len(all_dets)
    print("-- Diagnostics -------------------------------------------------")
    print(f"  Frames evaluated          : {n}")
    print(f"  Frames with >0 detections : {frames_with_det}  "
          f"({frames_with_det / max(n, 1) * 100:.1f}%)")
    print(f"  Total detections          : {sum(det_counts.values())}")
    for c in class_names:
        pct = gt_inrange[c] / max(gt_counts[c], 1) * 100
        print(f"    {c:<12}  det={det_counts[c]:>5}   "
              f"gt_total={gt_counts[c]:>5}  "
              f"gt_in_range={gt_inrange[c]:>5} ({pct:.0f}%)")
    if all_scores:
        arr = np.array(all_scores)
        print(f"  Score  min={arr.min():.3f}  p50={np.percentile(arr,50):.3f}  "
              f"p90={np.percentile(arr,90):.3f}  max={arr.max():.3f}")
    else:
        print(f"  *** ZERO detections above score_thr={score_thr} -- "
              f"try --score-thr 0.01 ***")
    print("----------------------------------------------------------------\n")


def _print_results(summary: dict):
    """Print nuScenes-style results table matching the paper format."""
    class_names = summary["class_names"]
    n_gt        = summary["n_gt"]
    per_ap      = summary["per_class_ap"]
    D           = summary["dist_thresholds"]
    tp_errors   = summary["tp_errors"]

    # Header
    dist_hdr = "  ".join(f"AP@{d:.1f}m" for d in D)
    header   = f"{'Class':<14} {'N_GT':>6}    {dist_hdr}    {'mAP':>7}"
    sep      = "-" * (len(header) + 4)

    print()
    print("=" * (len(header) + 4))
    print("  nuScenes Detection Metrics  (centre-distance, D={0.5,1,2,4} m)")
    print("=" * (len(header) + 4))
    print(header)
    print(sep)

    for cls in class_names:
        aps = [per_ap[cls].get(d, 0.0) * 100 for d in D]
        mean_ap = summary["class_ap"][cls] * 100
        ap_cols = "  ".join(f"{a:>7.2f}" for a in aps)
        print(f"{cls:<14} {n_gt[cls]:>6}    {ap_cols}    {mean_ap:>7.2f}")

    print(sep)
    mean_cols = "  ".join(
        f"{summary['mAP_per_dist'].get(d, 0.0)*100:>7.2f}" for d in D
    )
    print(f"{'Mean':<14} {'':>6}    {mean_cols}    {summary['mAP']*100:>7.2f}")
    print("=" * (len(header) + 4))

    # TP metrics per class
    print()
    print(f"  True Positive Metrics  (matched at d <= {summary.get('tp_match_dist', 2.0):.0f} m)")
    print(f"  {'Class':<14}  {'ATE (m)':>8}  {'ASE (1-IoU)':>11}  {'AOE (rad)':>10}  n_matched")
    print(f"  {'-'*14}  {'-'*8}  {'-'*11}  {'-'*10}  {'-'*9}")
    for cls in class_names:
        ate_v = tp_errors[cls]['ATE']
        ase_v = tp_errors[cls]['ASE']
        aoe_v = tp_errors[cls]['AOE']
        n_m   = len(ate_v)
        ate   = float(np.mean(ate_v)) if ate_v else float('nan')
        ase   = float(np.mean(ase_v)) if ase_v else float('nan')
        aoe   = float(np.mean(aoe_v)) if aoe_v else float('nan')
        print(f"  {cls:<14}  {ate:>8.3f}  {ase:>11.3f}  {aoe:>10.3f}  {n_m:>9}")

    print()
    mAP  = summary["mAP"]  * 100
    mATE = summary["mATE"]
    mASE = summary["mASE"]
    mAOE = summary["mAOE"]
    NDS  = summary["NDS"]

    print(f"  mAP  (mean over D and classes)  :  {mAP:6.2f}%")
    print(f"  mATE (translation error)         :  {mATE:.3f} m")
    print(f"  mASE (scale error,  1-IoU)       :  {mASE:.3f}")
    print(f"  mAOE (orientation error)         :  {mAOE:.3f} rad")
    print(f"  NDS  (partial, 3 TP metrics)*    :  {NDS:.4f}")
    print()
    print("  * Full NDS includes mAVE + mAAE (not predicted by this model).")
    print("    Missing terms contribute 0 to the score (worst-case assumed).")
    print()


def _devkit_eval(all_dets, all_gts, dataset, cfg):
    """
    Run the official nuScenes DetectionEval if nuscenes-devkit is installed.
    pip install nuscenes-devkit
    """
    try:
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        print("nuscenes-devkit not installed — skipping official eval.")
        print("  pip install nuscenes-devkit")
        return

    import json, tempfile

    DEVKIT_NAME = {"Car": "car", "Bus": "bus", "Motorcycle": "motorcycle",
                   "Pedestrian": "pedestrian.adult"}

    results = {}
    for i, (frame_dets, _) in enumerate(zip(all_dets, all_gts)):
        token = dataset.get_token(i)
        preds = []
        for d in frame_dets:
            cx, cy, cz = d["box"][:3]
            l, w, h    = d["box"][3], d["box"][4], d["box"][5]
            yaw        = d["box"][6]
            q = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]   # [w,x,y,z] around Z
            preds.append({
                "sample_token":    token,
                "translation":     [float(cx), float(cy), float(cz)],
                "size":            [float(w), float(l), float(h)],  # nuScenes: W,L,H
                "rotation":        q,
                "velocity":        [0.0, 0.0],
                "detection_name":  DEVKIT_NAME.get(cfg.model.class_names[d["class_id"]], "car"),
                "detection_score": float(d["score"]),
                "attribute_name":  "vehicle.parked",
            })
        results[token] = preds

    submission = {"meta": {"use_camera": False, "use_lidar": True,
                            "use_radar": False, "use_map": False,
                            "use_external": False},
                  "results": results}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(submission, f)
        sub_path = f.name

    print(f"\nRunning official DetectionEval (submission saved to {sub_path}) ...")
    nusc = NuScenes(version="v1.0-trainval",
                    dataroot="lidar_detection/data/nuscenes_full/v1.0-trainval_meta",
                    verbose=False)
    eval_cfg = config_factory("detection_cvpr_2019")
    ev = DetectionEval(nusc, config=eval_cfg, result_path=sub_path,
                       eval_set="val", output_dir=tempfile.mkdtemp(), verbose=True)
    ev.main(render_curves=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",   type=str, default=DEFAULT_DATAROOT,
                        help="path to the nuScenes trainval root "
                             f"(default: {DEFAULT_DATAROOT})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path (default: checkpoint_nuscenes{_center}.pth)")
    parser.add_argument("--blobs",      type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--sweeps",     type=int, default=1,
                        help="LiDAR sweeps for val inference (default 1 for speed)")
    parser.add_argument("--n",          type=int, default=None)
    parser.add_argument("--score-thr",  type=float, default=0.05)
    parser.add_argument("--nms-thr",    type=float, default=0.5)
    parser.add_argument("--devkit",     action="store_true",
                        help="also run official nuscenes-devkit DetectionEval")
    parser.add_argument("--center",     action="store_true",
                        help="load a CenterPointPillars checkpoint")
    args = parser.parse_args()

    dataroot  = args.dataroot
    meta_root = f"{dataroot}/v1.0-trainval_meta"
    blob_tmpl = f"{dataroot}/v1.0-trainval{{:02d}}_blobs"

    # ── dataset ───────────────────────────────────────────────────────────────
    data_roots = []
    for b in args.blobs:
        p = blob_tmpl.format(b)
        if os.path.isdir(p):
            data_roots.append(p)
        else:
            print(f"  skipping missing blob: {p}")

    if not data_roots:
        print("ERROR: no blob directories found.")
        sys.exit(1)

    print("Loading dataset ...")
    dataset = NuScenesDataset(
        meta_root=meta_root, data_roots=data_roots, version=VERSION, split="val",
        num_sweeps=args.sweeps)
    if len(dataset) == 0:
        print("ERROR: no frames found. Check meta_root and data_roots.")
        sys.exit(1)

    n_eval = min(args.n, len(dataset)) if args.n else len(dataset)

    ckpt_path = args.checkpoint or ("checkpoint_nuscenes_center.pth" if args.center
                                     else "checkpoint_nuscenes.pth")

    # ── model ─────────────────────────────────────────────────────────────────
    cfg    = PipelineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = (CenterPointPillars(cfg) if args.center else PointPillars(cfg)).to(device)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    saved_epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    n_params    = sum(p.numel() for p in model.parameters())
    head_label  = "CenterPoint" if args.center else "Anchor"
    print(f"Loaded  : {ckpt_path}  (epoch {saved_epoch}, {head_label} head)")
    print(f"Device  : {device}  |  Params: {n_params/1e6:.2f}M")
    print(f"Dataset : {len(dataset)} frames  |  Evaluating: {n_eval}")
    print(f"Settings: score_thr={args.score_thr}  nms_thr={args.nms_thr}")
    print()

    # ── inference ─────────────────────────────────────────────────────────────
    evaluator = Evaluator(model, cfg, device)
    print("Running inference ...")
    t0 = time.time()
    all_dets, all_gts = evaluator.run(
        dataset,
        n_samples=n_eval,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
    )
    elapsed = time.time() - t0
    print(f"Done: {elapsed:.1f}s  ({n_eval / elapsed:.1f} frames/s)\n")

    _print_diagnostics(all_dets, all_gts, cfg, args.score_thr)

    # ── nuScenes metrics ──────────────────────────────────────────────────────
    print("Computing nuScenes metrics ...")
    summary = compute_nuscenes_metrics(
        all_dets,
        all_gts,
        class_names=cfg.model.class_names,
    )
    summary['tp_match_dist'] = 2.0

    _print_results(summary)

    if args.devkit:
        _devkit_eval(all_dets, all_gts, dataset, cfg)


if __name__ == "__main__":
    main()
