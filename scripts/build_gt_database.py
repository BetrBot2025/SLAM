"""
Build the GT (copy-paste) database from the nuScenes training set.

For each GT box in the training split, this script:
  1. Extracts all LiDAR points that fall inside the box
  2. Converts those points to the box-local frame (center at origin, yaw-aligned)
  3. Saves the result to gt_database.pkl

Run once before training (takes ~30-60 min for 10+ blobs):
    python scripts/build_gt_database.py
    python scripts/build_gt_database.py --blobs 1 2 3  # specify blobs

Then pass the database to the trainer with --gt-database gt_database.pkl
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from lidar_detection.data.nuscenes_loader import NuScenesDataset, CLASS_NAMES

DATAROOT  = "lidar_detection/data/nuscenes_full"
META_ROOT = f"{DATAROOT}/v1.0-trainval_meta"
BLOB_TMPL = f"{DATAROOT}/v1.0-trainval{{:02d}}_blobs"
VERSION   = "v1.0-trainval"

# Min points inside a box to include it in the database
MIN_POINTS = 5
# Max DB entries per class (trim if larger, keeps RAM sane)
MAX_PER_CLASS = 5000


def _points_in_box(points: np.ndarray, box: dict) -> np.ndarray:
    cx, cy, cz = box["center"]
    l, w, h    = box["size"]
    yaw        = box["yaw"]
    dx, dy, dz = points[:,0]-cx, points[:,1]-cy, points[:,2]-cz
    c, s  = np.cos(-yaw), np.sin(-yaw)
    lx    = c*dx - s*dy
    ly    = s*dx + c*dy
    return (np.abs(lx) <= l/2) & (np.abs(ly) <= w/2) & (np.abs(dz) <= h/2)


def _to_local(points: np.ndarray, box: dict) -> np.ndarray:
    """Transform points to box-local frame (center at origin, yaw aligned)."""
    cx, cy, cz = box["center"]
    yaw        = box["yaw"]
    pts = points.copy()
    pts[:,0] -= cx
    pts[:,1] -= cy
    pts[:,2] -= cz
    c, s  = np.cos(-yaw), np.sin(-yaw)
    lx    = c*pts[:,0] - s*pts[:,1]
    ly    = s*pts[:,0] + c*pts[:,1]
    pts[:,0], pts[:,1] = lx, ly
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blobs",  type=int, nargs="+", default=list(range(1, 11)),
                        help="blob numbers to include (default: 1-10)")
    parser.add_argument("--out",    type=str, default="gt_database.pkl")
    parser.add_argument("--sweeps", type=int, default=1,
                        help="number of LiDAR sweeps (1 = keyframe only, faster build)")
    args = parser.parse_args()

    data_roots = []
    for b in args.blobs:
        p = BLOB_TMPL.format(b)
        if os.path.isdir(p):
            data_roots.append(p)
        else:
            print(f"  skipping missing blob: {p}")

    if not data_roots:
        print("ERROR: no blob directories found.")
        sys.exit(1)

    print(f"Loading dataset from {len(data_roots)} blob(s) ...")
    dataset = NuScenesDataset(
        meta_root  = META_ROOT,
        data_roots = data_roots,
        version    = VERSION,
        split      = "train",
        num_sweeps = args.sweeps,
    )
    print(f"  {len(dataset)} training frames\n")

    database: dict = {cn: [] for cn in CLASS_NAMES}
    t0 = time.time()

    for i in range(len(dataset)):
        if i % 500 == 0:
            elapsed = time.time() - t0
            fps = max(i, 1) / elapsed
            print(f"  [{i:5d}/{len(dataset)}]  {elapsed:.0f}s  ({fps:.1f} frames/s)")

        points, gt_boxes = dataset[i]

        for box in gt_boxes:
            cn = box.get("class_name", CLASS_NAMES[box["class_id"]])
            mask = _points_in_box(points, box)
            n_pts = int(mask.sum())
            if n_pts < MIN_POINTS:
                continue
            rel_pts = _to_local(points[mask], box)
            database[cn].append({"box": box, "rel_pts": rel_pts})

    # Trim to cap
    for cn in CLASS_NAMES:
        entries = database[cn]
        if len(entries) > MAX_PER_CLASS:
            idx = np.random.choice(len(entries), MAX_PER_CLASS, replace=False)
            database[cn] = [entries[j] for j in idx]

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    for cn in CLASS_NAMES:
        print(f"  {cn}: {len(database[cn])} GT entries")

    with open(args.out, "wb") as f:
        pickle.dump(database, f)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nSaved: {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
