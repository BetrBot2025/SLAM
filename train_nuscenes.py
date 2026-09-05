"""
Train PointPillars on nuScenes trainval data.

Key improvements vs. earlier version:
  - All 10 blobs by default (use --blobs to restrict)
  - Steps = one full pass over training data (len(dataset) // batch_size)
  - 20 epochs default (standard for full nuScenes)
  - CBGS enabled by default (--no-cbgs to disable)
  - GT-sampling augmentation when --gt-database is provided
  - 10-sweep accumulation by default (--sweeps to change)

Usage:
    python train_nuscenes.py                                   # full run
    python train_nuscenes.py --blobs 1 2 3                    # subset of blobs
    python train_nuscenes.py --epochs 40 --sweeps 1           # faster experiment
    python train_nuscenes.py --gt-database gt_database.pkl    # with GT sampling
    python train_nuscenes.py --resume checkpoint_nuscenes.pth
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, ".")
from lidar_detection.config import PipelineConfig
from lidar_detection.data.nuscenes_loader import NuScenesDataset
from lidar_detection.training.trainer import Trainer
from lidar_detection.training.metrics_logger import MetricsLogger

DEFAULT_DATAROOT = "lidar_detection/data/nuscenes_full"
VERSION          = "v1.0-trainval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot",    type=str, default=DEFAULT_DATAROOT,
                        help="path to the nuScenes trainval root "
                             f"(default: {DEFAULT_DATAROOT})")
    parser.add_argument("--epochs",      type=int, default=20,
                        help="total training epochs (default 20 = standard nuScenes)")
    parser.add_argument("--steps",       type=int, default=None,
                        help="optimizer steps per epoch; default = len(dataset)//batch_size "
                             "(one full pass over the training data)")
    parser.add_argument("--batch-size",  type=int, default=1,
                        help="samples per optimizer step (default 1 for memory safety)")
    parser.add_argument("--sweeps",      type=int, default=10,
                        help="number of LiDAR sweeps to accumulate (default 10)")
    parser.add_argument("--blobs",       type=int, nargs="+", default=list(range(1, 11)),
                        help="blob numbers to use (default: all 10)")
    parser.add_argument("--out",         type=str, default=None,
                        help="checkpoint path (default: checkpoint_nuscenes{_center}.pth)")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--log",         type=str, default="training_log.jsonl")
    parser.add_argument("--gt-database", type=str, default=None,
                        help="path to gt_database.pkl (build with scripts/build_gt_database.py)")
    parser.add_argument("--no-cbgs",     action="store_true",
                        help="disable Class-Balanced Group Sampling")
    parser.add_argument("--center",      action="store_true",
                        help="use CenterPoint heatmap head instead of anchor head")
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

    dataset = NuScenesDataset(
        meta_root  = meta_root,
        data_roots = data_roots,
        version    = VERSION,
        split      = "train",
        num_sweeps = args.sweeps,
    )
    if len(dataset) == 0:
        print("ERROR: no frames found.")
        sys.exit(1)

    out_path   = args.out or ("checkpoint_nuscenes_center.pth" if args.center
                               else "checkpoint_nuscenes.pth")
    use_cbgs   = not args.no_cbgs
    batch_size = args.batch_size
    steps      = args.steps or max(1, len(dataset) // batch_size)

    print(f"\nDataset : {len(dataset)} training frames  |  sweeps={args.sweeps}")
    print(f"CBGS    : {'ON' if use_cbgs else 'OFF'}")
    print(f"Steps   : {steps}/epoch  x  batch_size={batch_size}"
          f"  = {steps * batch_size} frames/epoch")

    # ── GT database ───────────────────────────────────────────────────────────
    gt_database = None
    if args.gt_database:
        if os.path.exists(args.gt_database):
            print(f"GT-DB   : loading {args.gt_database} ...")
            with open(args.gt_database, "rb") as f:
                gt_database = pickle.load(f)
            for cn, entries in gt_database.items():
                print(f"  {cn}: {len(entries)} entries")
        else:
            print(f"WARNING: --gt-database not found: {args.gt_database}  (skipping)")

    # ── logger ────────────────────────────────────────────────────────────────
    logger = MetricsLogger(args.log)
    logger.log_config(
        epochs          = args.epochs,
        steps_per_epoch = steps,
        batch_size      = batch_size,
        dataset_size    = len(dataset),
        checkpoint_out  = out_path,
        blobs           = args.blobs,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    cfg                  = PipelineConfig()
    cfg.train.batch_size = batch_size
    total_steps          = args.epochs * steps
    trainer              = Trainer(cfg, logger=logger, total_steps=total_steps,
                                   gt_database=gt_database,
                                   use_center=args.center)

    resume_path = args.resume or out_path
    if os.path.exists(resume_path):
        start_epoch = trainer.load(resume_path, steps_per_epoch=steps)
    else:
        start_epoch = 1
        print("No checkpoint found -- starting from scratch.")

    remaining = args.epochs - (start_epoch - 1)
    print(f"\nEpochs  : {start_epoch} -> {args.epochs}  ({remaining} remaining)")
    print(f"Output  : {out_path}  (saved on completion or Ctrl+C)\n")

    if remaining <= 0:
        print("Already at target epoch. Delete checkpoint to retrain.")
        return

    # ── training loop ─────────────────────────────────────────────────────────
    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch

            epoch_indices = dataset.get_cbgs_indices() if use_cbgs else None
            trainer.train_epoch(epoch, steps, dataset=dataset,
                                cbgs_indices=epoch_indices)

        print(f"\nTraining complete -- saving checkpoint ...")
        trainer.save(out_path, epoch=current_epoch)
        logger.log_done()
        print("Done.")

    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {current_epoch} -- saving checkpoint ...")
        trainer.save(out_path, epoch=current_epoch)
        logger.log_done()
        print("Done.")


if __name__ == "__main__":
    main()
