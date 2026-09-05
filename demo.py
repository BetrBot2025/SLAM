"""
LiDAR 3D Object Detection — end-to-end demo.

Run:
    python demo.py                   # inference only (random weights)
    python demo.py --train 5         # train 5 epochs then infer
    python demo.py --save bev.png    # save BEV image
"""
import argparse
import sys
import numpy as np

sys.path.insert(0, ".")
from lidar_detection.config import PipelineConfig
from lidar_detection.data.generator import generate_scene
from lidar_detection.pipeline import DetectionPipeline
from lidar_detection.visualization.visualizer import draw_bev, draw_detection_summary


def main():
    parser = argparse.ArgumentParser(description="LiDAR 3D Detection Demo")
    parser.add_argument("--train", type=int, default=0, help="train N epochs before inference")
    parser.add_argument("--save", type=str, default=None, help="save BEV plot to file")
    parser.add_argument("--save-scene", type=str, default=None, help="save generated scene BEV to file")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PipelineConfig()

    # --- Optional training ---
    if args.train > 0:
        from lidar_detection.training.trainer import Trainer
        trainer = Trainer(cfg)
        trainer.train(epochs=args.train, steps_per_epoch=20)
        trainer.save("checkpoint.pth")
        ckpt = "checkpoint.pth"
    else:
        ckpt = None

    # --- Generate synthetic scene ---
    print(f"\nGenerating synthetic LiDAR scene (seed={args.seed})...")
    points, gt_boxes = generate_scene(n_cars=4, n_pedestrians=3, n_cyclists=2, seed=args.seed)
    print(f"  Points: {len(points):,}")
    print(f"  GT objects: {len(gt_boxes)} ({[g['class_name'] for g in gt_boxes]})")

    gt_for_vis = [
        {
            "box": [g["center"][0], g["center"][1], g["center"][2],
                    g["size"][0], g["size"][1], g["size"][2], g["yaw"]],
            "class_name": g["class_name"],
        }
        for g in gt_boxes
    ]

    print("\nVisualizing generated scene...")
    draw_bev(
        points,
        gt_boxes=gt_for_vis,
        pc_range=cfg.voxel.point_cloud_range,
        title=f"Generated Scene — {len(gt_boxes)} GT objects",
        save_path=args.save_scene,
    )

    # --- Run inference ---
    pipe = DetectionPipeline(cfg=cfg, checkpoint=ckpt)

    print("\nRunning inference...")
    detections = pipe.run(points, score_threshold=args.score_threshold)
    print(f"  Detections: {len(detections)}")

    # --- Summary ---
    draw_detection_summary(detections, gt_boxes)

    # --- Visualize ---
    print("\nGenerating BEV visualization...")
    draw_bev(
        points,
        gt_boxes=gt_for_vis,
        pred_boxes=detections,
        pc_range=cfg.voxel.point_cloud_range,
        title=f"LiDAR 3D Detection — {len(detections)} predictions | {len(gt_boxes)} GT",
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
