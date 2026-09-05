# LiDAR 3D Object Detection Pipeline

A PointPillars-style perception pipeline for 3D object detection from LiDAR point clouds. Built with PyTorch and NumPy — no specialized LiDAR libraries required.

Two detection heads are available:
- **Anchor head** — original PointPillars anchor-based detection (Lang et al., 2019)
- **CenterPoint head** — heatmap-based center detection (Yin et al., CVPR 2021) — drop-in replacement, ~4 mAP gain on nuScenes

## Architecture

### Shared encoder (both heads)

```
Raw LiDAR points (N, 4)
        │
        ▼
  [Preprocessing]
  • ROI crop to detection range
  • RANSAC ground removal
  • Voxelization → pillars (V, P, 9)
        │
        ▼
  [Pillar Feature Net]
  • Pointwise MLP (9 → 64)
  • Max-pool per pillar → (V, 64)
  • Scatter to BEV canvas (512, 512, 64)
        │
        ▼
  [2D CNN Backbone]
  • Block1(S=2, 4 layers, 64ch)
  • Block2(S=2, 6 layers, 128ch)
  • Block3(S=2, 6 layers, 256ch)
  • FPN upsample + concat → (256, 256, 384)
```

### Anchor head (default)

```
  [Detection Head]
  • Class scores  (H, W, A × C)
  • Box regression (H, W, A × 7)  ← dx,dy,dz,dl,dw,dh,sin(Δyaw)
  • Direction bins (H, W, A × 2)
        │
        ▼
  [Post-processing]
  • Anchor decoding  • Score filter  • BEV NMS
```

### CenterPoint head (`--center`)

```
  [CenterPoint Head]
  Shared 3×3 conv → per-task branches (each: Conv+GN+ReLU → Conv):
  • Heatmap  (K channels) — Gaussian peak per GT centre
  • Offset   (2)          — sub-voxel (dx, dy) within feature cell
  • Height   (1)          — z-centre in metres
  • Size     (3)          — log(l), log(w), log(h)
  • Rotation (2)          — sin(yaw), cos(yaw)
        │
        ▼
  [Post-processing]
  • 3×3 max-pool NMS  • Top-K peaks per class  • Score filter
```

**Supported classes:** Car, Bus, Motorcycle, Pedestrian  
**Detection range:** ±51.2 m in X and Y  
**Model size:** a few million parameters (anchor head slightly larger than CenterPoint; exact count scales with `num_classes`)

## Requirements

Python 3.14 with the packages already in `.venv`:

| Package | Version |
|---|---|
| torch | 2.12.0 |
| numpy | 2.4.6 |
| matplotlib | 3.10.9 |
| opencv-python | 4.13.0.92 |
| scikit-learn | 1.8.0 |

## Setup

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Or run scripts directly with:
.venv/Scripts/python.exe <script>
```

## Running the Demo

### Inference with random weights

Generates a synthetic LiDAR scene and runs the model. Detections will be absent since the model is untrained — but the full pipeline executes and a BEV visualization is saved.

```bash
python demo.py
```

### Save BEV visualization to file

```bash
python demo.py --save output.png
```

### Train then infer

Trains on synthetic data for N epochs, saves a checkpoint, then runs inference on a new scene.

```bash
python demo.py --train 80 --save result.png
```

For a quick smoke test (faster, lower quality):

```bash
python demo.py --train 5 --save result.png
```

### Full option reference

```
python demo.py [--train N] [--save PATH] [--score-threshold T] [--seed S]

  --train N            Train for N epochs before inference (default: 0)
  --save PATH          Save BEV plot to file instead of displaying (default: show)
  --score-threshold T  Detection confidence threshold 0–1 (default: 0.3)
  --seed S             Random seed for synthetic scene (default: 42)
```

## Using as a Library

### Single-frame inference

```python
import numpy as np
from lidar_detection.pipeline import DetectionPipeline

pipe = DetectionPipeline(checkpoint="checkpoint.pth")

# points: (N, 4) float32 — x, y, z, intensity
points = np.load("frame.npy")
detections = pipe.run(points, score_threshold=0.3)

for det in detections:
    print(det["class_name"], det["score"], det["box"])
    # box = [x, y, z, length, width, height, yaw]
```

### Batch inference

```python
results = pipe.run_batch([points_a, points_b, points_c])
```

### Training from scratch

```python
from lidar_detection.config import PipelineConfig
from lidar_detection.training.trainer import Trainer

cfg = PipelineConfig()

# Anchor head (default)
trainer = Trainer(cfg)
trainer.train(epochs=80, steps_per_epoch=50)
trainer.save("checkpoint.pth")

# CenterPoint head
trainer = Trainer(cfg, use_center=True)
trainer.train(epochs=80, steps_per_epoch=50)
trainer.save("checkpoint_center.pth")
```

### Custom configuration

```python
from lidar_detection.config import PipelineConfig, VoxelConfig, TrainConfig

cfg = PipelineConfig()
cfg.voxel.point_cloud_range = [0, -50, -3, 70, 50, 1]  # wider FOV
cfg.train.lr = 1e-4
cfg.train.epochs = 120
```

## nuScenes Training

### Anchor head (standard PointPillars)

```bash
python train_nuscenes.py                          # 20 epochs, all 10 blobs, CBGS on
python train_nuscenes.py --epochs 40 --sweeps 1  # faster experiment
python train_nuscenes.py --gt-database gt_database.pkl  # with GT-sampling augmentation
python train_nuscenes.py --resume checkpoint_nuscenes.pth
```

### CenterPoint head

```bash
python train_nuscenes.py --center                 # saves checkpoint_nuscenes_center.pth
python train_nuscenes.py --center --epochs 20 --gt-database gt_database.pkl
```

### Validation

```bash
python validate_nuscenes.py                       # anchor checkpoint
python validate_nuscenes.py --center              # CenterPoint checkpoint
python validate_nuscenes.py --n 500 --score-thr 0.05
```

Reports official nuScenes metrics: AP at D = {0.5, 1, 2, 4} m, mAP, mATE, mASE, mAOE, NDS.

### Build GT-sampling database (optional, +1–2 mAP)

```bash
python scripts/build_gt_database.py
# produces gt_database.pkl (~5000 entries per class)
```

## Visualization & Monitoring Tools

### Live training dashboard

Streams loss curves from a running training job's JSONL log in the browser.

```bash
python train_nuscenes.py                 # start training (writes training_log.jsonl)
python training_dashboard.py              # in a second terminal
python training_dashboard.py --log my_run.jsonl --port 5501
```

### Interactive nuScenes BEV viewer

Browse dataset frames in a pan/zoom BEV view served locally.

```bash
python visualize_nuscenes_web.py
python visualize_nuscenes_web.py --root lidar_detection/data/nuScenes_mini --port 5500
```

Controls: scroll to zoom, drag to pan, `R` to reset, arrow keys to step frames, `Space` to play/pause.

### Synthetic scene animation export

Renders a batch of synthetic training scenes to a single self-contained animated HTML file.

```bash
python visualize_training_scenes.py                # writes scenes.html
python visualize_training_scenes.py --out my.html --n 50 --epoch 2
```

## Project Structure

```
lidar_detection/
├── config.py                  # All hyperparameters (voxel, model, anchor, training)
├── pipeline.py                # High-level inference API
├── data/
│   ├── generator.py           # Synthetic point cloud + GT box generator
│   └── nuscenes_loader.py     # nuScenes dataset with multi-sweep + CBGS
├── preprocessing/
│   ├── voxelizer.py           # Pillarization + D=9 feature augmentation
│   └── filters.py             # Ground removal (RANSAC), ROI crop
├── models/
│   ├── pointpillars.py        # PointPillars + CenterPointPillars model classes
│   ├── pillar_feature_net.py  # Per-pillar MLP + max-pool (GroupNorm)
│   ├── backbone.py            # BEV 2D CNN + FPN neck (paper-exact layer counts)
│   ├── detection_head.py      # Anchor cls / reg / direction heads
│   └── center_head.py         # CenterPoint heatmap + regression heads
├── detection/
│   ├── anchor.py              # 3D anchor grid generation
│   ├── nms.py                 # BEV NMS
│   ├── postprocess.py         # Anchor box decoding + score filtering
│   └── center_postprocess.py  # Heatmap peak extraction + box decoding
├── training/
│   ├── trainer.py             # Training loop (anchor & center), AMP, CBGS
│   ├── loss.py                # Focal loss + Smooth-L1 (anchor head)
│   ├── center_loss.py         # Gaussian focal loss + masked L1 (center head)
│   └── augmentation.py        # Random flip, rotation, scale, GT-sampling
├── evaluation/
│   ├── evaluator.py           # Runs model over dataset (both head types)
│   └── metrics.py             # Official nuScenes AP / mAP / NDS
└── visualization/
    └── visualizer.py          # BEV matplotlib plots with GT/pred overlay
demo.py                        # End-to-end runnable demo (synthetic data)
train_nuscenes.py              # nuScenes training script
validate_nuscenes.py           # nuScenes validation + metrics
training_dashboard.py          # Live loss-curve dashboard (reads training_log*.jsonl)
visualize_nuscenes_web.py      # Interactive browser BEV viewer for nuScenes frames
visualize_training_scenes.py   # Exports animated synthetic scenes to HTML
scripts/
└── build_gt_database.py       # Builds GT-sampling database from training set
planning/
└── DSPSLAMpp_Integration_Plan.tex  # Design doc: extending this detector into a
                                     # DSP-SLAM++-style multi-class object SLAM pipeline
```

## Expected Training Behaviour

### Synthetic data (demo)

Loss should decrease from ~0.54 toward ~0.20–0.30 over 80 epochs.
Detections at `--score-threshold 0.3` typically appear after ~20–30 epochs.

```
Epoch   1 | loss 0.5404  (cls 0.4800  reg 0.0510  dir 0.0094)
Epoch   2 | loss 0.5358
...
Epoch  80 | loss ~0.22
```

### nuScenes — anchor head

```
Epoch   1 | loss 1.2340  (cls 0.9100  reg 0.2800  dir 0.0440)  lr 0.002000
...
Epoch  20 | loss ~0.35
```

### nuScenes — CenterPoint head

```
Epoch   1 | loss 2.1800  (hm 1.9200  reg 0.2600)  lr 0.002000
...
Epoch  20 | loss ~0.48
```

The heatmap loss starts higher than the anchor focal loss because it covers every pixel (not just positive anchors) but converges to comparable validation mAP — typically +3–4 mAP vs the anchor head on nuScenes.

## Extending to Real Data (KITTI)

Point clouds from KITTI are stored as `(N, 4)` float32 binary files. Load and pass directly:

```python
import numpy as np
points = np.fromfile("velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
detections = pipe.run(points)
```

Adjust `cfg.voxel.point_cloud_range` to match the sensor's coordinate frame if needed.
