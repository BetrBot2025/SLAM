"""
Render 100 training scenes as a self-contained animated HTML page.

Usage:
    python visualize_training_scenes.py                   # writes scenes.html
    python visualize_training_scenes.py --out my.html     # custom output path
    python visualize_training_scenes.py --epoch 2         # use epoch-2 seeds
    python visualize_training_scenes.py --n 50            # only 50 scenes
"""
import argparse
import base64
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, ".")
from lidar_detection.config import PipelineConfig
from lidar_detection.data.generator import generate_scene


CLASS_COLORS = {
    "Car":        "#2196F3",
    "Pedestrian": "#4CAF50",
    "Cyclist":    "#FF9800",
}
GT_COLOR = "#00FF88"


def _box_corners_bev(x, y, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2, w / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ local.T).T + np.array([x, y])


def render_scene_to_base64(points, gt_boxes, pc_range, scene_label):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    z = points[:, 2]
    z_norm = np.clip((z - pc_range[2]) / (pc_range[5] - pc_range[2]), 0, 1)
    ax.scatter(points[:, 0], points[:, 1], c=z_norm, cmap="plasma",
               s=0.3, alpha=0.6, rasterized=True)

    for g in gt_boxes:
        cx, cy = g["center"][0], g["center"][1]
        l, w = g["size"][0], g["size"][1]
        corners = _box_corners_bev(cx, cy, l, w, g["yaw"])
        poly = plt.Polygon(corners, fill=False, edgecolor=GT_COLOR,
                           linewidth=1.5, linestyle="--")
        ax.add_patch(poly)
        mid_front = (corners[0] + corners[1]) / 2
        ax.annotate("", xy=mid_front, xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=GT_COLOR, lw=1.2))
        color = CLASS_COLORS.get(g["class_name"], "#9E9E9E")
        ax.text(cx, cy, g["class_name"], color=color, fontsize=6,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6))

    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_xlabel("X (forward) [m]", color="white", fontsize=9)
    ax.set_ylabel("Y (left) [m]", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.set_title(scene_label, color="white", fontsize=11)

    legend_handles = [mpatches.Patch(color=GT_COLOR, label="GT box")]
    for name, color in CLASS_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=name))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Scene Animation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f0f23;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px 16px;
    min-height: 100vh;
  }}
  h1 {{ font-size: 1.3rem; margin-bottom: 18px; color: #90caf9; letter-spacing: 0.04em; }}
  #frame-img {{
    border: 1px solid #333;
    border-radius: 6px;
    max-width: 860px;
    width: 100%;
  }}
  #controls {{
    margin-top: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    width: 100%;
    max-width: 860px;
  }}
  #scrubber {{
    width: 100%;
    accent-color: #2196F3;
    cursor: pointer;
  }}
  .btn-row {{
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    justify-content: center;
  }}
  button {{
    background: #1e3a5f;
    color: #90caf9;
    border: 1px solid #2196F3;
    border-radius: 4px;
    padding: 6px 16px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.15s;
  }}
  button:hover {{ background: #2196F3; color: #fff; }}
  #info {{
    font-size: 0.85rem;
    color: #aaa;
    min-width: 160px;
    text-align: center;
  }}
  .speed-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    color: #aaa;
  }}
  #fps-slider {{ accent-color: #FF9800; width: 100px; }}
</style>
</head>
<body>
<h1>Training Scenes — {n_scenes} frames (epoch {epoch})</h1>
<img id="frame-img" src="" alt="scene frame">
<div id="controls">
  <input id="scrubber" type="range" min="0" max="{max_idx}" value="0" step="1">
  <div class="btn-row">
    <button id="btn-prev">&#9664; Prev</button>
    <button id="btn-play">&#9654; Play</button>
    <button id="btn-next">Next &#9654;</button>
    <span id="info">Scene 1 / {n_scenes}</span>
    <div class="speed-row">
      <label for="fps-slider">FPS:</label>
      <input id="fps-slider" type="range" min="1" max="24" value="6" step="1">
      <span id="fps-label">6</span>
    </div>
  </div>
</div>
<script>
const frames = {frames_json};
const meta   = {meta_json};
let cur = 0, timer = null, fps = 6;

const img      = document.getElementById('frame-img');
const scrubber = document.getElementById('scrubber');
const info     = document.getElementById('info');
const btnPlay  = document.getElementById('btn-play');
const fpsSlider = document.getElementById('fps-slider');
const fpsLabel  = document.getElementById('fps-label');

function show(i) {{
  cur = i;
  img.src = 'data:image/png;base64,' + frames[i];
  scrubber.value = i;
  const m = meta[i];
  info.textContent = `Scene ${{i+1}} / ${{frames.length}}  |  seed ${{m.seed}}  |  ${{m.n_pts.toLocaleString()}} pts  |  ${{m.n_gt}} GT`;
}}

function step() {{ show((cur + 1) % frames.length); }}

function play() {{
  if (timer) {{ clearInterval(timer); timer = null; btnPlay.textContent = '▶ Play'; }}
  else       {{ timer = setInterval(step, 1000 / fps); btnPlay.textContent = '⏸ Pause'; }}
}}

document.getElementById('btn-play').onclick  = play;
document.getElementById('btn-prev').onclick  = () => {{ show((cur - 1 + frames.length) % frames.length); }};
document.getElementById('btn-next').onclick  = () => {{ show((cur + 1) % frames.length); }};
scrubber.oninput = () => show(parseInt(scrubber.value));

fpsSlider.oninput = () => {{
  fps = parseInt(fpsSlider.value);
  fpsLabel.textContent = fps;
  if (timer) {{ clearInterval(timer); timer = setInterval(step, 1000 / fps); }}
}};

show(0);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="scenes.html",   help="output HTML file")
    parser.add_argument("--epoch", type=int, default=1,     help="training epoch to simulate (affects seeds)")
    parser.add_argument("--n",     type=int, default=100,   help="number of scenes (max 1000)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    pc_range = cfg.voxel.point_cloud_range

    frames_b64 = []
    meta = []

    print(f"Rendering {args.n} scenes (epoch={args.epoch})...")
    for step in range(args.n):
        seed = args.epoch * 1000 + step
        points, gt_boxes = generate_scene(
            n_cars=4, n_pedestrians=3, n_cyclists=2, seed=seed
        )
        label = f"Epoch {args.epoch}  Step {step:03d}  (seed {seed})  —  {len(gt_boxes)} GT objects"
        b64 = render_scene_to_base64(points, gt_boxes, pc_range, label)
        frames_b64.append(b64)
        meta.append({"seed": seed, "n_pts": len(points), "n_gt": len(gt_boxes)})

        if (step + 1) % 10 == 0:
            print(f"  {step + 1}/{args.n}")

    html = HTML_TEMPLATE.format(
        n_scenes=args.n,
        epoch=args.epoch,
        max_idx=args.n - 1,
        frames_json=json.dumps(frames_b64),
        meta_json=json.dumps(meta),
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved: {args.out}  ({len(html) / 1024 / 1024:.1f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
