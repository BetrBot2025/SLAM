"""BEV and 3D box visualization using matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Optional, Tuple


CLASS_COLORS = {
    "Car": "#2196F3",
    "Bus": "#9C27B0",
    "Motorcycle": "#FF9800",
    "Pedestrian": "#4CAF50",
    "Unknown": "#9E9E9E",
}

GT_COLORS = {
    "Car": "#0D47A1",
    "Bus": "#4A148C",
    "Motorcycle": "#E65100",
    "Pedestrian": "#1B5E20",
}


def _box_corners_bev(box: list) -> np.ndarray:
    """Return 4 BEV corners for a box [x,y,z,l,w,h,yaw]."""
    x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2, w / 2
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    rot = np.array([[c, -s], [s, c]])
    return (rot @ local.T).T + np.array([x, y])


def draw_bev(
    points: np.ndarray,
    gt_boxes: Optional[List[Dict]] = None,
    pred_boxes: Optional[List[Dict]] = None,
    pc_range: List[float] = None,
    title: str = "Bird's Eye View",
    save_path: Optional[str] = None,
):
    if pc_range is None:
        pc_range = [0, -39.68, -3, 69.12, 39.68, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    # Points coloured by height
    z = points[:, 2]
    z_norm = np.clip((z - pc_range[2]) / (pc_range[5] - pc_range[2]), 0, 1)
    ax.scatter(points[:, 0], points[:, 1], c=z_norm, cmap="plasma",
               s=0.3, alpha=0.6, rasterized=True)

    def _draw_box(box_list, color, linestyle="-", label_prefix=""):
        for det in box_list:
            box = det["box"] if "box" in det else det
            if isinstance(box, dict):
                box_arr = box["center"] + box["size"] + [box["yaw"]]
                # reorder to x,y,z,l,w,h,yaw
                cx, cy, cz = box["center"]
                l, w, h = box["size"]
                yaw = box["yaw"]
                box = [cx, cy, cz, l, w, h, yaw]
            corners = _box_corners_bev(box)
            poly = plt.Polygon(corners, fill=False, edgecolor=color,
                               linewidth=1.5, linestyle=linestyle)
            ax.add_patch(poly)
            # Direction arrow (front of box)
            cx_b = np.mean(corners[:, 0])
            cy_b = np.mean(corners[:, 1])
            mid_front = (corners[0] + corners[1]) / 2
            ax.annotate("", xy=mid_front, xytext=(cx_b, cy_b),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
            # Label
            if "class_name" in det:
                sc = f" {det['score']:.2f}" if "score" in det else ""
                ax.text(cx_b, cy_b, det["class_name"] + sc, color=color,
                        fontsize=6, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))

    if gt_boxes:
        _draw_box(gt_boxes, color="#00FF88", linestyle="--")
    if pred_boxes:
        for det in pred_boxes:
            color = CLASS_COLORS.get(det.get("class_name", ""), "#9E9E9E")
            _draw_box([det], color=color, linestyle="-")

    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_xlabel("X (forward) [m]", color="white")
    ax.set_ylabel("Y (left) [m]", color="white")
    ax.tick_params(colors="white")
    ax.set_title(title, color="white", fontsize=14)

    # Legend
    legend_handles = [mpatches.Patch(color="#00FF88", label="GT boxes (dashed)")]
    for name, color in CLASS_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=f"Pred: {name}"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _box_corners_3d(box: list) -> np.ndarray:
    """8 corners for a box [x,y,z,l,w,h,yaw] (z = box centre height)."""
    x, y, z, l, w, h, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw, hh = l / 2, w / 2, h / 2
    local = np.array([
        [ hl,  hw, -hh], [ hl, -hw, -hh], [-hl, -hw, -hh], [-hl,  hw, -hh],
        [ hl,  hw,  hh], [ hl, -hw,  hh], [-hl, -hw,  hh], [-hl,  hw,  hh],
    ])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.array([x, y, z])


_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]


def draw_3d_map(
    objects: List[Dict],
    trajectory_est: Optional[np.ndarray] = None,   # (N, 3)
    trajectory_gt: Optional[np.ndarray] = None,     # (N, 3)
    title: str = "3D Object Map",
    save_path: Optional[str] = None,
):
    """WP10: Fig. 7/9-style 3D map -- object landmarks (mesh where available,
    else a wireframe box placeholder) plus the estimated/GT trajectories.

    objects: [{'center':[x,y,z], 'size':[l,w,h], 'yaw':float,
               'class_name':str, 'mesh': Optional[(verts (M,3), faces (K,3))]}]
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for obj in objects:
        color = CLASS_COLORS.get(obj.get("class_name", ""), "#9E9E9E")
        mesh = obj.get("mesh")
        if mesh is not None:
            verts, faces = mesh
            cx, cy, cz = obj["center"]
            world_verts = verts + np.array([cx, cy, cz])
            poly = Poly3DCollection(world_verts[faces], alpha=0.6, facecolor=color, edgecolor="none")
            ax.add_collection3d(poly)
        else:
            box = list(obj["center"]) + list(obj["size"]) + [obj.get("yaw", 0.0)]
            corners = _box_corners_3d(box)
            for i, j in _BOX_EDGES:
                ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=1.2)

    if trajectory_gt is not None and len(trajectory_gt) > 0:
        ax.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], trajectory_gt[:, 2],
                color="#00FF88", linewidth=1.5, linestyle="--", label="GT")
    if trajectory_est is not None and len(trajectory_est) > 0:
        ax.plot(trajectory_est[:, 0], trajectory_est[:, 1], trajectory_est[:, 2],
                color="#2196F3", linewidth=1.5, label="Estimate")

    handles = [mpatches.Patch(color=c, label=n) for n, c in CLASS_COLORS.items() if n != "Unknown"]
    if trajectory_gt is not None:
        handles.append(plt.Line2D([0], [0], color="#00FF88", linestyle="--", label="GT trajectory"))
    if trajectory_est is not None:
        handles.append(plt.Line2D([0], [0], color="#2196F3", label="Est. trajectory"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(title)

    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]
    if all(s > 0 for s in spans):
        ax.set_box_aspect(spans)   # equal data scale on all 3 axes, not just equal pixel extent

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def draw_detection_summary(detections: List[Dict], gt_boxes: List[Dict]):
    """Print a concise detection summary table."""
    from collections import Counter
    pred_counts = Counter(d["class_name"] for d in detections)
    gt_counts = Counter(d["class_name"] for d in gt_boxes)

    print("\n" + "=" * 50)
    print(f"{'Class':<15} {'GT':>6} {'Pred':>6} {'Avg Score':>10}")
    print("-" * 50)
    all_classes = sorted(set(list(pred_counts.keys()) + list(gt_counts.keys())))
    for cls in all_classes:
        gt_n = gt_counts.get(cls, 0)
        pd_n = pred_counts.get(cls, 0)
        avg_sc = np.mean([d["score"] for d in detections if d["class_name"] == cls]) if pd_n > 0 else 0.0
        print(f"{cls:<15} {gt_n:>6} {pd_n:>6} {avg_sc:>10.3f}")
    print("=" * 50)
