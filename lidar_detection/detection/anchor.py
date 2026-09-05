"""3D anchor generation on the BEV feature grid."""
import numpy as np
from ..config import PipelineConfig


def generate_anchors(cfg: PipelineConfig, feature_map_size: list) -> np.ndarray:
    """
    Generate anchors for each cell and each (class x rotation) combination.

    Args:
        cfg: pipeline config
        feature_map_size: [H, W] of the BEV feature map from backbone

    Returns:
        anchors: (H, W, num_anchors, 7) — [x, y, z, l, w, h, yaw]
    """
    vcfg = cfg.voxel
    acfg = cfg.anchor

    H, W = feature_map_size
    pc_range = np.array(vcfg.point_cloud_range, dtype=np.float32)
    x_range = np.linspace(pc_range[0], pc_range[3], W + 1)
    y_range = np.linspace(pc_range[1], pc_range[4], H + 1)
    x_centers = (x_range[:-1] + x_range[1:]) / 2
    y_centers = (y_range[:-1] + y_range[1:]) / 2

    # Build anchor templates: (num_classes * num_rotations, 7)
    templates = []
    for (l, w, h, z, _yaw) in acfg.anchor_sizes:
        for rot in acfg.anchor_rotations:
            templates.append([0, 0, z, l, w, h, rot])
    templates = np.array(templates, dtype=np.float32)  # (A, 7)
    A = len(templates)

    # Grid: (H, W, A, 7)
    xx, yy = np.meshgrid(x_centers, y_centers)  # each (H, W)
    anchors = np.tile(templates[None, None, :, :], (H, W, 1, 1))  # (H, W, A, 7)
    anchors[:, :, :, 0] = xx[:, :, None]  # x
    anchors[:, :, :, 1] = yy[:, :, None]  # y
    # z center already set from template

    return anchors  # (H, W, A, 7)


def boxes_to_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """
    Convert 3D boxes to BEV corners for IoU computation.

    Args:
        boxes: (..., 7) — x,y,z,l,w,h,yaw
    Returns:
        corners: (..., 4, 2) — BEV corners
    """
    x, y, l, w, yaw = boxes[..., 0], boxes[..., 1], boxes[..., 3], boxes[..., 4], boxes[..., 6]
    cos_a = np.cos(yaw)
    sin_a = np.sin(yaw)

    # Local corners (4 x 2)
    hl, hw = l / 2, w / 2
    local = np.stack([
        np.stack([hl, hw], axis=-1),
        np.stack([hl, -hw], axis=-1),
        np.stack([-hl, -hw], axis=-1),
        np.stack([-hl, hw], axis=-1),
    ], axis=-2)  # (..., 4, 2)

    rot = np.stack([
        np.stack([cos_a, -sin_a], axis=-1),
        np.stack([sin_a, cos_a], axis=-1),
    ], axis=-2)  # (..., 2, 2)

    corners = (rot[..., None, :, :] @ local[..., None]).squeeze(-1)
    center = np.stack([x, y], axis=-1)[..., None, :]  # (..., 1, 2)
    return corners + center
