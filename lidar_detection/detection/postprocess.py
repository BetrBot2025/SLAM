"""Decode network outputs into 3D boxes."""
import numpy as np
import torch
from typing import List, Dict
from .nms import nms_3d
from .anchor import generate_anchors
from ..config import PipelineConfig


def decode_boxes(
    reg_preds: np.ndarray,   # (H, W, A, 7)
    anchors: np.ndarray,     # (H, W, A, 7)
) -> np.ndarray:
    """
    Inverse of PointPillars box encoding:
        x = xa + dx * da   (da = sqrt(la^2 + wa^2))
        y = ya + dy * da
        z = za + dz * ha
        l = la * exp(dl)
        w = wa * exp(dw)
        h = ha * exp(dh)
        yaw = yaw_a + d_yaw
    """
    xa, ya, za = anchors[..., 0], anchors[..., 1], anchors[..., 2]
    la, wa, ha = anchors[..., 3], anchors[..., 4], anchors[..., 5]
    yaw_a = anchors[..., 6]

    da = np.sqrt(la ** 2 + wa ** 2)

    dx, dy, dz = reg_preds[..., 0], reg_preds[..., 1], reg_preds[..., 2]
    dl, dw, dh = reg_preds[..., 3], reg_preds[..., 4], reg_preds[..., 5]
    d_yaw = reg_preds[..., 6]

    x = xa + dx * da
    y = ya + dy * da
    z = za + dz * ha
    l = la * np.exp(np.clip(dl, -3, 3))
    w = wa * np.exp(np.clip(dw, -3, 3))
    h = ha * np.exp(np.clip(dh, -3, 3))
    yaw = yaw_a + d_yaw

    return np.stack([x, y, z, l, w, h, yaw], axis=-1)


def postprocess(
    cls_preds: torch.Tensor,   # (B, H, W, A*C)
    reg_preds: torch.Tensor,   # (B, H, W, A*7)
    cfg: PipelineConfig,
    feature_map_size: List[int],  # [H, W]
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5,
) -> List[List[Dict]]:
    """
    Returns a list (batch) of lists of detections:
        [{'box': (7,), 'score': float, 'class_id': int, 'class_name': str}]
    """
    mcfg = cfg.model
    acfg = cfg.anchor

    anchors = generate_anchors(cfg, feature_map_size)  # (H, W, A, 7)
    A = anchors.shape[2]
    C = mcfg.num_classes

    cls_np = cls_preds.sigmoid().detach().cpu().numpy()  # (B, H, W, A*C)
    reg_np = reg_preds.detach().cpu().numpy()             # (B, H, W, A*7)

    B = cls_np.shape[0]
    H, W = feature_map_size

    cls_np = cls_np.reshape(B, H, W, A, C)
    reg_np = reg_np.reshape(B, H, W, A, 7)

    # Decode all boxes at once
    decoded = decode_boxes(reg_np, anchors[None])  # (B, H, W, A, 7)

    all_results = []
    for b in range(B):
        dets = []
        for c in range(C):
            scores = cls_np[b, :, :, :, c].ravel()          # (H*W*A,)
            boxes = decoded[b].reshape(-1, 7)                # (H*W*A, 7)

            pos = scores >= score_threshold
            if pos.sum() == 0:
                continue

            s = scores[pos]
            bx = boxes[pos]
            keep = nms_3d(bx, s, nms_threshold)

            for ki in keep:
                dets.append({
                    "box": bx[ki].tolist(),
                    "score": float(s[ki]),
                    "class_id": c,
                    "class_name": mcfg.class_names[c],
                })
        all_results.append(dets)
    return all_results
