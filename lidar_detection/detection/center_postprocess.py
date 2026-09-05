"""Decode CenterPoint head predictions to 3D bounding boxes.

Inference pipeline:
  1. Sigmoid on heatmap
  2. 3×3 max-pool NMS — zero out cells that are not local maxima
  3. Per class: select top-K peaks by score
  4. Gather regression values (offset, height, size, rot) at peak cells
  5. Decode physical box coordinates
  6. Filter by score threshold
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Sequence, Union

from ..config import PipelineConfig

ScoreThreshold = Union[float, Sequence[float]]


def _gather_at(feat: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor):
    """
    Gather feature values at integer (xi, yi) locations.

    feat: (C, H, W)
    xi, yi: (K,) integer tensors
    Returns: (K, C)
    """
    C, H, W = feat.shape
    flat_idx = (yi * W + xi).long()          # (K,)
    flat     = feat.reshape(C, H * W)        # (C, H*W)
    return flat[:, flat_idx].T               # (K, C)


def center_postprocess(
    preds:           dict,
    cfg:             PipelineConfig,
    score_threshold: ScoreThreshold = 0.1,
    top_k:           int   = 500,
) -> List[List[Dict]]:
    """
    Decode CenterPoint predictions per batch item.

    Args:
        score_threshold: either a single float applied to all classes, or a
            per-class sequence (length == num_classes, indexed by class_id) —
            e.g. cfg.model.class_score_thresholds — matching Algorithm 1's
            class-specific confidence thresholds tau_{c,3D}.

    Returns list of per-frame detections:
        [{'box': [cx,cy,cz,l,w,h,yaw], 'score': float,
          'class_id': int, 'class_name': str}]
    """
    if isinstance(score_threshold, (list, tuple)):
        if len(score_threshold) != len(cfg.model.class_names):
            raise ValueError(
                f"score_threshold has {len(score_threshold)} entries but "
                f"cfg.model.class_names has {len(cfg.model.class_names)}"
            )
    mcfg = cfg.model
    vcfg = cfg.voxel
    pc   = vcfg.point_cloud_range

    # Physical resolution of the feature map (backbone stride = 2 over BEV grid)
    feat_cell = vcfg.voxel_size[0] * 2   # 0.4 m per feature cell

    heatmap = preds['heatmap'].sigmoid()   # (B, K_cls, H, W)
    offset  = preds['offset']              # (B, 2, H, W)
    height  = preds['height']             # (B, 1, H, W)
    size    = preds['size']               # (B, 3, H, W)
    rot     = preds['rot']                # (B, 2, H, W)

    # 3×3 max-pool NMS: keep only local maxima
    hmax    = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * (hmax == heatmap).float()

    B, K_cls, H_feat, W_feat = heatmap.shape
    K = min(top_k, H_feat * W_feat)

    all_results: List[List[Dict]] = []

    for b in range(B):
        dets: List[Dict] = []

        for c in range(K_cls):
            hm_c = heatmap[b, c]            # (H, W)
            flat_scores = hm_c.reshape(-1)  # (H*W,)

            topk_scores, topk_flat = flat_scores.topk(K)
            topk_yi = topk_flat // W_feat
            topk_xi = topk_flat % W_feat

            off_vals = _gather_at(offset[b], topk_xi, topk_yi)   # (K, 2)
            hei_vals = _gather_at(height[b], topk_xi, topk_yi)   # (K, 1)
            siz_vals = _gather_at(size[b],   topk_xi, topk_yi)   # (K, 3)
            rot_vals = _gather_at(rot[b],    topk_xi, topk_yi)   # (K, 2)

            # Decode physical coordinates
            cx  = (topk_xi.float() + off_vals[:, 0]) * feat_cell + pc[0]
            cy  = (topk_yi.float() + off_vals[:, 1]) * feat_cell + pc[1]
            cz  = hei_vals[:, 0]
            l   = torch.exp(siz_vals[:, 0].clamp(-3, 3))
            w   = torch.exp(siz_vals[:, 1].clamp(-3, 3))
            h   = torch.exp(siz_vals[:, 2].clamp(-3, 3))
            yaw = torch.atan2(rot_vals[:, 0], rot_vals[:, 1])

            thr_c = score_threshold[c] if isinstance(score_threshold, (list, tuple)) else score_threshold

            scores_np = topk_scores.cpu().numpy()
            keep      = np.where(scores_np >= thr_c)[0]

            cx_np  = cx.detach().cpu().numpy()
            cy_np  = cy.detach().cpu().numpy()
            cz_np  = cz.detach().cpu().numpy()
            l_np   = l.detach().cpu().numpy()
            w_np   = w.detach().cpu().numpy()
            h_np   = h.detach().cpu().numpy()
            yaw_np = yaw.detach().cpu().numpy()

            for k in keep:
                dets.append({
                    'box': [float(cx_np[k]), float(cy_np[k]), float(cz_np[k]),
                             float(l_np[k]),  float(w_np[k]),  float(h_np[k]),
                             float(yaw_np[k])],
                    'score':      float(scores_np[k]),
                    'class_id':   c,
                    'class_name': mcfg.class_names[c],
                })

        all_results.append(dets)

    return all_results
