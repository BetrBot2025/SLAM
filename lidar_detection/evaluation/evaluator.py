"""Run a trained model over a dataset and collect detections + GTs.

Supports both PointPillars (anchor-based) and CenterPointPillars (heatmap-based).
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from ..config import PipelineConfig
from ..preprocessing.voxelizer import voxelize, compute_pillar_features
from ..preprocessing.filters import crop_to_range
from ..detection.postprocess import postprocess
from ..detection.center_postprocess import center_postprocess


class Evaluator:
    def __init__(self, model, cfg: PipelineConfig, device=None):
        self.model  = model
        self.cfg    = cfg
        self.device = device or torch.device("cpu")

    def _prepare(self, points: np.ndarray):
        points = crop_to_range(points, self.cfg.voxel.point_cloud_range)
        voxels, coords, num_pts = voxelize(points, self.cfg.voxel)
        feats = compute_pillar_features(voxels, coords, num_pts, self.cfg.voxel)
        return (
            torch.from_numpy(feats).to(self.device),
            torch.from_numpy(coords).to(self.device),
            torch.from_numpy(num_pts).to(self.device),
        )

    @staticmethod
    def _normalise_gts(gt_boxes: List[Dict], pc_range: List[float]) -> List[Dict]:
        """
        Convert loader GT format → {'box': [x,y,z,l,w,h,yaw], 'class_id': int}.
        GTs whose center is outside the point cloud range are excluded — they are
        undetectable by the model and should not count as false negatives.
        """
        xmin, ymin, _, xmax, ymax, _ = pc_range
        out = []
        for gt in gt_boxes:
            cx, cy, cz = gt["center"]
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                continue
            l, w, h = gt["size"]
            out.append({
                "box":      [cx, cy, cz, l, w, h, float(gt["yaw"])],
                "class_id": int(gt["class_id"]),
            })
        return out

    @torch.no_grad()
    def run(
        self,
        dataset,
        n_samples:  Optional[int] = None,
        score_thr:  float = 0.3,
        nms_thr:    float = 0.5,
        verbose:    bool  = True,
    ) -> Tuple[List[List[Dict]], List[List[Dict]]]:
        """
        Run inference over the first n_samples frames of dataset.

        Returns:
            all_dets: per-frame list of {'box', 'score', 'class_id', 'class_name'}
            all_gts:  per-frame list of {'box', 'class_id'}
        """
        self.model.eval()
        gx, gy, _ = self.cfg.voxel.grid_size
        feat_size  = [gy // 2, gx // 2]

        n = min(n_samples, len(dataset)) if n_samples else len(dataset)

        all_dets: List[List[Dict]] = []
        all_gts:  List[List[Dict]] = []

        for i in range(n):
            if verbose and (i + 1) % 100 == 0:
                print(f"  [{i+1:4d}/{n}]", flush=True)

            points, gt_boxes = dataset[i]
            gts_norm = self._normalise_gts(gt_boxes, self.cfg.voxel.point_cloud_range)

            if points is None or len(points) == 0:
                all_dets.append([])
                all_gts.append(gts_norm)
                continue

            pillars, coords, num_pts = self._prepare(points)
            if pillars.shape[0] == 0:
                all_dets.append([])
                all_gts.append(gts_norm)
                continue

            if getattr(self.model, 'is_center_head', False):
                preds = self.model(pillars, coords, num_pts)
                dets  = center_postprocess(preds, self.cfg,
                                           score_threshold=score_thr)
            else:
                cls_preds, reg_preds, dir_preds = self.model(pillars, coords, num_pts)
                dets = postprocess(cls_preds, reg_preds, self.cfg, feat_size,
                                   score_threshold=score_thr, nms_threshold=nms_thr)
            all_dets.append(dets[0])
            all_gts.append(gts_norm)

        return all_dets, all_gts
