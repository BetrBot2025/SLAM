"""
LiDAR 3D Object Detection Pipeline.

Usage:
    from lidar_detection.pipeline import DetectionPipeline
    pipe = DetectionPipeline()
    detections = pipe.run(points)        # (N, 4) numpy array
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

from .config import PipelineConfig
from .preprocessing.voxelizer import voxelize, compute_pillar_features
from .preprocessing.filters import remove_ground_ransac, crop_to_range
from .models.pointpillars import PointPillars, CenterPointPillars
from .detection.postprocess import postprocess
from .detection.center_postprocess import center_postprocess, ScoreThreshold


class DetectionPipeline:
    def __init__(
        self,
        cfg: Optional[PipelineConfig] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        use_center: bool = False,
    ):
        """
        Args:
            use_center: if True, build CenterPointPillars and decode with
                center_postprocess (matches checkpoints saved with
                Trainer(..., use_center=True), e.g. checkpoint_nuscenes_center.pth).
                If False (default), builds the anchor-head PointPillars model.
        """
        self.cfg = cfg or PipelineConfig()
        self.use_center = use_center
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = (CenterPointPillars(self.cfg) if use_center
                      else PointPillars(self.cfg)).to(self.device)

        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint: {checkpoint}  [{'CenterPoint' if use_center else 'Anchor'} head]")

        self.model.eval()

    def preprocess(self, points: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Raw points (N, 4+) → voxelized tensors ready for model.
        Returns (pillars, coordinates, num_points) as torch tensors.
        """
        cfg = self.cfg
        # 1. Crop to detection range
        points = crop_to_range(points, cfg.voxel.point_cloud_range)

        # 2. Voxelize
        voxels, coords, num_pts = voxelize(points, cfg.voxel)

        # 3. Pillar feature augmentation
        pillar_feats = compute_pillar_features(voxels, coords, num_pts, cfg.voxel)

        # 4. To tensors
        pillars_t = torch.from_numpy(pillar_feats).to(self.device)
        coords_t = torch.from_numpy(coords).to(self.device)
        num_pts_t = torch.from_numpy(num_pts).to(self.device)

        return pillars_t, coords_t, num_pts_t

    def _resolve_score_threshold(self, score_threshold: Optional[ScoreThreshold]) -> ScoreThreshold:
        """Explicit arg > cfg.model.class_score_thresholds (per-class) > 0.3 default."""
        thr = score_threshold
        if thr is None:
            thr = self.cfg.model.class_score_thresholds
        if thr is None:
            thr = 0.3
        if isinstance(thr, (list, tuple)) and not self.use_center:
            raise ValueError(
                "Per-class score thresholds are only supported for the CenterPoint "
                "head (use_center=True); the anchor-head postprocess() takes a "
                "single scalar threshold."
            )
        return thr

    @torch.no_grad()
    def run(
        self,
        points: np.ndarray,
        score_threshold: Optional[ScoreThreshold] = None,
        nms_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Full inference pass on a single frame.

        Args:
            points: (N, 4) — x, y, z, intensity
            score_threshold: scalar or per-class sequence (see
                cfg.model.class_score_thresholds). Defaults to the config
                value if set, else 0.3.

        Returns:
            list of dicts: {'box': [x,y,z,l,w,h,yaw], 'score': float,
                            'class_id': int, 'class_name': str}
        """
        pillars, coords, num_pts = self.preprocess(points)

        if pillars.shape[0] == 0:
            return []

        thr = self._resolve_score_threshold(score_threshold)

        if self.use_center:
            preds = self.model(pillars, coords, num_pts, batch_size=1)
            results = center_postprocess(preds, self.cfg, score_threshold=thr)
            return results[0]

        cls_preds, reg_preds, dir_preds = self.model(
            pillars, coords, num_pts, batch_size=1
        )

        # Feature map spatial size (H, W) after backbone (half of voxel grid)
        gx, gy, _ = self.cfg.voxel.grid_size
        feat_h = gy // 2
        feat_w = gx // 2

        results = postprocess(
            cls_preds, reg_preds, self.cfg,
            feature_map_size=[feat_h, feat_w],
            score_threshold=thr,
            nms_threshold=nms_threshold,
        )
        return results[0]

    def run_batch(
        self,
        batch_points: List[np.ndarray],
        score_threshold: Optional[ScoreThreshold] = None,
        nms_threshold: float = 0.5,
    ) -> List[List[Dict]]:
        """Run on a list of point clouds (same batch, padded)."""
        all_pillars, all_coords, all_num_pts, batch_idx = [], [], [], []

        for b_i, pts in enumerate(batch_points):
            pillars, coords, num_pts = self.preprocess(pts)
            b_vec = torch.full((len(coords),), b_i, dtype=torch.long, device=self.device)
            all_pillars.append(pillars)
            all_coords.append(coords)
            all_num_pts.append(num_pts)
            batch_idx.append(b_vec)

        if not all_pillars:
            return [[] for _ in batch_points]

        pillars = torch.cat(all_pillars, dim=0)
        coords = torch.cat(all_coords, dim=0)
        num_pts = torch.cat(all_num_pts, dim=0)
        batch_idx_t = torch.cat(batch_idx, dim=0)

        thr = self._resolve_score_threshold(score_threshold)

        with torch.no_grad():
            if self.use_center:
                preds = self.model(
                    pillars, coords, num_pts,
                    batch_size=len(batch_points), batch_idx=batch_idx_t,
                )
                return center_postprocess(preds, self.cfg, score_threshold=thr)

            cls_preds, reg_preds, _ = self.model(
                pillars, coords, num_pts,
                batch_size=len(batch_points),
                batch_idx=batch_idx_t,
            )

        gx, gy, _ = self.cfg.voxel.grid_size
        results = postprocess(
            cls_preds, reg_preds, self.cfg,
            feature_map_size=[gy // 2, gx // 2],
            score_threshold=thr,
            nms_threshold=nms_threshold,
        )
        return results

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Checkpoint saved: {path}")
