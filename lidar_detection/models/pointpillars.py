"""Full PointPillars model: PFN → scatter → backbone → head.

Two variants:
  PointPillars      — anchor-based head (original)
  CenterPointPillars — center-based heatmap head (Yin et al., CVPR 2021)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional

from ..config import PipelineConfig
from .pillar_feature_net import PillarFeatureNet
from .backbone import BEVBackbone
from .detection_head import DetectionHead
from .center_head import CenterHead


class PointPillars(nn.Module):
    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        vcfg = cfg.voxel
        mcfg = cfg.model

        # PFN input: original features + 5 augmented offsets
        pfn_in = mcfg.num_point_features + 5
        self.pfn = PillarFeatureNet(pfn_in, mcfg.pillar_feature_dim)

        self.backbone = BEVBackbone(
            in_channels=mcfg.pillar_feature_dim,
            out_channels=384,
        )

        # 2 rotations per anchor class
        n_anchors = mcfg.num_classes * 2
        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=mcfg.num_classes,
            num_anchors_per_cell=n_anchors,
        )

        self.grid_size = vcfg.grid_size  # [X, Y, Z]

    def scatter_to_bev(
        self,
        pillar_features: torch.Tensor,  # (V, C)
        coordinates: torch.Tensor,      # (V, 3) — z, y, x
        batch_size: int,
        batch_idx: Optional[torch.Tensor] = None,  # (V,) if batched
    ) -> torch.Tensor:
        """Convert sparse pillar features to dense (B, C, Y, X) BEV map."""
        C = pillar_features.shape[1]
        X, Y, _ = self.grid_size
        # Backbone operates on H/2 x W/2; we scatter at full res
        canvas = torch.zeros(batch_size, C, Y, X, device=pillar_features.device, dtype=pillar_features.dtype)

        b_idx = batch_idx if batch_idx is not None else torch.zeros(len(coordinates), dtype=torch.long, device=pillar_features.device)
        y_idx = coordinates[:, 1].long()
        x_idx = coordinates[:, 2].long()

        canvas[b_idx, :, y_idx, x_idx] = pillar_features
        return canvas

    def forward(
        self,
        pillars: torch.Tensor,       # (V, P, C)
        coordinates: torch.Tensor,   # (V, 3)
        num_points: torch.Tensor,    # (V,)
        batch_size: int = 1,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Pillar feature extraction
        pf = self.pfn(pillars, num_points)  # (V, pillar_dim)

        # 2. Scatter to BEV pseudo-image
        bev = self.scatter_to_bev(pf, coordinates, batch_size, batch_idx)  # (B, C, Y, X)

        # 3. 2D backbone
        feat = self.backbone(bev)  # (B, 384, Y/2, X/2)

        # 4. Detection head
        cls_preds, reg_preds, dir_preds = self.head(feat)
        return cls_preds, reg_preds, dir_preds


class CenterPointPillars(nn.Module):
    """PointPillars backbone + CenterPoint detection head."""

    is_center_head = True   # lets the trainer/evaluator identify this variant

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg
        vcfg = cfg.voxel
        mcfg = cfg.model

        pfn_in = mcfg.num_point_features + 5
        self.pfn = PillarFeatureNet(pfn_in, mcfg.pillar_feature_dim)

        self.backbone = BEVBackbone(
            in_channels=mcfg.pillar_feature_dim,
            out_channels=384,
        )

        self.head = CenterHead(
            in_channels=self.backbone.out_channels,
            num_classes=mcfg.num_classes,
        )

        self.grid_size = vcfg.grid_size  # [X, Y, Z]

    def scatter_to_bev(
        self,
        pillar_features: torch.Tensor,
        coordinates:     torch.Tensor,
        batch_size:      int,
        batch_idx:       Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        C = pillar_features.shape[1]
        X, Y, _ = self.grid_size
        canvas = torch.zeros(batch_size, C, Y, X,
                             device=pillar_features.device,
                             dtype=pillar_features.dtype)
        b_idx = (batch_idx if batch_idx is not None
                 else torch.zeros(len(coordinates), dtype=torch.long,
                                  device=pillar_features.device))
        canvas[b_idx, :, coordinates[:, 1].long(), coordinates[:, 2].long()] = pillar_features
        return canvas

    def forward(
        self,
        pillars:    torch.Tensor,
        coordinates: torch.Tensor,
        num_points: torch.Tensor,
        batch_size: int = 1,
        batch_idx:  Optional[torch.Tensor] = None,
    ) -> dict:
        pf   = self.pfn(pillars, num_points)
        bev  = self.scatter_to_bev(pf, coordinates, batch_size, batch_idx)
        feat = self.backbone(bev)
        return self.head(feat)
