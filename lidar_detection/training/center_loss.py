"""CenterPoint losses (Yin et al., CVPR 2021).

Heatmap loss: modified focal loss for Gaussian targets (Law & Deng, CornerNet).
  L_hm = -(1-Y')^2 * log(Y')            at peak locations where Y == 1
         -(1-Y)^4  * Y'^2 * log(1-Y')   at non-peak locations where Y < 1
  Normalised by the total number of GT objects in the batch.

Regression loss: L1 at GT center locations only.
  L_reg = w_off * L1(offset) + w_hei * L1(height)
        + w_siz * L1(size)   + w_rot * L1(rot)
  Each L1 term also normalised by the number of GT objects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CenterLoss(nn.Module):
    def __init__(
        self,
        hm_weight:  float = 1.0,
        off_weight: float = 1.0,
        hei_weight: float = 1.0,
        siz_weight: float = 0.25,
        rot_weight: float = 0.1,
    ):
        super().__init__()
        self.hm_weight  = hm_weight
        self.off_weight = off_weight
        self.hei_weight = hei_weight
        self.siz_weight = siz_weight
        self.rot_weight = rot_weight

        self.last_hm_loss  = 0.0
        self.last_reg_loss = 0.0

    @staticmethod
    def _gaussian_focal_loss(pred_logit: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
        """
        Modified focal loss for Gaussian heatmap targets in [0, 1].
        Returns sum (not mean) — caller divides by N_obj.
        """
        pred = torch.clamp(pred_logit.sigmoid(), 1e-4, 1 - 1e-4)
        pos_mask    = (target == 1).float()
        neg_weights = (1 - target).pow(4)

        pos_loss = pos_mask  *           (1 - pred).pow(2) * torch.log(pred)
        neg_loss = neg_weights * (1 - pos_mask) * pred.pow(2) * torch.log(1 - pred)
        return -(pos_loss + neg_loss).sum()

    @staticmethod
    def _l1_at_mask(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """L1 sum at masked locations. mask: (B, 1, H, W) bool."""
        if not mask.any():
            return pred.sum() * 0.0
        expanded = mask.expand_as(pred)
        return F.l1_loss(pred[expanded], target[expanded], reduction='sum')

    def forward(
        self,
        preds:      Dict[str, torch.Tensor],
        hm_target:  torch.Tensor,   # (B, K, H, W) Gaussian in [0,1]
        reg_mask:   torch.Tensor,   # (B, 1, H, W) bool — GT center cells
        off_target: torch.Tensor,   # (B, 2, H, W)
        hei_target: torch.Tensor,   # (B, 1, H, W)
        siz_target: torch.Tensor,   # (B, 3, H, W)
        rot_target: torch.Tensor,   # (B, 2, H, W)
    ) -> torch.Tensor:

        N_obj = reg_mask.sum().clamp(min=1).float()

        hm_loss  = self._gaussian_focal_loss(preds['heatmap'], hm_target) / N_obj
        hm_loss  = hm_loss * self.hm_weight

        off_loss = self._l1_at_mask(preds['offset'], off_target, reg_mask) / N_obj
        hei_loss = self._l1_at_mask(preds['height'], hei_target, reg_mask) / N_obj
        siz_loss = self._l1_at_mask(preds['size'],   siz_target, reg_mask) / N_obj
        rot_loss = self._l1_at_mask(preds['rot'],    rot_target, reg_mask) / N_obj

        reg_loss = (off_loss * self.off_weight
                    + hei_loss * self.hei_weight
                    + siz_loss * self.siz_weight
                    + rot_loss * self.rot_weight)

        self.last_hm_loss  = hm_loss.item()
        self.last_reg_loss = reg_loss.item()

        return hm_loss + reg_loss
