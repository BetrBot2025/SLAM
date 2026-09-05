"""
PointPillars losses matching the paper exactly (Lang et al. / SECOND).

Total loss (Eq. in paper):
    L = (1/N_pos) * (beta_loc * L_loc + beta_cls * L_cls + beta_dir * L_dir)

  beta_loc = 2,  beta_cls = 1,  beta_dir = 0.2

L_cls  — Focal loss  (alpha=0.25, gamma=2),  summed over ALL anchors
L_loc  — SmoothL1   over 7 residuals,        summed over positive anchors
L_dir  — CE (2-bin direction),                summed over positive anchors

All three sums are then divided by N_pos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PointPillarsLoss(nn.Module):
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 2.0,
        dir_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.cls_weight  = cls_weight
        self.reg_weight  = reg_weight
        self.dir_weight  = dir_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Diagnostics — filled after each forward, readable by the training loop
        self.last_cls_loss = 0.0
        self.last_reg_loss = 0.0
        self.last_dir_loss = 0.0

    def _focal_loss_sum(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sum (not mean) focal loss over all anchors."""
        p     = torch.sigmoid(pred)
        ce    = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t   = p * target + (1 - p) * (1 - target)
        a_t   = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        loss  = a_t * (1 - p_t) ** self.focal_gamma * ce
        return loss.sum()

    def _smooth_l1_sum(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SmoothL1 (beta=1/9) sum over all elements of pred/target."""
        beta = 1.0 / 9.0
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return loss.sum()

    def forward(
        self,
        cls_preds:   torch.Tensor,          # (B, H, W, A*C)
        reg_preds:   torch.Tensor,          # (B, H, W, A*7)
        dir_preds:   torch.Tensor,          # (B, H, W, A*2)
        cls_targets: torch.Tensor,          # (B, H, W, A*C)  — 0/1 float
        reg_targets: torch.Tensor,          # (B, H, W, A*7)
        pos_mask:    torch.Tensor,          # (B, H, W, A)    — bool
        dir_targets: Optional[torch.Tensor] = None,  # (B, H, W, A) — int64
    ) -> torch.Tensor:

        N_pos = pos_mask.sum().clamp(min=1).float()

        # ── Classification loss (focal, all anchors) ──────────────────────────
        cls_loss_sum = self._focal_loss_sum(cls_preds, cls_targets)
        cls_loss     = cls_loss_sum / N_pos * self.cls_weight

        # ── Regression loss (smooth-L1, positive anchors only) ────────────────
        A  = pos_mask.shape[-1]
        # Expand pos_mask from (B,H,W,A) to (B,H,W,A*7)
        pm7 = pos_mask.unsqueeze(-1).expand(*pos_mask.shape, 7).reshape(
            *pos_mask.shape[:-1], A * 7)
        if pm7.any():
            reg_loss_sum = self._smooth_l1_sum(reg_preds[pm7], reg_targets[pm7])
        else:
            reg_loss_sum = reg_preds.sum() * 0.0
        reg_loss = reg_loss_sum / N_pos * self.reg_weight

        # ── Direction loss (2-bin CE, positive anchors only) ──────────────────
        dir_loss = cls_loss * 0.0    # zero tensor on same device, same dtype
        if dir_targets is not None and pos_mask.any():
            B, H, W, Ach = pos_mask.shape
            dir_flat  = dir_preds.reshape(B, H, W, Ach, 2)
            dir_pos   = dir_flat[pos_mask]               # (N_pos, 2)  logits
            tgt_pos   = dir_targets[pos_mask].long()     # (N_pos,)
            dir_loss  = F.cross_entropy(dir_pos, tgt_pos, reduction="sum") / N_pos
            dir_loss  = dir_loss * self.dir_weight

        # ── Store for logging ─────────────────────────────────────────────────
        self.last_cls_loss = cls_loss.item()
        self.last_reg_loss = reg_loss.item()
        self.last_dir_loss = dir_loss.item()

        return cls_loss + reg_loss + dir_loss
