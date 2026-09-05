"""Anchor-based 3D detection head."""
import torch
import torch.nn as nn
from ..config import ModelConfig


class DetectionHead(nn.Module):
    """
    Predicts per-anchor: class scores + 7-DOF box regression (dx,dy,dz,dl,dw,dh,dyaw).
    """
    def __init__(self, in_channels: int, num_classes: int, num_anchors_per_cell: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_cell

        self.cls_head = nn.Conv2d(
            in_channels,
            num_anchors_per_cell * num_classes,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            in_channels,
            num_anchors_per_cell * 7,
            kernel_size=1,
        )
        self.dir_head = nn.Conv2d(
            in_channels,
            num_anchors_per_cell * 2,  # 2-bin direction classification
            kernel_size=1,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, -2.0)  # low initial confidence
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.constant_(self.reg_head.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) BEV feature map
        Returns:
            cls_preds: (B, H, W, A*num_classes)
            reg_preds: (B, H, W, A*7)
            dir_preds: (B, H, W, A*2)
        """
        cls = self.cls_head(x).permute(0, 2, 3, 1).contiguous()
        reg = self.reg_head(x).permute(0, 2, 3, 1).contiguous()
        dir = self.dir_head(x).permute(0, 2, 3, 1).contiguous()
        return cls, reg, dir
