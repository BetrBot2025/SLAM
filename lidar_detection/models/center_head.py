"""CenterPoint detection head (Yin et al., CVPR 2021).

Replaces anchor-based detection with center-based heatmap detection:
  - K-channel Gaussian heatmap (one per class)
  - Sub-voxel offset (2): fractional displacement within feature cell
  - Height (1): z-center in LiDAR frame metres
  - Size (3): log(l), log(w), log(h)
  - Rotation (2): sin(yaw), cos(yaw) — resolves π-ambiguity without direction head

Architecture (Section 4.2): shared 3×3 conv + GN + ReLU, then per-output
branches of two 3×3 convs separated by GN + ReLU.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ── Gaussian rendering utilities ──────────────────────────────────────────────

def gaussian_radius(det_size: tuple, min_overlap: float = 0.1) -> float:
    """CornerNet-style Gaussian radius from box (height, width) in feature-map cells."""
    h, w = det_size

    a1 = 1.0;  b1 = h + w;  c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / 2

    a2 = 4.0;  b2 = 2 * (h + w);  c2 = (1 - min_overlap) * w * h
    sq2 = math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / 2

    a3 = 4 * min_overlap;  b3 = -2 * min_overlap * (h + w);  c3 = (min_overlap - 1) * w * h
    sq3 = math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def _gaussian2d(shape: tuple, sigma: float = 1.0) -> np.ndarray:
    m, n = [(s - 1.0) / 2.0 for s in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


def draw_gaussian(heatmap: np.ndarray, center: tuple, radius: int) -> None:
    """Draw a 2D Gaussian blob on heatmap (H, W) in-place, taking element-wise max."""
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])   # x=col, y=row
    H, W  = heatmap.shape

    left   = min(x, radius);          right  = min(W - x, radius + 1)
    top    = min(y, radius);           bottom = min(H - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return

    np.maximum(
        heatmap[y - top: y + bottom, x - left: x + right],
        gaussian[radius - top: radius + bottom, radius - left: radius + right],
        out=heatmap[y - top: y + bottom, x - left: x + right],
    )


# ── CenterHead ───────────────────────────────────────────────────────────────

def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, channels)


def _branch(in_c: int, mid_c: int, out_c: int) -> nn.Sequential:
    """Two-conv output branch: Conv+GN+ReLU → Conv (no activation on last layer)."""
    return nn.Sequential(
        nn.Conv2d(in_c, mid_c, 3, padding=1, bias=False),
        _gn(mid_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_c, out_c, 3, padding=1, bias=True),
    )


class CenterHead(nn.Module):
    """
    CenterPoint first-stage detection head.

    forward() returns a dict:
        'heatmap': (B, K, H, W)   raw logits — apply sigmoid before use
        'offset':  (B, 2, H, W)   sub-voxel (dx, dy) in feature-cell units
        'height':  (B, 1, H, W)   z-centre in metres
        'size':    (B, 3, H, W)   log(l), log(w), log(h)
        'rot':     (B, 2, H, W)   sin(yaw), cos(yaw)
    """
    def __init__(self, in_channels: int, num_classes: int, head_channels: int = 64):
        super().__init__()
        self.num_classes = num_classes

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
            _gn(head_channels),
            nn.ReLU(inplace=True),
        )

        self.heatmap_head = _branch(head_channels, head_channels, num_classes)
        self.offset_head  = _branch(head_channels, head_channels, 2)
        self.height_head  = _branch(head_channels, head_channels, 1)
        self.size_head    = _branch(head_channels, head_channels, 3)
        self.rot_head     = _branch(head_channels, head_channels, 2)

        self._init_weights()

    def _init_weights(self):
        # Initialise heatmap bias to -log((1-π)/π) with π=0.1 (RetinaNet trick)
        # so initial predictions are uniformly ~10% confident, avoiding loss explosion
        bias_val = -math.log((1 - 0.1) / 0.1)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_val)
        for head in (self.offset_head, self.height_head, self.size_head, self.rot_head):
            nn.init.normal_(head[-1].weight, std=0.001)
            nn.init.constant_(head[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict:
        shared = self.shared_conv(x)
        return {
            'heatmap': self.heatmap_head(shared),
            'offset':  self.offset_head(shared),
            'height':  self.height_head(shared),
            'size':    self.size_head(shared),
            'rot':     self.rot_head(shared),
        }
