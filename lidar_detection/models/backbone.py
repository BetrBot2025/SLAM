"""
2D CNN backbone matching the PointPillars paper exactly (Lang et al., 2019).

Paper architecture (C = 64, S = 2 for car / single unified model):
  Block1(S,  4, C)   stride-S entry + 3 stride-1 layers,  C   = 64  channels
  Block2(2S, 6, 2C)  stride-2 entry + 5 stride-1 layers,  2C  = 128 channels
  Block3(4S, 6, 4C)  stride-2 entry + 5 stride-1 layers,  4C  = 256 channels

  Up1(S→S,   2C)  ConvTranspose 1×1, no spatial change  → 128 ch
  Up2(2S→S,  2C)  ConvTranspose 2×2, 2× upsample        → 128 ch
  Up3(4S→S,  2C)  ConvTranspose 4×4, 4× upsample        → 128 ch

  Concat(Up1, Up2, Up3) → 6C = 384 channels

Note: the paper trains separate backbones for Car (S=2) and Ped/Cyclist (S=1).
We use a single shared backbone at S=2, which is standard for unified nuScenes models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, channels)


def _make_block(in_c: int, out_c: int, num_layers: int, stride: int) -> nn.Sequential:
    """
    One backbone block: strided entry layer + (num_layers - 1) stride-1 layers.
    Matches Block(stride, num_layers, out_c) notation from the paper.
    """
    layers = [
        nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
        _gn(out_c), nn.ReLU(inplace=True),
    ]
    for _ in range(num_layers - 1):
        layers += [
            nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
            _gn(out_c), nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class BEVBackbone(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 384):
        super().__init__()
        C = 64   # base channel width

        # Paper: Block1(S=2, 4 layers, C), Block2(2S, 6 layers, 2C), Block3(4S, 6 layers, 4C)
        self.block1 = _make_block(in_channels, C,     num_layers=4, stride=2)
        self.block2 = _make_block(C,           2 * C, num_layers=6, stride=2)
        self.block3 = _make_block(2 * C,       4 * C, num_layers=6, stride=2)

        # Paper: Up1(S→S, 2C), Up2(2S→S, 2C), Up3(4S→S, 2C)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(C,     2 * C, kernel_size=1, stride=1),
            _gn(2 * C), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(2 * C, 2 * C, kernel_size=2, stride=2),
            _gn(2 * C), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(4 * C, 2 * C, kernel_size=4, stride=4),
            _gn(2 * C), nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels   # 3 × 2C = 6C = 384

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.block1(x)    # (B,  C, H/2, W/2)
        s2 = self.block2(s1)   # (B, 2C, H/4, W/4)
        s3 = self.block3(s2)   # (B, 4C, H/8, W/8)

        u1 = self.up1(s1)      # (B, 2C, H/2, W/2)
        u2 = self.up2(s2)      # (B, 2C, H/2, W/2)
        u3 = self.up3(s3)      # (B, 2C, H/2, W/2)

        # Align spatial dims via interpolation (handles rounding at odd sizes)
        h, w = u1.shape[2], u1.shape[3]
        u2 = F.interpolate(u2, size=(h, w), mode='bilinear', align_corners=False)
        u3 = F.interpolate(u3, size=(h, w), mode='bilinear', align_corners=False)

        return torch.cat([u1, u2, u3], dim=1)   # (B, 6C=384, H/2, W/2)
