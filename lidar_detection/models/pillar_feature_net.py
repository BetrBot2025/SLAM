"""Pillar Feature Network (PFN) — per-pillar pointwise MLP + max-pool."""
import torch
import torch.nn as nn


class _GN(nn.GroupNorm):
    """GroupNorm that accepts 2-D (N, C) by briefly adding a dummy spatial dim."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return super().forward(x.unsqueeze(-1)).squeeze(-1)
        return super().forward(x)


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        # GroupNorm replaces BatchNorm1d — batch-size-independent and immune to
        # the sparse-BEV issue where most pillars are zero-padded.
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            _GN(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels, bias=False),
            _GN(8, out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, pillars: torch.Tensor, num_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillars:    (V, P, C) — pillar point features
            num_points: (V,)      — valid point count per pillar
        Returns:
            (V, out_channels) — per-pillar feature vector
        """
        V, P, C = pillars.shape
        x = pillars.view(V * P, C)
        x = self.net(x)
        x = x.view(V, P, self.out_channels)

        mask = torch.arange(P, device=pillars.device)[None, :] < num_points[:, None]
        x = x * mask.unsqueeze(-1).float()
        x[~mask] = -1e9
        x, _ = x.max(dim=1)  # (V, out_channels)
        return x
