"""DeepSDF decoder network (Park et al., CVPR 2019) -- an 8-layer MLP
conditioned on a latent shape code, with the standard skip connection
re-injecting (xyz, latent) at the 4th hidden layer.

One instance of this per class (see multi_decoder.py); this file only
defines the architecture. Weights are randomly initialized until trained
on ShapeNet (or another asset source) per WP6 of the planning doc -- exactly
the same "runs, but untrained" status the existing lidar_detection demo
already documents for random-weight inference.
"""
import torch
import torch.nn as nn


class DeepSDFDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256, dropout_p: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        in_dim = 3 + latent_dim

        self.fc1 = nn.utils.parametrizations.weight_norm(nn.Linear(in_dim, hidden_dim))
        self.fc2 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        # skip connection: re-concatenate (xyz, latent) here, per DeepSDF Fig. 2
        self.fc4 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim + in_dim, hidden_dim))
        self.fc5 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc6 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc7 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.tanh = nn.Tanh()

    def forward(self, xyz: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        xyz:    (N, 3) query points, object-canonical frame.
        latent: (latent_dim,) or (N, latent_dim) shape code.
        Returns: (N, 1) signed distance, in [-1, 1] (tanh-clamped, per DeepSDF).
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0).expand(xyz.shape[0], -1)
        x_in = torch.cat([xyz, latent], dim=-1)

        x = self.relu(self.dropout(self.fc1(x_in)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.relu(self.dropout(self.fc3(x)))
        x = torch.cat([x, x_in], dim=-1)
        x = self.relu(self.dropout(self.fc4(x)))
        x = self.relu(self.dropout(self.fc5(x)))
        x = self.relu(self.dropout(self.fc6(x)))
        x = self.relu(self.dropout(self.fc7(x)))
        return self.tanh(self.fc_out(x))
