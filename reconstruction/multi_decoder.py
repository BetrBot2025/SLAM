"""WP6: modular multi-decoder dispatch + per-object latent-code optimization
(Sec. IV-C). One independent DeepSDFDecoder per class, selected at runtime by
the object's associated class label (from WP5's map association).

The optimization objective here -- minimize |SDF(surface point; z)| over the
object's accumulated point cloud -- is a simplified stand-in for the full
DSP-SLAM baseline objective (which also uses free-space samples along camera
rays and a 2D silhouette/mask term); it's enough to make z converge toward a
shape whose zero level-set passes through the observed points, which is the
right direction but not a reproduction of DSP-SLAM's exact loss. Swap in the
full objective once WP2/WP4 are far enough along to supply masks and camera
rays per keyframe.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .deepsdf import DeepSDFDecoder

try:
    from skimage import measure
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


@dataclass
class ReconstructionOutput:
    latent_code: np.ndarray
    loss_history: List[float]
    mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (verts, faces)


def extract_mesh(
    decoder: DeepSDFDecoder,
    latent: torch.Tensor,
    resolution: int = 32,
    bound: float = 1.0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Marching cubes over the decoder's SDF on a resolution^3 grid.
    Returns None if scikit-image isn't installed, or if the SDF has no zero
    crossing inside the grid (nothing to mesh)."""
    if not _HAS_SKIMAGE:
        return None
    device = latent.device
    coords = torch.linspace(-bound, bound, resolution, device=device)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    with torch.no_grad():
        sdf = decoder(pts, latent).reshape(resolution, resolution, resolution).cpu().numpy()

    if sdf.min() > 0 or sdf.max() < 0:
        return None
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
    verts = verts / (resolution - 1) * 2 * bound - bound
    return verts, faces


class MultiDecoderReconstructor:
    def __init__(self, class_names: List[str], latent_dim: int = 128,
                 hidden_dim: int = 256, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latent_dim = latent_dim
        self.decoders: Dict[str, DeepSDFDecoder] = {
            c: DeepSDFDecoder(latent_dim, hidden_dim).to(self.device).eval()
            for c in class_names
        }

    def load_decoder(self, class_name: str, state_dict_path: str):
        """Swap in a pretrained decoder once one exists (ShapeNet-trained, per
        the planning doc's WP6 risk note on bus/motorcycle asset scarcity)."""
        state = torch.load(state_dict_path, map_location=self.device, weights_only=True)
        self.decoders[class_name].load_state_dict(state)

    def reconstruct(
        self,
        class_name: str,
        points_object_frame: np.ndarray,   # (N, 3), object-canonical frame
        num_iters: int = 200,
        lr: float = 1e-2,
        extract: bool = True,
    ) -> ReconstructionOutput:
        if class_name not in self.decoders:
            raise KeyError(f"no decoder for class {class_name!r}; available: {list(self.decoders)}")
        decoder = self.decoders[class_name]

        pts = torch.from_numpy(points_object_frame).float().to(self.device)
        z = torch.zeros(self.latent_dim, device=self.device, requires_grad=True)   # DeepSDF: init at mean shape
        optimizer = torch.optim.Adam([z], lr=lr)

        loss_history = []
        for _ in range(num_iters):
            optimizer.zero_grad()
            sdf = decoder(pts, z)
            loss = sdf.abs().mean()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

        mesh = extract_mesh(decoder, z.detach()) if extract else None
        return ReconstructionOutput(
            latent_code=z.detach().cpu().numpy(),
            loss_history=loss_history,
            mesh=mesh,
        )


def _selftest():
    """Runnable directly: python -m reconstruction.multi_decoder"""
    recon = MultiDecoderReconstructor(class_names=["Car", "Bus", "Motorcycle", "Pedestrian"],
                                       latent_dim=16, hidden_dim=32)
    # Synthetic "surface" points on a unit sphere -- decoder is untrained/random,
    # so this only proves the optimization loop runs and loss is finite, not
    # that the recovered shape is meaningful.
    theta = np.random.uniform(0, 2 * np.pi, 200)
    phi = np.random.uniform(0, np.pi, 200)
    pts = np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=-1) * 0.5

    out = recon.reconstruct("Car", pts.astype(np.float32), num_iters=20, extract=False)
    assert len(out.loss_history) == 20
    assert np.isfinite(out.loss_history[-1])
    print(f"reconstruction.multi_decoder self-test OK: loss {out.loss_history[0]:.4f} -> {out.loss_history[-1]:.4f}"
          f" (untrained decoder -- values are not meaningful shape quality, only that optimization runs)")


if __name__ == "__main__":
    _selftest()
