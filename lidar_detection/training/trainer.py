"""Training loop for PointPillars."""
import time
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from typing import Optional, List, Dict, Tuple

from ..config import PipelineConfig
from ..models.pointpillars import PointPillars, CenterPointPillars
from ..data.generator import generate_scene
from ..preprocessing.voxelizer import voxelize, compute_pillar_features
from ..preprocessing.filters import crop_to_range
from .loss import PointPillarsLoss
from .center_loss import CenterLoss
from .augmentation import augment
from ..detection.anchor import generate_anchors
from ..models.center_head import gaussian_radius, draw_gaussian


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou_bev_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Axis-aligned BEV IoU between N boxes1 and M boxes2.
    Returns (N, M).
    """
    def _ext(b):
        x, y, l, w, yaw = b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]
        hl = l * 0.5 * np.abs(np.cos(yaw)) + w * 0.5 * np.abs(np.sin(yaw))
        hw = l * 0.5 * np.abs(np.sin(yaw)) + w * 0.5 * np.abs(np.cos(yaw))
        return x - hl, y - hw, x + hl, y + hw

    x1a, y1a, x1b, y1b = _ext(boxes1)
    x2a, y2a, x2b, y2b = _ext(boxes2)
    area1 = (x1b - x1a) * (y1b - y1a)
    area2 = (x2b - x2a) * (y2b - y2a)

    ix1   = np.maximum(x1a[:, None], x2a[None, :])
    iy1   = np.maximum(y1a[:, None], y2a[None, :])
    ix2   = np.minimum(x1b[:, None], x2b[None, :])
    iy2   = np.minimum(y1b[:, None], y2b[None, :])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


# ── Anchor target assignment ──────────────────────────────────────────────────

def _encode_targets(gt_boxes: List[Dict], anchors: np.ndarray, cfg: PipelineConfig):
    """
    IoU-based anchor assignment.

    Rules (matching reference PointPillars):
      1. Each GT's best anchor is forced positive if IoU >= neg_iou_thr.
      2. All anchors with IoU >= pos_iou_thr are positive.

    Uses spatial pre-filtering so only anchors near each GT are tested.

    Returns cls_targets (H,W,A,C), reg_targets (H,W,A,7), pos_mask (H,W,A),
            dir_targets (H,W,A) int8.
    """
    H, W, A, _ = anchors.shape
    C           = cfg.model.num_classes

    cls_targets = np.zeros((H, W, A, C), dtype=np.float32)
    reg_targets = np.zeros((H, W, A, 7), dtype=np.float32)
    pos_mask    = np.zeros((H, W, A),    dtype=bool)
    dir_targets = np.zeros((H, W, A),    dtype=np.int8)

    if not gt_boxes:
        return cls_targets, reg_targets, pos_mask, dir_targets

    pc      = cfg.voxel.point_cloud_range
    cell_x  = (pc[3] - pc[0]) / W
    cell_y  = (pc[4] - pc[1]) / H
    max_l   = max(s[0] for s in cfg.anchor.anchor_sizes)
    max_w   = max(s[1] for s in cfg.anchor.anchor_sizes)
    pos_thr = cfg.anchor.pos_iou_threshold
    neg_thr = cfg.anchor.neg_iou_threshold

    for gt in gt_boxes:
        cx, cy, cz = gt["center"]
        l, w, h    = gt["size"]
        yaw        = gt["yaw"]
        cid        = gt["class_id"]

        gt7 = np.array([[cx, cy, cz, l, w, h, yaw]], dtype=np.float32)

        sx = (l + max_l) * 0.5 + cell_x
        sy = (w + max_w) * 0.5 + cell_y

        xi_lo = max(0,     int((cx - sx - pc[0]) / cell_x))
        xi_hi = min(W - 1, int((cx + sx - pc[0]) / cell_x))
        yi_lo = max(0,     int((cy - sy - pc[1]) / cell_y))
        yi_hi = min(H - 1, int((cy + sy - pc[1]) / cell_y))

        if xi_lo > xi_hi or yi_lo > yi_hi:
            continue

        loc       = anchors[yi_lo:yi_hi+1, xi_lo:xi_hi+1, :, :]
        _, w_w    = loc.shape[:2]
        flat      = loc.reshape(-1, 7)

        iou       = _iou_bev_matrix(gt7, flat)[0]
        best_idx  = int(iou.argmax())
        best_iou  = float(iou[best_idx])

        pos_local = iou >= pos_thr
        if best_iou >= neg_thr:
            pos_local[best_idx] = True

        for local_idx in np.where(pos_local)[0]:
            ai  = local_idx % A
            tmp = local_idx // A
            xi  = xi_lo + (tmp % w_w)
            yi  = yi_lo + (tmp // w_w)

            anc = anchors[yi, xi, ai]
            xa, ya, za, la, wa, ha, yaw_a = anc
            da  = np.sqrt(la ** 2 + wa ** 2)

            cls_targets[yi, xi, ai, cid] = 1.0
            reg_targets[yi, xi, ai] = [
                (cx - xa) / (da  + 1e-6),
                (cy - ya) / (da  + 1e-6),
                (cz - za) / (ha  + 1e-6),
                np.log(np.clip(l  / (la + 1e-6), 1e-4, 1e4)),
                np.log(np.clip(w  / (wa + 1e-6), 1e-4, 1e4)),
                np.log(np.clip(h  / (ha + 1e-6), 1e-4, 1e4)),
                np.sin(yaw - yaw_a),   # paper: sin(θ_gt − θ_a), not raw diff
            ]
            # Direction bin: 0 = GT heading same half-circle as anchor,
            #                1 = GT heading in opposite half-circle (π-flipped)
            dir_targets[yi, xi, ai] = int(np.sin(yaw - yaw_a) < 0)
            pos_mask[yi, xi, ai] = True

    return cls_targets, reg_targets, pos_mask, dir_targets


def _encode_center_targets(
    gt_boxes: List[Dict],
    H_feat: int,
    W_feat: int,
    num_classes: int,
    pc_range: List[float],
    feat_cell: float,       # physical size of one feature map cell (metres)
    min_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render CenterPoint training targets for one frame.

    Returns (all as float32 numpy arrays):
        hm_target:  (K, H, W)  Gaussian heatmap in [0, 1]
        reg_mask:   (1, H, W)  bool — True at GT center cells
        off_target: (2, H, W)  sub-voxel (dx, dy) in [0, 1) cell units
        hei_target: (1, H, W)  z-centre in metres
        siz_target: (3, H, W)  log(l), log(w), log(h)
        rot_target: (2, H, W)  sin(yaw), cos(yaw)
    """
    hm_target  = np.zeros((num_classes, H_feat, W_feat), dtype=np.float32)
    reg_mask   = np.zeros((1, H_feat, W_feat), dtype=bool)
    off_target = np.zeros((2, H_feat, W_feat), dtype=np.float32)
    hei_target = np.zeros((1, H_feat, W_feat), dtype=np.float32)
    siz_target = np.zeros((3, H_feat, W_feat), dtype=np.float32)
    rot_target = np.zeros((2, H_feat, W_feat), dtype=np.float32)

    for gt in gt_boxes:
        cx, cy, cz = gt["center"]
        l, w, h    = gt["size"]
        yaw        = float(gt["yaw"])
        cid        = int(gt["class_id"])

        # Project to feature map grid (fractional coordinates)
        fx = (cx - pc_range[0]) / feat_cell
        fy = (cy - pc_range[1]) / feat_cell
        xi = int(fx)
        yi = int(fy)

        if not (0 <= xi < W_feat and 0 <= yi < H_feat):
            continue

        # Gaussian radius from box footprint in feature-map cells
        l_cells = max(l / feat_cell, 1.0)
        w_cells = max(w / feat_cell, 1.0)
        r = max(int(gaussian_radius((l_cells, w_cells))), min_radius)

        draw_gaussian(hm_target[cid], (xi, yi), r)

        # Regression targets at the exact integer center cell
        reg_mask[0, yi, xi]     = True
        off_target[:, yi, xi]   = [fx - xi, fy - yi]
        hei_target[0, yi, xi]   = cz
        siz_target[:, yi, xi]   = [np.log(max(l, 1e-3)),
                                    np.log(max(w, 1e-3)),
                                    np.log(max(h, 1e-3))]
        rot_target[:, yi, xi]   = [np.sin(yaw), np.cos(yaw)]

    return hm_target, reg_mask, off_target, hei_target, siz_target, rot_target


# ── Batch collation ───────────────────────────────────────────────────────────

def _collate_batch(
    prepared: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge variable-length (pillars, coords, num_pts) tuples into one batch.
    Adds a batch_idx tensor so the scatter op knows which sample each pillar
    belongs to.

    Returns: pillars, coords, num_pts, batch_idx  (all concatenated)
    """
    all_p, all_c, all_n, all_b = [], [], [], []
    for b, (pil, crd, npt) in enumerate(prepared):
        all_p.append(pil)
        all_c.append(crd)
        all_n.append(npt)
        all_b.append(torch.full((len(pil),), b, dtype=torch.long, device=pil.device))
    return torch.cat(all_p), torch.cat(all_c), torch.cat(all_n), torch.cat(all_b)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        cfg:         Optional[PipelineConfig] = None,
        device:      Optional[str]            = None,
        logger                                = None,
        total_steps: Optional[int]            = None,
        gt_database: Optional[dict]           = None,
        use_center:  bool                     = False,  # True = CenterPoint head
    ):
        self.cfg        = cfg or PipelineConfig()
        self.use_center = use_center
        self.device     = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if use_center:
            self.model     = CenterPointPillars(self.cfg).to(self.device)
            self.criterion = CenterLoss()
        else:
            self.model     = PointPillars(self.cfg).to(self.device)
            self.criterion = PointPillarsLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            betas=(0.95, 0.99),
            weight_decay=self.cfg.train.weight_decay,
        )
        self.scheduler = None
        if total_steps:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.train.lr * 10,
                total_steps=total_steps,
                pct_start=0.4,
                anneal_strategy="cos",
            )
        self.scaler      = GradScaler("cuda") if self.device.type == "cuda" else None
        self.gt_database = gt_database
        self.logger = logger
        head_str = "CenterPoint" if use_center else "Anchor"
        print(f"Training on {self.device}  [{head_str} head]")
        n_params = sum(p.numel() for p in self.model.parameters())
        amp_str = "AMP ON" if self.scaler else "AMP OFF (CPU)"
        print(f"Model parameters: {n_params / 1e6:.2f}M  |  batch_size={self.cfg.train.batch_size}  |  {amp_str}")
        if self.scheduler:
            print(f"OneCycleLR: lr {self.cfg.train.lr} -> {self.cfg.train.lr * 10}  "
                  f"total_steps={total_steps}")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def load(self, path: str, steps_per_epoch: Optional[int] = None) -> int:
        """
        Load model + optimizer + scheduler from checkpoint.
        Returns the next epoch number to train (last saved epoch + 1).
        Falls back gracefully if the file is the old bare-state-dict format.

        The OneCycleLR schedule is a single warmup+anneal cycle sized for a
        specific total_steps. If this run's total_steps (epochs x steps/epoch)
        matches what the checkpoint was trained with, its state is restored
        verbatim to continue the same cycle exactly (e.g. resuming after a
        crash with the same --epochs). If the target has changed (e.g.
        extending training to more epochs), restoring the old state would
        pin the scheduler back to its old, already-exhausted total_steps and
        crash on the very next .step(). In that case we keep the freshly
        constructed scheduler (already sized for the new total_steps) and
        fast-forward it to the equivalent step instead.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass
            if self.scheduler and ckpt.get("scheduler"):
                saved_sched = ckpt["scheduler"]
                if saved_sched.get("total_steps") == self.scheduler.total_steps:
                    try:
                        self.scheduler.load_state_dict(saved_sched)
                    except Exception:
                        pass
                elif steps_per_epoch:
                    done_steps = ckpt.get("epoch", 0) * steps_per_epoch
                    print(f"Scheduler total_steps changed ({saved_sched.get('total_steps')} -> "
                          f"{self.scheduler.total_steps}); rebuilding LR schedule and "
                          f"fast-forwarding {done_steps} steps instead of restoring old state.")
                    for _ in range(done_steps):
                        self.scheduler.step()
                else:
                    print("Scheduler total_steps changed but steps_per_epoch not given "
                          "to load(); starting the new schedule from step 0.")
            if self.scaler and ckpt.get("scaler"):
                try:
                    self.scaler.load_state_dict(ckpt["scaler"])
                except Exception:
                    pass
            last_epoch = ckpt.get("epoch", 0)
            print(f"Resumed  : {path}  (epoch {last_epoch} -> next epoch {last_epoch + 1})")
            return last_epoch + 1
        else:
            # Legacy format: bare state dict, no epoch info
            self.model.load_state_dict(ckpt)
            print(f"Loaded weights (legacy format): {path}")
            return 1

    def save(self, path: str, epoch: int = 0):
        """Save model + optimizer + scheduler + scaler + epoch number."""
        torch.save({
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler":    self.scaler.state_dict()    if self.scaler    else None,
        }, path)
        print(f"Saved checkpoint (epoch {epoch}): {path}")

    # ── Training ──────────────────────────────────────────────────────────────

    def _prepare_single(self, points: np.ndarray):
        """Voxelize one point cloud and return (pillars, coords, num_pts) tensors."""
        points = crop_to_range(points, self.cfg.voxel.point_cloud_range)
        voxels, coords, num_pts = voxelize(points, self.cfg.voxel)
        feats = compute_pillar_features(voxels, coords, num_pts, self.cfg.voxel)
        return (
            torch.from_numpy(feats).to(self.device),
            torch.from_numpy(coords).to(self.device),
            torch.from_numpy(num_pts).to(self.device),
        )

    def train_epoch(
        self,
        epoch: int,
        steps_per_epoch: int = 50,
        dataset=None,
        cbgs_indices=None,   # pre-built CBGS index list from dataset.get_cbgs_indices()
    ) -> float:
        self.model.train()
        total_loss = total_cls = total_reg = total_dir = 0.0
        batch_size = self.cfg.train.batch_size

        gx, gy, _ = self.cfg.voxel.grid_size
        feat_size  = [gy // 2, gx // 2]   # [H_feat, W_feat]
        H_feat, W_feat = feat_size

        if not self.use_center:
            anchors = generate_anchors(self.cfg, feat_size)
        feat_cell = self.cfg.voxel.voxel_size[0] * 2   # 0.4 m per feature cell

        # Pre-build a shuffled index list large enough for all steps × batch_size
        if dataset is not None:
            n_needed = steps_per_epoch * batch_size
            if cbgs_indices is not None:
                # CBGS: use the pre-balanced indices, wrap if needed
                indices = (cbgs_indices * (n_needed // len(cbgs_indices) + 1))[:n_needed]
            else:
                indices = list(range(len(dataset)))
                random.shuffle(indices)
                indices = (indices * (n_needed // len(indices) + 1))[:n_needed]

        actual_bs = batch_size  # updated each step
        t0 = time.time()

        for step in range(steps_per_epoch):
            # ── collect one batch ────────────────────────────────────────────
            batch_prepared: List = []
            batch_gts:      List = []

            if dataset is not None:
                start = step * batch_size
                for idx in indices[start:start + batch_size]:
                    points, gt_boxes = dataset[idx]
                    points, gt_boxes = augment(points, gt_boxes,
                                              gt_database=self.gt_database)
                    prep = self._prepare_single(points)
                    if prep[0].shape[0] > 0:
                        batch_prepared.append(prep)
                        batch_gts.append(gt_boxes)
            else:
                for sub in range(batch_size):
                    points, gt_boxes = generate_scene(
                        seed=epoch * 10000 + step * batch_size + sub)
                    prep = self._prepare_single(points)
                    if prep[0].shape[0] > 0:
                        batch_prepared.append(prep)
                        batch_gts.append(gt_boxes)

            if not batch_prepared:
                continue

            actual_bs = len(batch_prepared)
            pillars, coords, num_pts, batch_idx = _collate_batch(batch_prepared)

            # ── build targets + forward + backward ───────────────────────────
            self.optimizer.zero_grad()
            device_type = self.device.type

            if self.use_center:
                # ── CenterPoint targets ───────────────────────────────────────
                pc_range = self.cfg.voxel.point_cloud_range
                num_cls  = self.cfg.model.num_classes

                hm_list, mask_list = [], []
                off_list, hei_list, siz_list, rot_list = [], [], [], []
                for gt_b in batch_gts:
                    hm, mask, off, hei, siz, rr = _encode_center_targets(
                        gt_b, H_feat, W_feat, num_cls, pc_range, feat_cell)
                    hm_list.append(hm);    mask_list.append(mask)
                    off_list.append(off);  hei_list.append(hei)
                    siz_list.append(siz);  rot_list.append(rr)

                def _t(lst):
                    return torch.from_numpy(np.stack(lst)).to(self.device)

                hm_t   = _t(hm_list)    # (B, K, H, W)
                mask_t = _t(mask_list)  # (B, 1, H, W) bool
                off_t  = _t(off_list)   # (B, 2, H, W)
                hei_t  = _t(hei_list)   # (B, 1, H, W)
                siz_t  = _t(siz_list)   # (B, 3, H, W)
                rot_t  = _t(rot_list)   # (B, 2, H, W)

                with autocast(device_type, enabled=(self.scaler is not None)):
                    preds = self.model(pillars, coords, num_pts,
                                       batch_size=actual_bs, batch_idx=batch_idx)
                    loss  = self.criterion(preds, hm_t, mask_t,
                                           off_t, hei_t, siz_t, rot_t)

                cls_val = self.criterion.last_hm_loss
                reg_val = self.criterion.last_reg_loss
                dir_val = 0.0

            else:
                # ── Anchor targets ────────────────────────────────────────────
                cls_list, reg_list, pos_list, dir_list = [], [], [], []
                for gt_b in batch_gts:
                    c, r, p, d = _encode_targets(gt_b, anchors, self.cfg)
                    cls_list.append(c); reg_list.append(r)
                    pos_list.append(p); dir_list.append(d)

                H, W, A, C_cls = cls_list[0].shape
                cls_t_t = torch.from_numpy(
                    np.stack(cls_list).reshape(actual_bs, H, W, A * C_cls)).to(self.device)
                reg_t_t = torch.from_numpy(
                    np.stack(reg_list).reshape(actual_bs, H, W, A * 7)).to(self.device)
                pos_t   = torch.from_numpy(np.stack(pos_list)).to(self.device)
                dir_t   = torch.from_numpy(np.stack(dir_list).astype(np.int64)).to(self.device)

                with autocast(device_type, enabled=(self.scaler is not None)):
                    cls_preds, reg_preds, dir_preds = self.model(
                        pillars, coords, num_pts,
                        batch_size=actual_bs, batch_idx=batch_idx,
                    )
                    loss = self.criterion(cls_preds, reg_preds, dir_preds,
                                          cls_t_t, reg_t_t, pos_t, dir_t)

                cls_val = self.criterion.last_cls_loss
                reg_val = self.criterion.last_reg_loss
                dir_val = self.criterion.last_dir_loss

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            total_cls  += cls_val
            total_reg  += reg_val
            total_dir  += dir_val

            if self.logger:
                self.logger.log_step(epoch, step + 1, steps_per_epoch,
                                     loss_val, cls_val, reg_val)


        duration = time.time() - t0
        avg      = total_loss / steps_per_epoch
        avg_cls  = total_cls  / steps_per_epoch
        avg_reg  = total_reg  / steps_per_epoch
        lr_now   = self.optimizer.param_groups[0]["lr"]

        avg_dir  = total_dir  / steps_per_epoch
        if self.use_center:
            print(f"Epoch {epoch:3d} | loss {avg:.4f}  "
                  f"(hm {avg_cls:.4f}  reg {avg_reg:.4f})  "
                  f"lr {lr_now:.6f}  {duration:.0f}s")
        else:
            print(f"Epoch {epoch:3d} | loss {avg:.4f}  "
                  f"(cls {avg_cls:.4f}  reg {avg_reg:.4f}  dir {avg_dir:.4f})  "
                  f"lr {lr_now:.6f}  {duration:.0f}s")

        if self.logger:
            self.logger.log_epoch(epoch, self.cfg.train.epochs,
                                  avg, avg_cls, avg_reg, duration)
        return avg

    def train(self, epochs: Optional[int] = None, steps_per_epoch: int = 50,
              dataset=None):
        n_epochs = epochs or self.cfg.train.epochs
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch, steps_per_epoch, dataset=dataset)
        print("Training complete.")
