"""
nuScenes dataset loader (no nuscenes-devkit dependency).

Supports two layouts:

  Mini (single directory):
    NuScenesDataset(root="nuScenes_mini", version="v1.0-mini")

  Full trainval (split blob directories):
    NuScenesDataset(
        meta_root  = "nuscenes_full/v1.0-trainval_meta",
        data_roots = ["nuscenes_full/v1.0-trainval01_blobs", ...],
        version    = "v1.0-trainval",
    )

Key features:
  - num_sweeps: accumulate N LiDAR sweeps (default 10, like the paper)
  - split:      scene-level 90/10 train/val deterministic split
  - get_cbgs_indices(): Class-Balanced Group Sampling indices for one epoch
"""
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


CATEGORY_MAP = {
    "vehicle.car":                              "Car",
    "vehicle.truck":                            "Car",
    "vehicle.bus.bendy":                        "Bus",
    "vehicle.bus.rigid":                        "Bus",
    "vehicle.motorcycle":                       "Motorcycle",
    "human.pedestrian.adult":                   "Pedestrian",
    "human.pedestrian.child":                   "Pedestrian",
    "human.pedestrian.construction_worker":     "Pedestrian",
    "human.pedestrian.police_officer":          "Pedestrian",
    # vehicle.bicycle intentionally unmapped: not one of the four target
    # classes (car/bus/motorcycle/pedestrian) — bicycle instances are skipped.
}

CLASS_IDS   = {"Car": 0, "Bus": 1, "Motorcycle": 2, "Pedestrian": 3}
CLASS_NAMES = ["Car", "Bus", "Motorcycle", "Pedestrian"]


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _quat_to_rot(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _quat_mul(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inv(q) -> list:
    w, x, y, z = q
    return [w, -x, -y, -z]


def _quat_to_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))


# ── Low-level I/O ─────────────────────────────────────────────────────────────

def _load_pcd_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :4]   # drop ring index; keep x,y,z,intensity


# ── Dataset ───────────────────────────────────────────────────────────────────

class NuScenesDataset:
    """
    __getitem__ returns (points, gt_boxes) in the keyframe LiDAR sensor frame.
    points is (N, 4): x,y,z,intensity — from num_sweeps accumulated sweeps.
    """

    def __init__(
        self,
        root:       Optional[str]       = None,
        version:    str                 = "v1.0-mini",
        meta_root:  Optional[str]       = None,
        data_roots: Optional[List[str]] = None,
        split:      Optional[str]       = None,   # 'train' | 'val' | None
        num_sweeps: int                 = 10,      # LiDAR sweeps to accumulate
    ):
        if meta_root is None:
            meta_root = root
        if data_roots is None:
            data_roots = [root] if root else []

        meta_dir = os.path.join(meta_root, version)

        def _load(name):
            with open(os.path.join(meta_dir, name)) as f:
                return json.load(f)

        # ── file index: relative path → absolute path ─────────────────────────
        self._file_index: Dict[str, str] = {}
        for dr in data_roots:
            for subdir in ("samples", "sweeps"):
                lidar_dir = os.path.join(dr, subdir, "LIDAR_TOP")
                if not os.path.isdir(lidar_dir):
                    continue
                for fname in os.listdir(lidar_dir):
                    if fname.endswith(".pcd.bin"):
                        rel = f"{subdir}/LIDAR_TOP/{fname}"
                        self._file_index[rel] = os.path.join(lidar_dir, fname)

        # ── metadata ──────────────────────────────────────────────────────────
        self._ego_pose   = {e["token"]: e for e in _load("ego_pose.json")}
        self._cal_sensor = {c["token"]: c for c in _load("calibrated_sensor.json")}
        self._instances  = {i["token"]: i for i in _load("instance.json")}
        self._categories = {c["token"]: c["name"] for c in _load("category.json")}

        self._ann_by_sample: Dict[str, List[dict]] = {}
        for ann in _load("sample_annotation.json"):
            self._ann_by_sample.setdefault(ann["sample_token"], []).append(ann)

        # Index ALL LIDAR_TOP sample_data (keyframes + sweeps) by token
        self._sd_by_token: Dict[str, dict] = {}
        self._lidar_by_sample: Dict[str, dict] = {}
        for sd in _load("sample_data.json"):
            if "LIDAR_TOP" not in sd["filename"]:
                continue
            self._sd_by_token[sd["token"]] = sd
            if sd["is_key_frame"]:
                self._lidar_by_sample[sd["sample_token"]] = sd

        # ── available keyframes ────────────────────────────────────────────────
        samples_json   = _load("sample.json")
        scene_by_token = {s["token"]: s["scene_token"] for s in samples_json}
        all_tokens     = [
            s["token"] for s in samples_json
            if s["token"] in self._lidar_by_sample
        ]
        self._sample_tokens = [
            t for t in all_tokens
            if self._lidar_by_sample[t]["filename"] in self._file_index
        ]

        n_total = len(all_tokens)
        n_avail = len(self._sample_tokens)
        print(f"NuScenesDataset ({version}): {n_avail} / {n_total} keyframes available"
              + ("" if n_avail == n_total
                 else f"  ({n_total-n_avail} missing -- download more blobs)"))

        # ── deterministic 90/10 scene-level split ─────────────────────────────
        if split in ("train", "val"):
            avail_set    = set(self._sample_tokens)
            scenes_used  = sorted({scene_by_token[t] for t in avail_set
                                   if t in scene_by_token})
            n_train      = round(len(scenes_used) * 0.9)
            train_scenes = set(scenes_used[:n_train])
            val_scenes   = set(scenes_used[n_train:])
            keep_scenes  = train_scenes if split == "train" else val_scenes
            self._sample_tokens = [t for t in self._sample_tokens
                                   if scene_by_token.get(t) in keep_scenes]
            print(f"  Split '{split}': {len(self._sample_tokens)} frames  "
                  f"({len(keep_scenes)}/{len(scenes_used)} scenes)")
        elif split is not None:
            raise ValueError(f"split must be 'train', 'val', or None; got {split!r}")

        self.num_sweeps = num_sweeps

        # ── per-class sample index (for CBGS) ─────────────────────────────────
        # Lazy-built on first call to get_cbgs_indices()
        self._class_indices: Optional[Dict[str, List[int]]] = None

    # ── Internal: sweep accumulation ──────────────────────────────────────────

    def _get_sweep_chain(self, keyframe_sd: dict) -> List[dict]:
        """
        Walk `prev` pointers to collect up to (num_sweeps - 1) preceding sweeps.
        Returns list ordered oldest → newest (keyframe excluded, appended by caller).
        """
        chain  = []
        cursor = keyframe_sd.get("prev", "")
        while cursor and len(chain) < self.num_sweeps - 1:
            sd = self._sd_by_token.get(cursor)
            if sd is None:
                break
            chain.append(sd)
            cursor = sd.get("prev", "")
        chain.reverse()   # oldest first
        return chain

    def _transform_sweep(
        self,
        pts:     np.ndarray,   # (N,4) in sweep LiDAR frame
        sw_sd:   dict,          # sweep sample_data
        kf_sd:   dict,          # keyframe sample_data
    ) -> np.ndarray:
        """
        Transform points from sweep LiDAR frame → keyframe LiDAR frame.
        Chain: sweep_lidar → sweep_ego → global → kf_ego → kf_lidar
        """
        sw_cal = self._cal_sensor[sw_sd["calibrated_sensor_token"]]
        kf_cal = self._cal_sensor[kf_sd["calibrated_sensor_token"]]
        sw_ego = self._ego_pose[sw_sd["ego_pose_token"]]
        kf_ego = self._ego_pose[kf_sd["ego_pose_token"]]

        R_sw  = _quat_to_rot(sw_cal["rotation"])
        t_sw  = np.array(sw_cal["translation"])
        R_seg = _quat_to_rot(sw_ego["rotation"])
        t_seg = np.array(sw_ego["translation"])
        R_keg = _quat_to_rot(kf_ego["rotation"])
        t_keg = np.array(kf_ego["translation"])
        R_kcal = _quat_to_rot(kf_cal["rotation"])
        t_kcal  = np.array(kf_cal["translation"])

        xyz = pts[:, :3].T.astype(np.float64)       # (3, N)
        xyz = R_sw @ xyz + t_sw[:, None]             # → sweep ego
        xyz = R_seg @ xyz + t_seg[:, None]           # → global
        xyz = R_keg.T @ (xyz - t_keg[:, None])       # → kf ego
        xyz = R_kcal.T @ (xyz - t_kcal[:, None])     # → kf lidar

        out = pts.copy()
        out[:, :3] = xyz.T.astype(np.float32)
        return out

    # ── Public: dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._sample_tokens)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict]]:
        token = self._sample_tokens[idx]
        kf_sd = self._lidar_by_sample[token]

        # Keyframe points
        path     = self._file_index[kf_sd["filename"]]
        points   = _load_pcd_bin(path)

        # Accumulated previous sweeps
        if self.num_sweeps > 1:
            for sw_sd in self._get_sweep_chain(kf_sd):
                sw_path = self._file_index.get(sw_sd["filename"])
                if sw_path is None:
                    continue   # sweep file not downloaded
                sw_pts = _load_pcd_bin(sw_path)
                sw_pts = self._transform_sweep(sw_pts, sw_sd, kf_sd)
                points = np.concatenate([points, sw_pts], axis=0)

        # Sensor pose (for GT coordinate transform)
        cal      = self._cal_sensor[kf_sd["calibrated_sensor_token"]]
        ego_pose = self._ego_pose[kf_sd["ego_pose_token"]]
        R_sensor = _quat_to_rot(cal["rotation"])
        t_sensor = np.array(cal["translation"])
        R_ego    = _quat_to_rot(ego_pose["rotation"])
        t_ego    = np.array(ego_pose["translation"])

        # GT boxes
        gt_boxes = []
        for ann in self._ann_by_sample.get(token, []):
            inst       = self._instances[ann["instance_token"]]
            cat_name   = self._categories[inst["category_token"]]
            class_name = CATEGORY_MAP.get(cat_name)
            if class_name is None:
                continue
            if ann["num_lidar_pts"] == 0:
                continue

            t_global = np.array(ann["translation"])
            t_ego_f  = R_ego.T @ (t_global - t_ego)
            t_lidar  = R_sensor.T @ (t_ego_f - t_sensor)

            q_global   = ann["rotation"]
            q_ego_inv  = _quat_inv(ego_pose["rotation"])
            q_sens_inv = _quat_inv(cal["rotation"])
            q_lidar    = _quat_mul(_quat_mul(q_sens_inv, q_ego_inv), q_global)
            yaw        = _quat_to_yaw(q_lidar)

            w_ns, l_ns, h_ns = ann["size"]   # nuScenes: [width, length, height]
            gt_boxes.append({
                "center":     t_lidar.tolist(),
                "size":       [l_ns, w_ns, h_ns],
                "yaw":        yaw,
                "class_id":   CLASS_IDS[class_name],
                "class_name": class_name,
            })

        return points, gt_boxes

    def get_token(self, idx: int) -> str:
        return self._sample_tokens[idx]

    # ── CBGS ─────────────────────────────────────────────────────────────────

    def _build_class_indices(self):
        """Build per-class index (lazy, called once on first CBGS request)."""
        print("Building CBGS class index ...")
        cls_idx: Dict[str, List[int]] = {n: [] for n in CLASS_NAMES}
        for i, token in enumerate(self._sample_tokens):
            classes_in_frame = set()
            for ann in self._ann_by_sample.get(token, []):
                inst     = self._instances[ann["instance_token"]]
                cat_name = self._categories[inst["category_token"]]
                cn       = CATEGORY_MAP.get(cat_name)
                if cn:
                    classes_in_frame.add(cn)
            for cn in classes_in_frame:
                cls_idx[cn].append(i)
        self._class_indices = cls_idx
        for cn, idxs in cls_idx.items():
            print(f"  {cn}: {len(idxs)} frames contain at least one instance")

    def get_cbgs_indices(self) -> List[int]:
        """
        Return CBGS-balanced index list for one epoch.
        Each class group contributes len(dataset) samples (with replacement),
        so rare classes (Cyclist) appear as often as common ones (Car).
        """
        if self._class_indices is None:
            self._build_class_indices()

        N = len(self._sample_tokens)
        epoch_indices: List[int] = []
        for cn in CLASS_NAMES:
            idxs = self._class_indices[cn]
            if not idxs:
                continue
            # Sample N indices with replacement from this class group
            chosen = np.random.choice(idxs, size=N, replace=True)
            epoch_indices.extend(chosen.tolist())

        np.random.shuffle(epoch_indices)
        return epoch_indices
