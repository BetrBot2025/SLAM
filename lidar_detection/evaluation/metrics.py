"""
Official nuScenes detection metrics (Section 3.1 of the nuScenes paper).

Key differences from KITTI / COCO style:
  - Matching by 2D BEV *centre distance* (NOT IoU)
  - AP averaged over D = {0.5, 1, 2, 4} metres and over classes
  - True Positive metrics at d = 2 m: ATE, ASE, AOE
  - NDS = (1/10) * [5*mAP + sum(1 - min(1, mTP))]
    (AVE / AAE omitted — no velocity / attribute head in our model)

Reference: Caesar et al., nuScenes, CVPR 2020, Eq. (1)-(3).
"""
import numpy as np
from typing import Dict, List

DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]   # metres
TP_MATCH_DIST   = 2.0                      # metres (all TP metrics use 2 m)
MIN_RECALL      = 0.1                      # ignore recall < 10 %


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _center_dist_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """
    2D BEV centre distances between N detections and M GT boxes.
    Args:
        dets: (N, 7) — x,y,z,l,w,h,yaw
        gts:  (M, 7)
    Returns:
        dist: (N, M) in metres
    """
    dx = dets[:, 0:1] - gts[None, :, 0]   # (N, 1) - (1, M) → (N, M)
    dy = dets[:, 1:2] - gts[None, :, 1]
    return np.sqrt(dx**2 + dy**2)


def _scale_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    ASE: 1 - 3D IoU after aligning orientation and translation.
    Measures size accuracy only.
    """
    lp, wp, hp = pred[3], pred[4], pred[5]
    lg, wg, hg = gt[3],   gt[4],  gt[5]
    inter = min(lp, lg) * min(wp, wg) * min(hp, hg)
    union = lp*wp*hp + lg*wg*hg - inter + 1e-7
    return float(1.0 - inter / union)


def _orient_error(pred_yaw: float, gt_yaw: float) -> float:
    """AOE: smallest yaw angle difference on a full 360° period (radians)."""
    diff = abs(pred_yaw - gt_yaw) % (2.0 * np.pi)
    return float(min(diff, 2.0 * np.pi - diff))


# ── AP computation ────────────────────────────────────────────────────────────

def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    nuScenes AP: 101-point interpolation over recall in [MIN_RECALL, 1.0].
    Returns AP in [0, 1].
    """
    if len(recalls) == 0:
        return 0.0
    ap = 0.0
    for r in np.linspace(MIN_RECALL, 1.0, 101):
        mask = recalls >= r
        ap  += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 101.0


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_nuscenes_metrics(
    all_dets: List[List[Dict]],
    all_gts:  List[List[Dict]],
    class_names: List[str],
    dist_thresholds: List[float] = DIST_THRESHOLDS,
    tp_match_dist:   float       = TP_MATCH_DIST,
) -> Dict:
    """
    Compute official nuScenes detection metrics.

    all_dets[i] = [{'box': [x,y,z,l,w,h,yaw], 'score': float, 'class_id': int}, ...]
    all_gts[i]  = [{'box': [x,y,z,l,w,h,yaw], 'class_id': int}, ...]

    Returns dict with:
        per_class_ap[cls][dist]  AP for each (class, distance)
        class_ap[cls]            AP averaged over distances
        n_gt[cls]                number of GT boxes
        mAP                      mean over classes and distances
        mATE, mASE, mAOE         mean TP metrics
        NDS                      nuScenes Detection Score (partial, no AVE/AAE)
        mAP_per_dist[d]          mAP at each distance threshold
    """
    C            = len(class_names)
    per_class_ap = {n: {} for n in class_names}
    tp_errors    = {n: {'ATE': [], 'ASE': [], 'AOE': []} for n in class_names}
    n_gt         = {n: 0 for n in class_names}

    for frame_gts in all_gts:
        for g in frame_gts:
            cid = g['class_id']
            if 0 <= cid < C:
                n_gt[class_names[cid]] += 1

    for c, cls in enumerate(class_names):

        # ── AP at each distance threshold ─────────────────────────────────
        for d_thr in dist_thresholds:
            scores_all: List[float] = []
            tp_all:     List[bool]  = []
            n_gt_c = n_gt[cls]

            for frame_dets, frame_gts in zip(all_dets, all_gts):
                dets_c = [d for d in frame_dets if d['class_id'] == c]
                gts_c  = [g for g in frame_gts  if g['class_id'] == c]

                if not dets_c:
                    continue

                det_boxes  = np.array([d['box']   for d in dets_c], dtype=np.float32)
                det_scores = np.array([d['score'] for d in dets_c], dtype=np.float32)
                tp_flags   = np.zeros(len(dets_c), dtype=bool)

                if gts_c:
                    gt_boxes = np.array([g['box'] for g in gts_c], dtype=np.float32)
                    dist_mat = _center_dist_matrix(det_boxes, gt_boxes)
                    matched  = set()
                    for di in np.argsort(-det_scores):
                        best_gt   = int(dist_mat[di].argmin())
                        best_dist = float(dist_mat[di, best_gt])
                        if best_dist <= d_thr and best_gt not in matched:
                            tp_flags[di] = True
                            matched.add(best_gt)

                scores_all.extend(det_scores.tolist())
                tp_all.extend(tp_flags.tolist())

            if n_gt_c == 0 or not scores_all:
                per_class_ap[cls][d_thr] = 0.0
                continue

            order      = np.argsort(-np.array(scores_all))
            tp_sorted  = np.array(tp_all)[order]
            cum_tp     = np.cumsum(tp_sorted.astype(int))
            cum_fp     = np.cumsum((~tp_sorted).astype(int))
            recalls    = cum_tp / (n_gt_c  + 1e-7)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-7)
            per_class_ap[cls][d_thr] = _compute_ap(recalls, precisions)

        # ── TP metrics at tp_match_dist ───────────────────────────────────
        for frame_dets, frame_gts in zip(all_dets, all_gts):
            dets_c = [d for d in frame_dets if d['class_id'] == c]
            gts_c  = [g for g in frame_gts  if g['class_id'] == c]

            if not dets_c or not gts_c:
                continue

            det_boxes  = np.array([d['box']   for d in dets_c], dtype=np.float32)
            det_scores = np.array([d['score'] for d in dets_c], dtype=np.float32)
            gt_boxes   = np.array([g['box']   for g in gts_c],  dtype=np.float32)
            dist_mat   = _center_dist_matrix(det_boxes, gt_boxes)
            matched    = set()

            for di in np.argsort(-det_scores):
                best_gt   = int(dist_mat[di].argmin())
                best_dist = float(dist_mat[di, best_gt])
                if best_dist <= tp_match_dist and best_gt not in matched:
                    matched.add(best_gt)
                    pred = det_boxes[di]
                    gt   = gt_boxes[best_gt]
                    tp_errors[cls]['ATE'].append(best_dist)
                    tp_errors[cls]['ASE'].append(_scale_error(pred, gt))
                    tp_errors[cls]['AOE'].append(_orient_error(pred[6], gt[6]))

    # ── Aggregate ─────────────────────────────────────────────────────────
    class_ap = {
        n: float(np.mean(list(per_class_ap[n].values()))) for n in class_names
    }
    mAP = float(np.mean(list(class_ap.values())))

    def _mean_tp(metric):
        vals = []
        for n in class_names:
            v = tp_errors[n][metric]
            vals.append(float(np.mean(v)) if v else 1.0)
        return float(np.mean(vals))

    mATE = _mean_tp('ATE')
    mASE = _mean_tp('ASE')
    mAOE = _mean_tp('AOE')

    # NDS (partial — AVE and AAE omitted, set to worst-case 1.0 → contribute 0)
    # Full formula: NDS = (1/10) * [5*mAP + (1-min(1,mATE)) + ... (5 terms)]
    # We have 3 TP terms; score the 2 missing as 0 (worst case already excluded)
    tp_score = (1 - min(1.0, mATE)) + (1 - min(1.0, mASE)) + (1 - min(1.0, mAOE))
    NDS      = (5.0 * mAP + tp_score) / 10.0   # denominator stays 10 per paper

    return {
        'per_class_ap':    per_class_ap,
        'class_ap':        class_ap,
        'n_gt':            n_gt,
        'mAP':             mAP,
        'mATE':            mATE,
        'mASE':            mASE,
        'mAOE':            mAOE,
        'NDS':             NDS,
        'mAP_per_dist':    {d: float(np.mean([per_class_ap[n].get(d, 0.0)
                                               for n in class_names]))
                            for d in dist_thresholds},
        'dist_thresholds': dist_thresholds,
        'class_names':     class_names,
        'tp_errors':       tp_errors,
    }
