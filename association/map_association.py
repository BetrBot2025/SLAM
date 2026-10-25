"""WP5: class-aware map association (Algorithm 1 Stage 2, Sec. IV-B).

Per-class matching radius (footprint diagonal) instead of a uniform distance
threshold, plus a second global re-association pass against the whole map
(not just currently-tracked objects) to catch reappearance after tracking
loss / loop closure before spawning a duplicate.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .bipartite import Detection3D


@dataclass
class MapObject:
    object_id: int
    class_id: int
    class_name: str
    center: np.ndarray      # (3,) x,y,z
    size: np.ndarray        # (3,) l,w,h
    yaw: float
    last_seen_frame: int
    num_observations: int = 1


def class_radius(l: float, w: float) -> float:
    """r_c = sqrt(l_c^2 + w_c^2) -- average class footprint diagonal."""
    return float(np.hypot(l, w))


def _ground_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _greedy_nearest_match(
    objects: List[MapObject],
    candidates: List[Detection3D],
    remaining_idx: Set[int],
) -> List[Tuple[int, int]]:
    """For each map object, find the closest same-class, still-unassociated
    candidate within its class radius (Algorithm 1 lines 5-11)."""
    matches: List[Tuple[int, int]] = []   # (map_object_index, candidate_index)
    for m_idx, m in enumerate(objects):
        r_c = class_radius(m.size[0], m.size[1])
        best_j, best_d = None, float("inf")
        for j in remaining_idx:
            c = candidates[j]
            if c.class_id != m.class_id:
                continue
            d = _ground_dist(m.center, np.array(c.box[:3]))
            if d < r_c and d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            matches.append((m_idx, best_j))
            remaining_idx.discard(best_j)
    return matches


def associate_to_map(
    tracked_objects: List[MapObject],
    all_map_objects: List[MapObject],
    candidates: List[Detection3D],
    frame_id: int,
    next_object_id: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[MapObject]]:
    """
    Full Stage 2: tracked-object association, then a second global pass
    against all map objects, then new-object instantiation for the rest.

    Returns:
        tracked_matches: (index into tracked_objects, index into candidates)
        reappear_matches: (index into all_map_objects, index into candidates)
        new_objects: freshly instantiated MapObject list (not yet inserted
            into the caller's map -- Algorithm 1 line 15, GlobalMap.add)
    """
    remaining = set(range(len(candidates)))

    tracked_matches = _greedy_nearest_match(tracked_objects, candidates, remaining)

    tracked_ids = {tracked_objects[m].object_id for m, _ in tracked_matches}
    reappear_pool = [o for o in all_map_objects if o.object_id not in tracked_ids]
    reappear_matches_pool = _greedy_nearest_match(reappear_pool, candidates, remaining)
    # remap pool indices back to all_map_objects indices
    pool_id_to_all_idx = {o.object_id: i for i, o in enumerate(all_map_objects)}
    reappear_matches = [
        (pool_id_to_all_idx[reappear_pool[pi].object_id], cj)
        for pi, cj in reappear_matches_pool
    ]

    new_objects: List[MapObject] = []
    for j in sorted(remaining):
        c = candidates[j]
        new_objects.append(MapObject(
            object_id=next_object_id,
            class_id=c.class_id,
            class_name=c.class_name,
            center=np.array(c.box[:3]),
            size=np.array(c.box[3:6]),
            yaw=c.box[6],
            last_seen_frame=frame_id,
        ))
        next_object_id += 1

    return tracked_matches, reappear_matches, new_objects


def _selftest():
    """Runnable directly: python -m association.map_association"""
    car = MapObject(object_id=1, class_id=0, class_name="Car",
                     center=np.array([10.0, 0.0, 0.0]), size=np.array([4.6, 1.9, 1.7]),
                     yaw=0.0, last_seen_frame=0)

    # A candidate 1m away from the existing car -> within r_c ~= 5.0 -> associates.
    near_cand = Detection3D(class_id=0, class_name="Car", score=0.9,
                             box=[10.5, 0.5, 0.0, 4.6, 1.9, 1.7, 0.0],
                             projected_points=np.zeros((0, 2)))
    # A far pedestrian candidate -> no map object of that class -> new object.
    ped_cand = Detection3D(class_id=3, class_name="Pedestrian", score=0.8,
                            box=[50.0, 50.0, 0.0, 0.7, 0.7, 1.8, 0.0],
                            projected_points=np.zeros((0, 2)))

    tracked_matches, reappear_matches, new_objects = associate_to_map(
        tracked_objects=[car], all_map_objects=[car],
        candidates=[near_cand, ped_cand], frame_id=1, next_object_id=2,
    )
    assert tracked_matches == [(0, 0)], f"expected car matched to candidate 0, got {tracked_matches}"
    assert len(new_objects) == 1 and new_objects[0].class_name == "Pedestrian"
    print("association.map_association self-test OK:",
          tracked_matches, reappear_matches, [o.object_id for o in new_objects])


if __name__ == "__main__":
    _selftest()
