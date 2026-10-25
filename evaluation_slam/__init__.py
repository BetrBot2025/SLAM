"""WP9: SLAM-level evaluation -- ATE (Table I style) and per-stage latency
instrumentation (Table II style). Complements lidar_detection/evaluation/,
which is per-frame detection AP/NDS only, not trajectory/latency metrics."""
from .ate import align_trajectories, compute_ate
from .latency import LatencyTracker

__all__ = ["align_trajectories", "compute_ate", "LatencyTracker"]
