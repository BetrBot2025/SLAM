"""WP9: per-stage latency instrumentation -- Table II's Detect./Object/Max
Obj./KF BA Latency columns, plus a rolling System FPS counter."""
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Tuple
import numpy as np


class LatencyTracker:
    def __init__(self, fps_window: int = 30):
        self._durations: Dict[str, list] = defaultdict(list)
        self._frame_times: list = []
        self._fps_window = fps_window
        self._last_frame_t = None

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[stage].append((time.perf_counter() - t0) * 1000.0)   # ms

    def record(self, stage: str, duration_ms: float):
        """For durations measured elsewhere (e.g. a queue's own timestamps)."""
        self._durations[stage].append(duration_ms)

    def tick_frame(self):
        """Call once per processed frame to drive the FPS counter."""
        now = time.perf_counter()
        if self._last_frame_t is not None:
            self._frame_times.append(now - self._last_frame_t)
            self._frame_times = self._frame_times[-self._fps_window:]
        self._last_frame_t = now

    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        return 1.0 / float(np.mean(self._frame_times))

    def stage_stats(self, stage: str) -> Tuple[float, float, float]:
        """(mean_ms, std_ms, max_ms) for one stage, matching Table II's
        'Mean +/- Std [ms]' / 'Max ... [ms]' columns."""
        vals = self._durations.get(stage, [])
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std()), float(arr.max())

    def summary(self) -> Dict[str, Tuple[float, float, float]]:
        return {stage: self.stage_stats(stage) for stage in self._durations}

    def print_summary(self):
        print(f"{'Stage':<20} {'Mean [ms]':>10} {'Std [ms]':>10} {'Max [ms]':>10}")
        for stage, (mean, std, mx) in self.summary().items():
            print(f"{stage:<20} {mean:>10.2f} {std:>10.2f} {mx:>10.2f}")
        print(f"{'System FPS':<20} {self.fps():>10.2f}")


def _selftest():
    """Runnable directly: python -m evaluation_slam.latency"""
    tracker = LatencyTracker()
    for _ in range(5):
        tracker.tick_frame()
        with tracker.measure("detect"):
            time.sleep(0.01)
        with tracker.measure("object"):
            time.sleep(0.005)
        time.sleep(0.02)

    mean_ms, std_ms, max_ms = tracker.stage_stats("detect")
    assert 5.0 < mean_ms < 30.0, f"unexpected detect latency mean {mean_ms}ms"
    assert tracker.fps() > 0
    tracker.print_summary()
    print("evaluation_slam.latency self-test OK")


if __name__ == "__main__":
    _selftest()
