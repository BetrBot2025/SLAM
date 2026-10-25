"""WP7: Algorithm 2 -- decouple heavy shape reconstruction from the
time-critical mapping thread via a worker thread pool + task/result queues,
with an instantaneous bounding-box placeholder standing in for the object
until its mesh is ready.

This is real, working threading code (queue.Queue is already thread-safe,
so no extra locking is needed around it). What it does NOT reproduce is the
paper's actual measured speedup (Table II) -- that depends on a real
reconstruct_fn (WP6's DeepSDF optimization) and a real workload; here it's
proven correct with a synthetic reconstruct_fn in _selftest().
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np


@dataclass
class ReconstructionTask:
    object_id: int
    class_name: str
    point_cloud: np.ndarray     # (N, 3), object-local frame
    pose: np.ndarray            # (4, 4) initial pose estimate, camera/world frame
    box: List[float]            # [x,y,z,l,w,h,yaw] -- for the placeholder


@dataclass
class ReconstructionResult:
    object_id: int
    latent_code: Optional[np.ndarray]
    pose: np.ndarray
    mesh: Optional[object]       # (verts, faces) or None if extraction unavailable


@dataclass
class PlaceholderObject:
    object_id: int
    class_name: str
    box: List[float]


def make_placeholder(task: ReconstructionTask) -> PlaceholderObject:
    """Instantaneous map insertion (Algorithm 2, producer side) -- the object
    is visible in the map immediately as its bounding box, before any
    DeepSDF optimization has run."""
    return PlaceholderObject(object_id=task.object_id, class_name=task.class_name, box=task.box)


_STOP = object()   # sentinel


class AsyncReconstructionBackend:
    def __init__(
        self,
        reconstruct_fn: Callable[[ReconstructionTask], ReconstructionResult],
        num_workers: int = 2,
    ):
        self.reconstruct_fn = reconstruct_fn
        self.task_queue: "queue.Queue" = queue.Queue()
        self.result_queue: "queue.Queue" = queue.Queue()
        self._running = threading.Event()
        self._running.set()   # detection is prioritized over reconstruction (Sec. V);
                               # callers can .pause() this around latency-critical work
        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True, name=f"reconstruct-worker-{i}")
            for i in range(num_workers)
        ]
        for w in self._workers:
            w.start()

    def _worker_loop(self):
        while True:
            task = self.task_queue.get()
            if task is _STOP:
                return
            self._running.wait()   # blocks while paused, without dropping the task
            try:
                result = self.reconstruct_fn(task)
            except Exception as e:
                result = ReconstructionResult(object_id=task.object_id, latent_code=None,
                                               pose=task.pose, mesh=None)
                print(f"async_backend: reconstruction failed for object {task.object_id}: {e}")
            self.result_queue.put(result)

    # -- Producer (mapping thread) API --------------------------------------

    def submit_task(self, task: ReconstructionTask) -> PlaceholderObject:
        """Algorithm 2 lines 4-6: enqueue the task, return instantly."""
        self.task_queue.put(task)
        return make_placeholder(task)

    def drain_results(self) -> List[ReconstructionResult]:
        """Algorithm 2 lines 8-10: non-blocking drain of whatever is ready."""
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def pause(self):
        """Detection/association pre-empts reconstruction (Sec. V)."""
        self._running.clear()

    def resume(self):
        self._running.set()

    def shutdown(self, timeout: Optional[float] = 5.0):
        for _ in self._workers:
            self.task_queue.put(_STOP)
        self._running.set()
        for w in self._workers:
            w.join(timeout=timeout)


def _selftest():
    """Runnable directly: python -m async_backend.pipeline"""
    def slow_reconstruct(task: ReconstructionTask) -> ReconstructionResult:
        time.sleep(0.2)   # stand-in for DeepSDF optimization + mesh extraction
        return ReconstructionResult(object_id=task.object_id, latent_code=np.zeros(8),
                                     pose=task.pose, mesh=None)

    backend = AsyncReconstructionBackend(slow_reconstruct, num_workers=2)

    t0 = time.time()
    placeholders = []
    for i in range(4):
        task = ReconstructionTask(object_id=i, class_name="Car",
                                   point_cloud=np.zeros((10, 3)), pose=np.eye(4),
                                   box=[float(i), 0, 0, 4.6, 1.9, 1.7, 0.0])
        placeholders.append(backend.submit_task(task))
    submit_elapsed = time.time() - t0
    assert submit_elapsed < 0.05, f"submit_task blocked for {submit_elapsed:.3f}s -- not async"
    assert len(placeholders) == 4

    deadline = time.time() + 3.0
    results = []
    while len(results) < 4 and time.time() < deadline:
        results.extend(backend.drain_results())
        time.sleep(0.05)
    assert len(results) == 4, f"expected 4 results, got {len(results)}"

    backend.shutdown()
    print(f"async_backend self-test OK: 4 tasks submitted non-blockingly "
          f"({submit_elapsed*1000:.1f}ms), all {len(results)} results drained")


if __name__ == "__main__":
    _selftest()
