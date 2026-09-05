"""Lightweight JSONL metrics logger for training runs."""
import json
import os
import threading
import time


class MetricsLogger:
    """
    Appends one JSON record per line to a log file.
    Thread-safe so it can be called from a training loop without locks on the
    caller side.

    Record types written:
        config  — training hyper-parameters, written once at startup
        step    — per-step loss values
        epoch   — per-epoch averages and timing
        done    — written when training finishes normally
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _write(self, record: dict):
        record.setdefault("timestamp", time.time())
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def log_config(self, **kwargs):
        self._write({"type": "config", **kwargs})

    def log_step(self, epoch: int, step: int, total_steps: int,
                 loss: float, cls_loss: float, reg_loss: float):
        self._write({
            "type":       "step",
            "epoch":      epoch,
            "step":       step,
            "total_steps": total_steps,
            "loss":       round(loss,     5),
            "cls_loss":   round(cls_loss, 5),
            "reg_loss":   round(reg_loss, 5),
        })

    def log_epoch(self, epoch: int, total_epochs: int,
                  avg_loss: float, avg_cls: float, avg_reg: float,
                  duration_s: float):
        self._write({
            "type":         "epoch",
            "epoch":        epoch,
            "total_epochs": total_epochs,
            "avg_loss":     round(avg_loss, 5),
            "avg_cls":      round(avg_cls,  5),
            "avg_reg":      round(avg_reg,  5),
            "duration_s":   round(duration_s, 2),
        })

    def log_done(self):
        self._write({"type": "done"})
