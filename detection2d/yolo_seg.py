"""YOLO11x-seg wrapper (paper's Det2D choice, Sec. IV-A / VII-A) + a null
stand-in for bring-up before real weights/images are available.

Requires `ultralytics` (not in this repo's venv yet -- pip install ultralytics)
and a checkpoint fine-tuned on {car, bus, motorcycle, pedestrian}; the stock
COCO-pretrained yolo11x-seg.pt covers car/bus/motorcycle/person out of the box
(COCO has no distinct "pedestrian" class -- "person" is the closest match and
is remapped below), so it's a usable starting point before any fine-tuning.
"""
from typing import Dict, List, Optional, Sequence, Union
import numpy as np

from .interface import Detection2D, Detector2D, CLASS_NAMES

# COCO class name -> our taxonomy. Anything not listed is ignored.
_COCO_TO_OURS = {
    "car": "Car",
    "bus": "Bus",
    "truck": "Car",       # nearest match; re-evaluate once fine-tuned
    "motorcycle": "Motorcycle",
    "person": "Pedestrian",
}

ScoreThreshold = Union[float, Sequence[float]]


class YoloSegDetector2D(Detector2D):
    def __init__(
        self,
        weights: str = "yolo11x-seg.pt",
        class_score_thresholds: Optional[ScoreThreshold] = 0.25,
        device: Optional[str] = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YoloSegDetector2D needs the `ultralytics` package: "
                "pip install ultralytics"
            ) from e

        self.model = YOLO(weights)
        self.device = device
        self.class_score_thresholds = class_score_thresholds
        # If `weights` is a stock COCO checkpoint, self.model.names holds COCO
        # names; if it's fine-tuned on our 4 classes, names already match
        # CLASS_NAMES and _remap becomes an identity lookup.
        self._names: Dict[int, str] = self.model.names

    def _threshold_for(self, class_name: str) -> float:
        thr = self.class_score_thresholds
        if isinstance(thr, (list, tuple)):
            return thr[CLASS_NAMES.index(class_name)]
        return thr if thr is not None else 0.25

    def _remap(self, coco_name: str) -> Optional[str]:
        mapped = _COCO_TO_OURS.get(coco_name.lower())
        return mapped if mapped in CLASS_NAMES else (coco_name if coco_name in CLASS_NAMES else None)

    def run(self, image: np.ndarray) -> List[Detection2D]:
        results = self.model.predict(image, device=self.device, verbose=False)[0]
        dets: List[Detection2D] = []
        if results.masks is None:
            return dets

        masks = results.masks.data.cpu().numpy()          # (n, h, w) float in [0,1]
        boxes = results.boxes
        for i in range(len(boxes)):
            raw_name = self._names[int(boxes.cls[i])]
            class_name = self._remap(raw_name)
            if class_name is None:
                continue
            score = float(boxes.conf[i])
            if score < self._threshold_for(class_name):
                continue
            dets.append(Detection2D(
                mask=masks[i] > 0.5,
                class_id=CLASS_NAMES.index(class_name),
                class_name=class_name,
                score=score,
                bbox=boxes.xyxy[i].cpu().numpy(),
            ))
        return dets


class NullDetector2D(Detector2D):
    """Bring-up stand-in for when no 2D model/weights/images are wired yet
    (e.g. this repo's current nuScenes stream has no camera images — see
    datasets/nuscenes_stream.py). Always returns no detections, which makes
    downstream WP4 association a clean no-op rather than a crash, so the
    rest of the pipeline (WP5-WP10) stays testable in isolation.
    """
    def run(self, image: np.ndarray) -> List[Detection2D]:
        return []
