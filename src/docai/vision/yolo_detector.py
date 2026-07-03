"""Visual element detection with YOLO (Ultralytics).

Detects document layout elements — logos, signatures, tables, figures, text
blocks, titles, stamps — on high-resolution scans. In production this loads a
checkpoint fine-tuned on an annotated document corpus (see ``train_yolo.py``);
the class labels map onto :class:`docai.schemas.ElementType`.

The YOLOv5/YOLOv8 weights and the ``ultralytics`` runtime are imported lazily
so the package stays importable without them.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ..config import VisionSettings
from ..schemas import BBox, DetectedElement, ElementType

# Maps the trained model's class indices to our canonical element types.
# Override via the fine-tuned model's own ``names`` when available.
DEFAULT_CLASS_MAP = {
    0: ElementType.TEXT,
    1: ElementType.TITLE,
    2: ElementType.TABLE,
    3: ElementType.FIGURE,
    4: ElementType.LOGO,
    5: ElementType.SIGNATURE,
    6: ElementType.STAMP,
}


class YOLODetector:
    def __init__(self, cfg: VisionSettings | None = None,
                 class_map: dict[int, ElementType] | None = None):
        self.cfg = cfg or VisionSettings()
        self.class_map = class_map or DEFAULT_CLASS_MAP

    @cached_property
    def _model(self):
        from ultralytics import YOLO  # lazy heavy import

        model = YOLO(self.cfg.weights)
        return model

    def detect(self, img: np.ndarray) -> list[DetectedElement]:
        """Run detection and return canonical :class:`DetectedElement` objects."""
        results = self._model.predict(
            img,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.img_size,
            device=self.cfg.device,
            verbose=False,
        )
        elements: list[DetectedElement] = []
        for res in results:
            names = getattr(res, "names", None) or {}
            for box in res.boxes:
                cls_idx = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                label = self._resolve_label(cls_idx, names)
                if label is None:
                    continue
                elements.append(
                    DetectedElement(
                        label=label,
                        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        score=score,
                    )
                )
        return elements

    def _resolve_label(self, cls_idx: int, names: dict) -> ElementType | None:
        # Prefer the checkpoint's own class names if they match our enum.
        name = names.get(cls_idx)
        if name:
            try:
                return ElementType(name.lower())
            except ValueError:
                pass
        return self.class_map.get(cls_idx)
