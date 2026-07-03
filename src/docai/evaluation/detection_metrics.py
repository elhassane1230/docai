"""Detection metrics: mean Average Precision (mAP) for the YOLO layout model.

Implements the standard VOC/COCO-style AP with a configurable IoU threshold:

  1. Sort predictions by descending confidence.
  2. Greedily match each prediction to the highest-IoU unmatched GT of the
     same class above the IoU threshold -> TP, else FP.
  3. Build the precision/recall curve, integrate to get AP per class.
  4. mAP = mean of per-class AP.

``mAP@0.5`` is the headline number reported for the visual-detection stage.
No torch dependency — pure Python/numpy so it runs anywhere.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..schemas import DetectedElement


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-2010 style AP: area under the monotonically-decreasing PR curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing (envelope).
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    preds: list[DetectedElement],
    gts: list[DetectedElement],
    iou_threshold: float = 0.5,
) -> float:
    """AP for a single class over one image (or a concatenated dataset)."""
    if not gts:
        return 0.0 if preds else 1.0
    preds = sorted(preds, key=lambda p: p.score, reverse=True)
    matched = [False] * len(gts)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, pred in enumerate(preds):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            iou = pred.bbox.iou(gt.bbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp[i] = 1
            matched[best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / len(gts)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return _ap_from_pr(recall, precision)


def mean_average_precision(
    preds: list[DetectedElement],
    gts: list[DetectedElement],
    iou_threshold: float = 0.5,
) -> dict:
    """mAP across all classes present in the ground truth."""
    preds_by_cls: dict[str, list[DetectedElement]] = defaultdict(list)
    gts_by_cls: dict[str, list[DetectedElement]] = defaultdict(list)
    for p in preds:
        preds_by_cls[p.label].append(p)
    for g in gts:
        gts_by_cls[g.label].append(g)

    per_class = {}
    for cls in gts_by_cls:
        per_class[str(cls)] = average_precision(
            preds_by_cls.get(cls, []), gts_by_cls[cls], iou_threshold
        )
    m = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return {"mAP": m, "iou_threshold": iou_threshold, "per_class": per_class}
