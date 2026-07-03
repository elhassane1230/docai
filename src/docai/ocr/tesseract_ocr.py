"""Tesseract OCR wrapper producing structured, word-level output.

We use ``image_to_data`` (not ``image_to_string``) so every recognised word
comes back with a bounding box and a confidence score. That lets downstream
stages reason about *where* text is — needed to associate text with the
YOLO-detected regions (a table cell vs. a title vs. body text).
"""
from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from ..config import OCRSettings
from ..schemas import BBox, OCRResult, OCRWord


def _config(cfg: OCRSettings) -> str:
    return f"--oem {cfg.oem} --psm {cfg.psm}"


def run_ocr(img: np.ndarray, cfg: OCRSettings | None = None) -> OCRResult:
    """Recognise text in ``img`` and return words + boxes + confidences."""
    cfg = cfg or OCRSettings()
    data = pytesseract.image_to_data(
        img, lang=cfg.lang, config=_config(cfg), output_type=Output.DICT,
    )

    words: list[OCRWord] = []
    confidences: list[float] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        if conf < cfg.min_confidence:
            continue
        x, y, w, h = (data["left"][i], data["top"][i],
                      data["width"][i], data["height"][i])
        words.append(
            OCRWord(
                text=text,
                bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h),
                confidence=conf,
            )
        )
        confidences.append(conf)

    # Reconstruct reading-order text using Tesseract's line/block structure.
    full_text = _reconstruct_text(data, cfg)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return OCRResult(text=full_text, words=words, mean_confidence=mean_conf)


def _reconstruct_text(data: dict, cfg: OCRSettings) -> str:
    """Join words back into lines/paragraphs using block/par/line indices."""
    lines: dict[tuple, list[str]] = {}
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        if not text or conf < cfg.min_confidence:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(text)
    ordered = [" ".join(ws) for _, ws in sorted(lines.items())]
    return "\n".join(ordered)
