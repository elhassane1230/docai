"""End-to-end Document AI pipeline orchestrator.

Ties the stages together into one call:

    raw scan  ->  OpenCV preprocess  ->  Tesseract OCR
                                    \\->  YOLO layout detection
              ->  DistilBERT classification  ->  BERT NER
              ->  structured DocumentResult

Heavy stages (vision, NLP) are **optional and lazily constructed**. The
pipeline degrades gracefully: on a box with only OpenCV + Tesseract you still
get preprocessing + OCR + a TF-IDF fallback classification. Enable the deep
stages by passing the corresponding flags once the models are available.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import Settings, get_settings
from .preprocessing.opencv_ops import preprocess
from .ocr.tesseract_ocr import run_ocr
from .schemas import Classification, DocumentResult


class DocumentPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        enable_preprocess: bool = True,
        enable_vision: bool = False,
        enable_classification: bool = False,
        enable_ner: bool = False,
        classifier=None,     # inject a fitted TfidfBaseline / TransformerClassifier
        detector=None,       # inject a YOLODetector
        ner=None,            # inject an NERExtractor
    ):
        self.cfg = settings or get_settings()
        self.enable_preprocess = enable_preprocess
        self.enable_vision = enable_vision
        self.enable_classification = enable_classification
        self.enable_ner = enable_ner
        self._classifier = classifier
        self._detector = detector
        self._ner = ner

    # -- lazy component builders -------------------------------------------- #
    @property
    def classifier(self):
        if self._classifier is None and self.enable_classification:
            from .nlp import TransformerClassifier

            self._classifier = TransformerClassifier(self.cfg.nlp)
        return self._classifier

    @property
    def detector(self):
        if self._detector is None and self.enable_vision:
            from .vision import YOLODetector

            self._detector = YOLODetector(self.cfg.vision)
        return self._detector

    @property
    def ner(self):
        if self._ner is None and self.enable_ner:
            from .nlp import NERExtractor

            self._ner = NERExtractor(self.cfg.nlp)
        return self._ner

    # -- I/O ---------------------------------------------------------------- #
    @staticmethod
    def _load_image(source: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return img

    # -- main entrypoint ---------------------------------------------------- #
    def process(self, source: str | Path | np.ndarray,
                doc_id: Optional[str] = None) -> DocumentResult:
        doc_id = doc_id or uuid.uuid4().hex[:12]
        timings: dict[str, float] = {}

        img = self._load_image(source)

        # 1) Preprocess -----------------------------------------------------
        if self.enable_preprocess:
            t = time.perf_counter()
            proc = preprocess(img, self.cfg.preprocess)
            timings["preprocess_ms"] = (time.perf_counter() - t) * 1000
        else:
            proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        result = DocumentResult(doc_id=doc_id)

        # 2) OCR ------------------------------------------------------------
        t = time.perf_counter()
        ocr = run_ocr(proc, self.cfg.ocr)
        timings["ocr_ms"] = (time.perf_counter() - t) * 1000
        result.ocr = ocr
        result.text = ocr.text

        # 3) Visual layout detection (optional) -----------------------------
        if self.enable_vision and self.detector is not None:
            t = time.perf_counter()
            result.elements = self.detector.detect(img)
            timings["vision_ms"] = (time.perf_counter() - t) * 1000

        # 4) Classification (optional) --------------------------------------
        if self.classifier is not None and result.text.strip():
            t = time.perf_counter()
            result.classification = self._classify(result.text)
            timings["classification_ms"] = (time.perf_counter() - t) * 1000

        # 5) NER (optional) -------------------------------------------------
        if self.enable_ner and self.ner is not None and result.text.strip():
            t = time.perf_counter()
            result.entities = self.ner.extract(result.text)
            timings["ner_ms"] = (time.perf_counter() - t) * 1000

        result.timings_ms = {k: round(v, 2) for k, v in timings.items()}
        result.meta = {"ocr_mean_confidence": f"{ocr.mean_confidence:.1f}"}
        return result

    def _classify(self, text: str) -> Classification:
        # Works for both TfidfBaseline and TransformerClassifier (duck typing).
        return self.classifier.predict(text)
