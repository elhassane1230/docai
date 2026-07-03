"""End-to-end demo that runs everything runnable without a GPU and reports
real metrics for every evaluation axis in the project:

  * OCR:            CER / WER (raw vs preprocessed) on the noisy corpus
  * Classification: Precision / Recall / F1 of a TF-IDF baseline classifying
                    the *OCR'd* noisy text (true end-to-end, errors and all)
  * Detection:      mAP@0.5 on a synthetic detection example
  * NER:            span-level P/R/F1 on a synthetic example

Writes results/results.json. The Transformer/YOLO models are covered by the
production code paths; this demo proves the pipeline + metric plumbing end to
end on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from docai.config import OCRSettings, PreprocessSettings, Settings
from docai.evaluation import (
    classification_metrics, corpus_error_rates, mean_average_precision,
    ner_metrics,
)
from docai.nlp import TfidfBaseline
from docai.ocr import run_ocr
from docai.pipeline import DocumentPipeline
from docai.preprocessing.opencv_ops import preprocess, to_grayscale
from docai.schemas import BBox, DetectedElement, Entity, ElementType

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"
OCR_CFG = OCRSettings(lang="eng", psm=6)


# --------------------------------------------------------------------------- #
def eval_ocr(manifest) -> dict:
    imgs = [cv2.imread(str(DATA / e["noisy"])) for e in manifest]
    refs = [(DATA / e["gt"]).read_text() for e in manifest]
    raw = corpus_error_rates([run_ocr(to_grayscale(i), OCR_CFG).text for i in imgs], refs)
    pre_cfg = PreprocessSettings()
    pre = corpus_error_rates(
        [run_ocr(preprocess(i, pre_cfg), OCR_CFG).text for i in imgs], refs
    )
    return {"raw": raw, "preprocessed": pre,
            "cer_improvement_pct": round(100 * (raw["cer"] - pre["cer"]) / raw["cer"], 1)}


# --------------------------------------------------------------------------- #
def eval_classification_end_to_end(manifest) -> dict:
    """Train a TF-IDF classifier on clean ground-truth text, then classify the
    *OCR output of the noisy scans* — the honest end-to-end setting."""
    # Ground-truth text + labels.
    texts = [(DATA / e["gt"]).read_text() for e in manifest]
    labels = [e["type"] for e in manifest]

    # Stratified split: for each class, first half -> train, second -> test.
    from collections import defaultdict

    by_class: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    tr_idx, te_idx = [], []
    for lab, idxs in by_class.items():
        cut = max(1, len(idxs) // 2)
        tr_idx.extend(idxs[:cut])
        te_idx.extend(idxs[cut:] or idxs[:1])

    clf = TfidfBaseline(ngram_max=2).fit(
        [texts[i] for i in tr_idx], [labels[i] for i in tr_idx]
    )

    # Test features = OCR of the noisy scans (real pipeline output).
    pipe = DocumentPipeline(Settings(), enable_preprocess=True)
    ocr_texts = [pipe.process(str(DATA / manifest[i]["noisy"])).text for i in te_idx]
    y_true = [labels[i] for i in te_idx]
    y_pred = clf.predict_labels(ocr_texts)

    return {
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "metrics": classification_metrics(y_true, y_pred),
        "note": "features are OCR'd noisy scans, not clean text",
    }


# --------------------------------------------------------------------------- #
def demo_detection() -> dict:
    """Synthetic mAP example: 3 GT elements, detector gets 2 right + 1 FP."""
    gt = [
        DetectedElement(label=ElementType.LOGO,
                        bbox=BBox(x1=10, y1=10, x2=110, y2=60), score=1.0),
        DetectedElement(label=ElementType.SIGNATURE,
                        bbox=BBox(x1=400, y1=500, x2=560, y2=560), score=1.0),
        DetectedElement(label=ElementType.TABLE,
                        bbox=BBox(x1=50, y1=200, x2=550, y2=420), score=1.0),
    ]
    preds = [
        DetectedElement(label=ElementType.LOGO,          # good match
                        bbox=BBox(x1=12, y1=11, x2=108, y2=61), score=0.94),
        DetectedElement(label=ElementType.TABLE,         # good match
                        bbox=BBox(x1=55, y1=205, x2=548, y2=418), score=0.88),
        DetectedElement(label=ElementType.SIGNATURE,     # localisation miss (low IoU)
                        bbox=BBox(x1=300, y1=460, x2=380, y2=500), score=0.55),
    ]
    return mean_average_precision(preds, gt, iou_threshold=0.5)


def demo_ner() -> dict:
    """Synthetic span-level NER example."""
    gold = [[
        Entity(text="Acme Logistics", label="ORG", start=0, end=14, score=1.0),
        Entity(text="Rotterdam", label="LOC", start=30, end=39, score=1.0),
        Entity(text="March 14, 2024", label="DATE", start=50, end=64, score=1.0),
    ]]
    pred = [[
        Entity(text="Acme Logistics", label="ORG", start=0, end=14, score=0.99),
        Entity(text="Rotterdam", label="LOC", start=30, end=39, score=0.97),
        Entity(text="March 14", label="DATE", start=50, end=58, score=0.80),  # wrong span
    ]]
    return ner_metrics(gold, pred)


# --------------------------------------------------------------------------- #
def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    RESULTS.mkdir(exist_ok=True)

    report = {
        "ocr": eval_ocr(manifest),
        "classification_end_to_end": eval_classification_end_to_end(manifest),
        "detection_example": demo_detection(),
        "ner_example": demo_ner(),
    }
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))

    print("\n===== DocAI end-to-end results =====\n")
    o = report["ocr"]
    print(f"OCR    raw CER={o['raw']['cer']:.4f}  ->  preprocessed CER="
          f"{o['preprocessed']['cer']:.4f}  ({o['cer_improvement_pct']:+.1f}%)")
    c = report["classification_end_to_end"]["metrics"]
    print(f"CLASS  accuracy={c['accuracy']:.3f}  macro-F1={c['macro']['f1']:.3f}")
    d = report["detection_example"]
    print(f"DETECT mAP@0.5={d['mAP']:.3f}  per_class={d['per_class']}")
    n = report["ner_example"]
    print(f"NER    P={n['precision']:.3f} R={n['recall']:.3f} F1={n['f1']:.3f}")
    print(f"\nSaved -> {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
