"""Ablation study: does OpenCV preprocessing actually improve OCR?

For every degraded 'scan' in ``data/synthetic`` we run Tesseract under two
conditions and compute CER/WER against the ground truth:

    (A) raw            grayscale only, straight into Tesseract
    (B) preprocessed   full OpenCV chain (CLAHE, denoise, deskew, binarise)

We also sweep individual steps to attribute the improvement. Results are
written to ``data/synthetic/ocr_ablation_results.json`` and printed as a table.
This is the experiment behind the report's claim that preprocessing
significantly reduces CER on noisy documents before Tesseract.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from docai.config import OCRSettings, PreprocessSettings
from docai.evaluation import corpus_error_rates
from docai.ocr import run_ocr
from docai.preprocessing.opencv_ops import (
    apply_clahe, binarize, denoise, deskew, preprocess, to_grayscale,
)

DATA = Path(__file__).parents[1] / "data" / "synthetic"
OCR_CFG = OCRSettings(lang="eng", psm=6)  # psm 6 = uniform block of text


def _load(manifest):
    refs, imgs = [], []
    for entry in manifest:
        refs.append((DATA / entry["gt"]).read_text())
        imgs.append(cv2.imread(str(DATA / entry["noisy"])))
    return imgs, refs


def _ocr_all(imgs, transform):
    preds = []
    for img in imgs:
        proc = transform(img)
        preds.append(run_ocr(proc, OCR_CFG).text)
    return preds


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    imgs, refs = _load(manifest)

    # Condition A: raw grayscale (no cleaning).
    raw_preds = _ocr_all(imgs, to_grayscale)

    # Condition B: full OpenCV preprocessing chain.
    full_cfg = PreprocessSettings()
    full_preds = _ocr_all(imgs, lambda im: preprocess(im, full_cfg))

    # Per-step attribution (grayscale + one step at a time).
    def only(step):
        return lambda im: step(to_grayscale(im))

    step_conditions = {
        "grayscale_only": to_grayscale,
        "+clahe": only(apply_clahe),
        "+denoise": only(denoise),
        "+deskew": only(deskew),
        "+binarize": only(binarize),
        "full_chain": lambda im: preprocess(im, full_cfg),
    }

    results = {}
    for name, tf in step_conditions.items():
        preds = _ocr_all(imgs, tf)
        results[name] = corpus_error_rates(preds, refs)

    raw = corpus_error_rates(raw_preds, refs)
    full = corpus_error_rates(full_preds, refs)
    cer_impr = 100 * (raw["cer"] - full["cer"]) / raw["cer"] if raw["cer"] else 0.0
    wer_impr = 100 * (raw["wer"] - full["wer"]) / raw["wer"] if raw["wer"] else 0.0

    summary = {
        "raw_grayscale": raw,
        "full_preprocessing": full,
        "cer_relative_improvement_pct": round(cer_impr, 1),
        "wer_relative_improvement_pct": round(wer_impr, 1),
        "per_step": results,
        "n_documents": len(imgs),
    }
    out = DATA / "ocr_ablation_results.json"
    out.write_text(json.dumps(summary, indent=2))

    # Pretty print.
    print("\n=== OCR Ablation: impact of OpenCV preprocessing ===\n")
    print(f"{'condition':<18}{'CER':>10}{'WER':>10}")
    print("-" * 38)
    for name, m in results.items():
        print(f"{name:<18}{m['cer']:>10.4f}{m['wer']:>10.4f}")
    print("-" * 38)
    print(f"{'RAW grayscale':<18}{raw['cer']:>10.4f}{raw['wer']:>10.4f}")
    print(f"{'FULL preprocess':<18}{full['cer']:>10.4f}{full['wer']:>10.4f}")
    print(f"\nRelative CER improvement: {cer_impr:.1f}%")
    print(f"Relative WER improvement: {wer_impr:.1f}%")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
