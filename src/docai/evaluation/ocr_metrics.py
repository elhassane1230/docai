"""OCR quality metrics: Character Error Rate (CER) and Word Error Rate (WER).

Both are edit-distance based:

    CER = (S + D + I) / N_chars      WER = (S + D + I) / N_words

where S/D/I are substitutions/deletions/insertions from the Levenshtein
alignment between prediction and ground truth, and N is the reference length.
Lower is better; 0.0 is perfect. Values can exceed 1.0 when the prediction is
much longer than the reference (many insertions).
"""
from __future__ import annotations

from dataclasses import dataclass

import editdistance


@dataclass
class ErrorRate:
    rate: float
    edits: int
    ref_len: int

    def __float__(self) -> float:
        return self.rate


def _normalise(s: str) -> str:
    # Collapse whitespace so formatting differences don't inflate the score.
    return " ".join(s.split())


def cer(prediction: str, reference: str) -> ErrorRate:
    """Character Error Rate."""
    ref = _normalise(reference)
    pred = _normalise(prediction)
    if not ref:
        return ErrorRate(0.0 if not pred else 1.0, len(pred), 0)
    dist = editdistance.eval(pred, ref)
    return ErrorRate(dist / len(ref), dist, len(ref))


def wer(prediction: str, reference: str) -> ErrorRate:
    """Word Error Rate."""
    ref = _normalise(reference).split()
    pred = _normalise(prediction).split()
    if not ref:
        return ErrorRate(0.0 if not pred else 1.0, len(pred), 0)
    dist = editdistance.eval(pred, ref)
    return ErrorRate(dist / len(ref), dist, len(ref))


def corpus_error_rates(predictions: list[str], references: list[str]) -> dict:
    """Micro-averaged CER/WER over a corpus (sum of edits / sum of ref len)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    c_edits = c_len = w_edits = w_len = 0
    for p, r in zip(predictions, references):
        c = cer(p, r)
        w = wer(p, r)
        c_edits += c.edits
        c_len += c.ref_len
        w_edits += w.edits
        w_len += w.ref_len
    return {
        "cer": c_edits / c_len if c_len else 0.0,
        "wer": w_edits / w_len if w_len else 0.0,
        "n_documents": len(predictions),
    }
