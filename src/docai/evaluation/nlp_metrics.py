"""NLP metrics for classification and NER.

Classification: macro / micro / weighted precision, recall, F1 (via sklearn).
NER: *span-level* precision/recall/F1 — an entity counts as correct only when
its label AND its (start, end) span both match exactly. This is stricter (and
more honest) than token-level accuracy, which is inflated by the many trivial
'O' tokens.
"""
from __future__ import annotations

from sklearn.metrics import precision_recall_fscore_support

from ..schemas import Entity


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Precision / Recall / F1 for document classification."""
    out = {}
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[avg] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    out["accuracy"] = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    return out


def _span_set(entities: list[Entity]) -> set[tuple[str, int, int]]:
    return {(e.label, e.start, e.end) for e in entities}


def ner_metrics(
    true_entities: list[list[Entity]],
    pred_entities: list[list[Entity]],
) -> dict:
    """Span-level P/R/F1 micro-averaged over a list of documents."""
    tp = fp = fn = 0
    for gold, pred in zip(true_entities, pred_entities):
        gset, pset = _span_set(gold), _span_set(pred)
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }
