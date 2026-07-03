from docai.evaluation import (
    cer, wer, corpus_error_rates, mean_average_precision,
    classification_metrics, ner_metrics,
)
from docai.schemas import BBox, DetectedElement, Entity, ElementType


# --- OCR metrics ----------------------------------------------------------- #
def test_cer_perfect():
    assert float(cer("hello world", "hello world")) == 0.0


def test_cer_one_substitution():
    # "cat" vs "car": 1 edit / 3 chars.
    assert abs(float(cer("cat", "car")) - 1 / 3) < 1e-9


def test_wer_counts_words():
    assert abs(float(wer("the quick fox", "the slow fox")) - 1 / 3) < 1e-9


def test_corpus_micro_average():
    out = corpus_error_rates(["abc", "abcd"], ["abc", "abce"])
    assert out["n_documents"] == 2
    assert 0.0 <= out["cer"] <= 1.0


# --- Detection mAP --------------------------------------------------------- #
def test_map_perfect_detection():
    gt = [DetectedElement(label=ElementType.LOGO,
                          bbox=BBox(x1=0, y1=0, x2=10, y2=10), score=1.0)]
    pred = [DetectedElement(label=ElementType.LOGO,
                            bbox=BBox(x1=0, y1=0, x2=10, y2=10), score=0.9)]
    assert mean_average_precision(pred, gt)["mAP"] == 1.0


def test_map_wrong_class_is_zero():
    gt = [DetectedElement(label=ElementType.LOGO,
                          bbox=BBox(x1=0, y1=0, x2=10, y2=10), score=1.0)]
    pred = [DetectedElement(label=ElementType.TABLE,
                            bbox=BBox(x1=0, y1=0, x2=10, y2=10), score=0.9)]
    assert mean_average_precision(pred, gt)["mAP"] == 0.0


def test_iou_computation():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    b = BBox(x1=5, y1=0, x2=15, y2=10)
    # intersection 50, union 150 -> 1/3
    assert abs(a.iou(b) - 1 / 3) < 1e-9


# --- NLP metrics ----------------------------------------------------------- #
def test_classification_perfect():
    m = classification_metrics(["a", "b", "a"], ["a", "b", "a"])
    assert m["accuracy"] == 1.0
    assert m["macro"]["f1"] == 1.0


def test_ner_span_level():
    gold = [[Entity(text="X", label="ORG", start=0, end=1, score=1.0)]]
    pred = [[Entity(text="X", label="ORG", start=0, end=1, score=0.9)]]
    m = ner_metrics(gold, pred)
    assert m["f1"] == 1.0


def test_ner_wrong_span_penalised():
    gold = [[Entity(text="ABC", label="DATE", start=0, end=3, score=1.0)]]
    pred = [[Entity(text="AB", label="DATE", start=0, end=2, score=0.9)]]
    m = ner_metrics(gold, pred)
    assert m["f1"] == 0.0
