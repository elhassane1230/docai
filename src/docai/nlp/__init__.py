"""NLP components. Transformer models are imported lazily to avoid a hard
torch/transformers dependency for users who only need OCR + metrics."""
from .baseline_tfidf import TfidfBaseline  # noqa: F401

__all__ = ["TfidfBaseline", "TransformerClassifier", "NERExtractor"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "TransformerClassifier":
        from .classifier import TransformerClassifier
        return TransformerClassifier
    if name == "NERExtractor":
        from .ner import NERExtractor
        return NERExtractor
    raise AttributeError(name)
