"""A lightweight TF-IDF + Linear SVM classification baseline.

Why keep a classical baseline next to the Transformers?
  * It trains in seconds on CPU with no GPU or model download — it is the
    control condition in the classification ablation (how much does BERT
    actually buy over a strong bag-of-words baseline?).
  * It is a safe fallback path for the API when the heavy model is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..schemas import Classification


@dataclass
class TfidfBaseline:
    """Trainable TF-IDF + LinearSVC document classifier."""

    max_features: int = 20000
    ngram_max: int = 2

    def __post_init__(self):
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=(1, self.ngram_max),
                    sublinear_tf=True,
                    strip_accents="unicode",
                )),
                ("clf", LinearSVC()),
            ]
        )
        self._fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfBaseline":
        self.model.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, text: str) -> Classification:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict()")
        label = self.model.predict([text])[0]
        # LinearSVC exposes signed distances, not probabilities; squash them.
        margins = self.model.decision_function([text])[0]
        classes = list(self.model.named_steps["clf"].classes_)
        if hasattr(margins, "__len__"):
            import numpy as np

            exp = np.exp(margins - np.max(margins))
            probs = exp / exp.sum()
            scores = {c: float(p) for c, p in zip(classes, probs)}
        else:  # binary
            scores = {classes[1]: 1.0, classes[0]: 0.0}
        return Classification(label=label, score=scores.get(label, 1.0),
                              all_scores=scores)

    def predict_labels(self, texts: list[str]) -> list[str]:
        return list(self.model.predict(texts))
