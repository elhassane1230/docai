"""Document classification with BERT / DistilBERT (Hugging Face Transformers).

Design notes
------------
* ``transformers``/``torch`` are imported *lazily* inside methods so the rest of
  the package (preprocessing, OCR, metrics, API schema) imports fine on a
  machine without a deep-learning stack.
* The class works in two modes:
    - a fine-tuned sequence-classification checkpoint (production), or
    - zero-shot via an NLI model when no labelled data exists yet (cold start).
* ``DistilBERT`` is the default because it delivers ~97% of BERT's accuracy at
  ~40% lower latency — see ``scripts`` / the ablation study.
"""
from __future__ import annotations

from functools import cached_property

from ..config import NLPSettings
from ..schemas import Classification


class TransformerClassifier:
    def __init__(self, cfg: NLPSettings | None = None, zero_shot: bool = False):
        self.cfg = cfg or NLPSettings()
        self.zero_shot = zero_shot

    @cached_property
    def _pipeline(self):
        # Lazy heavy import.
        from transformers import pipeline

        if self.zero_shot:
            return pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1 if self.cfg.device == "cpu" else 0,
            )
        return pipeline(
            "text-classification",
            model=self.cfg.classifier_model,
            top_k=None,
            device=-1 if self.cfg.device == "cpu" else 0,
        )

    def predict(self, text: str) -> Classification:
        text = text[: self.cfg.max_length * 6]  # rough char budget
        if self.zero_shot:
            res = self._pipeline(text, candidate_labels=self.cfg.labels)
            scores = dict(zip(res["labels"], res["scores"]))
            top = res["labels"][0]
            return Classification(
                label=top, score=float(res["scores"][0]), all_scores=scores
            )
        res = self._pipeline(text)[0]  # list[{label, score}]
        scores = {d["label"]: float(d["score"]) for d in res}
        top = max(scores, key=scores.get)
        return Classification(label=top, score=scores[top], all_scores=scores)

    def predict_batch(self, texts: list[str]) -> list[Classification]:
        return [self.predict(t) for t in texts]
