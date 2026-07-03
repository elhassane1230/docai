"""Named-Entity Recognition with a BERT token-classification model.

Wraps the Hugging Face ``token-classification`` pipeline with aggregation so
sub-word tokens are merged back into whole entities with correct character
offsets (needed for downstream highlighting and semantic linking).
"""
from __future__ import annotations

from functools import cached_property

from ..config import NLPSettings
from ..schemas import Entity


class NERExtractor:
    def __init__(self, cfg: NLPSettings | None = None):
        self.cfg = cfg or NLPSettings()

    @cached_property
    def _pipeline(self):
        from transformers import pipeline  # lazy heavy import

        return pipeline(
            "token-classification",
            model=self.cfg.ner_model,
            aggregation_strategy="simple",  # merge B-/I- subwords into spans
            device=-1 if self.cfg.device == "cpu" else 0,
        )

    def extract(self, text: str) -> list[Entity]:
        raw = self._pipeline(text)
        entities: list[Entity] = []
        for ent in raw:
            entities.append(
                Entity(
                    text=ent["word"],
                    label=ent["entity_group"],
                    start=int(ent["start"]),
                    end=int(ent["end"]),
                    score=float(ent["score"]),
                )
            )
        return entities
