"""Semantic indexing & retrieval of the document archive.

Turns each document's text into a dense sentence embedding and stores it in a
FAISS index so operators can search archives *by meaning* ("find the supplier
contract about penalty clauses") rather than by exact keyword.

Embeddings (sentence-transformers) and FAISS are imported lazily.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import SemanticSettings


class SemanticIndex:
    def __init__(self, cfg: SemanticSettings | None = None):
        self.cfg = cfg or SemanticSettings()
        self._index = None
        self._ids: list[str] = []
        self._meta: dict[str, dict] = {}

    @property
    def _embedder(self):
        if not hasattr(self, "_emb"):
            from sentence_transformers import SentenceTransformer

            self._emb = SentenceTransformer(self.cfg.embed_model)
        return self._emb

    def _new_index(self):
        import faiss

        # Inner-product on L2-normalised vectors == cosine similarity.
        return faiss.IndexFlatIP(self.cfg.dim)

    def _embed(self, texts: list[str]):
        import numpy as np

        vecs = self._embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def add(self, doc_ids: list[str], texts: list[str],
            metas: list[dict] | None = None) -> None:
        if self._index is None:
            self._index = self._new_index()
        vecs = self._embed(texts)
        self._index.add(vecs)
        metas = metas or [{} for _ in doc_ids]
        for did, meta in zip(doc_ids, metas):
            self._ids.append(did)
            self._meta[did] = meta

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._index is None or not self._ids:
            return []
        qv = self._embed([query])
        scores, idxs = self._index.search(qv, min(k, len(self._ids)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            did = self._ids[idx]
            results.append({"doc_id": did, "score": float(score),
                            "meta": self._meta.get(did, {})})
        return results

    def save(self, path: str | None = None) -> None:
        import faiss

        path = path or self.cfg.index_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, path)
        with open(path + ".meta.json", "w") as f:
            json.dump({"ids": self._ids, "meta": self._meta}, f)

    def load(self, path: str | None = None) -> None:
        import faiss

        path = path or self.cfg.index_path
        self._index = faiss.read_index(path)
        with open(path + ".meta.json") as f:
            payload = json.load(f)
        self._ids = payload["ids"]
        self._meta = payload["meta"]
