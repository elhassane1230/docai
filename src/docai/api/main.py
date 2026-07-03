"""FastAPI real-time inference service for the Document AI pipeline.

Endpoints
---------
GET  /health              liveness/readiness probe
GET  /capabilities        which pipeline stages are currently enabled
POST /process             upload an image -> full DocumentResult (JSON)
POST /ocr                 upload an image -> OCR-only (fast path)
POST /search              semantic search across the indexed archive

The heavy models load lazily on first use, so the container starts instantly
and only pays the memory cost for stages that are actually enabled via env vars
(``DOCAI_*``). Designed to sit behind the internal operations UI.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..config import get_settings
from ..pipeline import DocumentPipeline
from ..schemas import DocumentResult

settings = get_settings()

app = FastAPI(
    title="DocAI Inference API",
    version="0.1.0",
    description="Vision + NLP document understanding pipeline.",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Feature flags come from env so ops can turn deep stages on/off per deployment.
_pipeline = DocumentPipeline(
    settings,
    enable_preprocess=os.getenv("DOCAI_ENABLE_PREPROCESS", "1") == "1",
    enable_vision=os.getenv("DOCAI_ENABLE_VISION", "0") == "1",
    enable_classification=os.getenv("DOCAI_ENABLE_CLS", "0") == "1",
    enable_ner=os.getenv("DOCAI_ENABLE_NER", "0") == "1",
)


def _read_upload(file: UploadFile) -> np.ndarray:
    raw = file.file.read()
    max_bytes = settings.api.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.api.max_upload_mb} MB")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image (supported: png/jpg/tiff)")
    return img


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/capabilities")
def capabilities() -> dict:
    return {
        "preprocess": _pipeline.enable_preprocess,
        "vision": _pipeline.enable_vision,
        "classification": _pipeline.enable_classification,
        "ner": _pipeline.enable_ner,
        "ocr_lang": settings.ocr.lang,
    }


@app.post("/process", response_model=DocumentResult)
def process(file: UploadFile = File(...)) -> DocumentResult:
    """Full pipeline: preprocess -> OCR -> (vision) -> (classify) -> (NER)."""
    img = _read_upload(file)
    return _pipeline.process(img, doc_id=file.filename)


class OCRResponse(BaseModel):
    doc_id: str
    text: str
    mean_confidence: float
    n_words: int


@app.post("/ocr", response_model=OCRResponse)
def ocr_only(file: UploadFile = File(...)) -> OCRResponse:
    """Fast path: preprocessing + OCR only (no deep models loaded)."""
    img = _read_upload(file)
    res = _pipeline.process(img, doc_id=file.filename)
    return OCRResponse(
        doc_id=res.doc_id,
        text=res.text,
        mean_confidence=res.ocr.mean_confidence if res.ocr else 0.0,
        n_words=len(res.ocr.words) if res.ocr else 0,
    )


class SearchQuery(BaseModel):
    query: str
    k: int = 5


@app.post("/search")
def search(q: SearchQuery) -> dict:
    """Semantic search over the indexed archive (requires a built FAISS index)."""
    try:
        from ..semantic import SemanticIndex

        index = SemanticIndex(settings.semantic)
        index.load()
        return {"results": index.search(q.query, q.k)}
    except FileNotFoundError:
        raise HTTPException(404, "No semantic index found. Build one first.")
