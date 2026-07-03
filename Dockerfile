# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# DocAI inference API image.
# Ships the CPU core (OpenCV + Tesseract + FastAPI). To enable the deep-learning
# stages (YOLO / BERT), build with:  --build-arg INSTALL_DL=1  (and use a GPU
# base image such as pytorch/pytorch for production).
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps: Tesseract OCR engine + language packs, and OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dl.txt ./
RUN pip install -r requirements.txt

ARG INSTALL_DL=0
RUN if [ "$INSTALL_DL" = "1" ]; then pip install -r requirements-dl.txt; fi

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

EXPOSE 8000

# Feature flags default to OCR-only; override at runtime.
ENV DOCAI_ENABLE_PREPROCESS=1 \
    DOCAI_ENABLE_VISION=0 \
    DOCAI_ENABLE_CLS=0 \
    DOCAI_ENABLE_NER=0

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "docai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
