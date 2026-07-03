# Architecture

## Pipeline stages

```
                    ┌──────────────────────────────────────────────┐
   raw scan  ─────▶ │ 1. OpenCV preprocess                         │
   (png/jpg/tiff)   │    grayscale → denoise → CLAHE? → deskew →   │
                    │    border-crop? → adaptive binarize          │
                    └───────────────┬──────────────────────────────┘
                                    │ clean raster
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     
   ┌────────────────────┐  ┌──────────────────┐
   │ 2. YOLO detection  │  │ 3. Tesseract OCR │
   │ logo / signature / │  │ word boxes +     │
   │ table / figure /   │  │ confidences +    │
   │ title / text /stamp│  │ reading order    │
   └─────────┬──────────┘  └────────┬─────────┘
             │ elements             │ text
             └───────────┬──────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │ 4. NLP                                  │
        │   DistilBERT → document class          │
        │   BERT       → named entities (NER)    │
        │   MiniLM     → embedding → FAISS index │
        └───────────────────┬────────────────────┘
                            ▼
                  DocumentResult (typed JSON)
                            ▼
                  FastAPI  ──▶  Streamlit ops UI
```

## Why these choices

### Preprocessing before OCR
Tesseract's LSTM engine is robust to mild noise but fails on **uneven
illumination** — a lighting gradient across a page defeats a *global*
threshold. The chain fixes geometry (deskew) and local contrast (adaptive
thresholding) so OCR sees clean, flat, binary glyphs. The ablation quantifies
each step's contribution (see `docs/RESULTS.md`).

### YOLO for layout
Document elements (logos, signatures, tables, stamps) are localised objects,
which is exactly what one-stage detectors excel at. Transfer learning from COCO
converges fast because low-level features transfer; only the detection head
specialises. Mosaic + copy-paste augmentation compensate for rare classes
(signatures, stamps) that are under-represented in most archives.

### DistilBERT over BERT for classification
DistilBERT keeps ~97% of BERT-base's accuracy with ~40% lower latency and ~40%
fewer parameters — the right default for a *real-time* API. When even lower
latency is required, the classifier is exported to **ONNX + INT8** (see
`docai/optimize/`), compounding the win with negligible accuracy loss.

### Span-level NER
Entity offsets are preserved end-to-end so downstream consumers can highlight
spans and link entities to layout regions. Evaluation is span-level (label +
exact boundaries), which is stricter and more honest than token accuracy.

### FAISS semantic index
Operators search archives *by meaning* ("supplier contract with penalty
clauses"), not exact keywords. Documents are embedded with a sentence encoder
and stored in a cosine-similarity FAISS index.

## Data contracts

Every stage communicates through `pydantic` models in `docai/schemas.py`
(`BBox`, `DetectedElement`, `OCRResult`, `Entity`, `Classification`,
`DocumentResult`). This makes stages independently testable and gives the API
automatic validation and OpenAPI docs.

## Deployment

The `Dockerfile` ships the CPU core (Tesseract + OpenCV + FastAPI) for instant
startup. Build with `--build-arg INSTALL_DL=1` (on a GPU base image) to bake in
the deep-learning stack. Feature flags (`DOCAI_ENABLE_*`) let a single image be
deployed OCR-only or full-stack without code changes.
