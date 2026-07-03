# Proposed improvements & roadmap

These are concrete extensions beyond the delivered scope, ordered roughly by
impact-to-effort. Several are stubbed or noted in the code so they slot in
cleanly.

## 1. LayoutLMv3 — unified multimodal understanding (highest impact)
Today classification/NER run on OCR text alone, discarding *where* the text
sits. **LayoutLMv3** jointly encodes text + 2-D layout (bounding boxes) + the
image patch, and is state-of-the-art for form/receipt/report understanding
(FUNSD, CORD, DocVIE). Because the OCR stage already emits per-word boxes
(`OCRResult.words`), the features LayoutLMv3 needs are *already produced* — this
is a natural upgrade of the `nlp/` stage that typically lifts NER F1
substantially on structured documents. Add as `nlp/layoutlm.py` behind the same
`predict()`/`extract()` interface.

## 2. Donut — OCR-free document understanding
For highly-structured docs (invoices, receipts), an OCR-free encoder–decoder
(**Donut / Pix2Struct**) reads the image and emits structured JSON directly,
sidestepping OCR error propagation entirely. Worth benchmarking head-to-head
against the OCR→LayoutLM path per document type; route each type to whichever
wins.

## 3. Table Structure Recognition
YOLO *detects* tables; it doesn't parse them. Add **Table Transformer (TATR)**
or a rule-assisted cell-detection pass to recover rows/columns/cells → emit
tables as DataFrames/CSV. This is usually the highest-value structured output
for operations teams.

## 4. Human-in-the-loop + active learning
- Route low-confidence outputs (OCR mean-confidence, softmax margin, detection
  score below thresholds) to a review queue instead of auto-accepting.
- Feed corrections back as labels; retrain periodically. An **active-learning**
  sampler (uncertainty / margin sampling) cuts annotation cost dramatically for
  the rare classes (signatures, stamps) — the exact classes copy-paste
  augmentation is compensating for today.

## 5. Experiment tracking (MLflow / Weights & Biases)
The ablation currently writes JSON. Log runs, params, metrics, and artefacts to
**MLflow** so preprocessing/model ablations are versioned, comparable, and
reproducible across the team. One decorator around `run_ocr_ablation.py` and
the trainers.

## 6. Inference optimisation for throughput
- **Dynamic batching** in the API (collect requests over a few-ms window) to
  raise GPU utilisation.
- **Knowledge distillation** of the fine-tuned BERT into a task-specific
  student, beyond off-the-shelf DistilBERT.
- The ONNX/INT8 path (`optimize/`) is built; add TensorRT for GPU serving.

## 7. Scale the semantic layer
Flat FAISS is fine to ~1M vectors. Beyond that, move to an ANN index
(HNSW/IVF-PQ) or a vector DB (**Qdrant / pgvector / Weaviate**) with metadata
filtering, and layer **RAG** on top so operators can *ask questions* of the
archive, not just retrieve documents.

## 8. Robust input handling
- Native **born-digital PDF** path (extract embedded text; only OCR the scanned
  pages) — much faster and error-free where text already exists.
- **Language detection** → pick Tesseract language packs automatically;
  multilingual OCR for mixed-language archives.
- Multi-page documents, page-level then document-level aggregation.

## 9. Observability & governance
- **OpenTelemetry** tracing + Prometheus metrics per stage (latency, confidence
  distributions, class mix) to catch **data/model drift**.
- **PII detection & redaction** before indexing/exports (GDPR); audit logging.
- Model/version pinning and canary rollout for new checkpoints.

## 10. Evaluation depth
- mAP@[0.5:0.95] (COCO-style) in addition to mAP@0.5.
- Confusion matrices and per-class error analysis surfaced in a dashboard.
- A small **real** annotated benchmark set (the current corpus is synthetic —
  synthetic proves the plumbing and the preprocessing effect; a real gold set
  is needed to certify production accuracy).
