# Results & evaluation methodology

Everything here is reproducible on CPU:

```bash
make data       # 24 synthetic docs, severity 0.9, seed 42
make ablation   # → data/synthetic/ocr_ablation_results.json
make demo       # → results/results.json
```

## Evaluation axes

| Stage | Metric | Rationale |
|-------|--------|-----------|
| OCR | CER, WER | Levenshtein edit distance / reference length; micro-averaged |
| Detection | mAP@0.5 | VOC-style AP, greedy IoU matching per class |
| Classification | Precision / Recall / F1 (macro, micro, weighted) | Class-imbalance aware |
| NER | span-level P / R / F1 | Exact label + boundary match (stricter than token accuracy) |

## 1. OCR preprocessing ablation

Corpus: 24 documents, realistic degradation (uneven illumination + defocus +
fade + Gaussian & salt-pepper noise + skew). Tesseract 5 LSTM, PSM 6.

| Condition | CER | WER |
|-----------|-----|-----|
| grayscale only (raw) | 0.385 | 0.701 |
| + CLAHE | 0.251 | — |
| + denoise | 0.301 | — |
| + deskew | 0.435 | — |
| + adaptive binarize | 0.169 | — |
| **full chain** | **0.075** | **0.241** |

**Relative improvement: −80.5% CER, −65.6% WER.**

Findings:
1. **Adaptive binarization dominates.** It alone more than halves CER, because
   the core failure mode (uneven illumination) is precisely what a *local*
   threshold fixes and a global one cannot.
2. **Order and interaction matter.** CLAHE *stacked before* adaptive
   thresholding is harmful — both react to local contrast and compound noise
   into hard artefacts (CER ≈ 0.84 in that configuration). CLAHE and border
   removal are therefore off by default; denoise runs *before* any contrast op.
3. **The tuned chain composes well**: denoise → deskew → adaptive binarize
   reaches CER 0.075, lower than any single step alone.

> Methodological note: results are degradation-profile dependent. The takeaway
> is not "these exact filters always win" but "ablate and tune to *your* scans".

## 2. Classification (end-to-end)

A TF-IDF + LinearSVC baseline is trained on clean ground-truth text and
evaluated on the **OCR output of the noisy scans** — so the score includes
upstream OCR error, the honest end-to-end setting. Result: accuracy 1.00,
macro-F1 1.00 on the held-out split. The document classes have distinct
vocabulary, and TF-IDF is robust to the residual OCR noise after preprocessing.

The baseline is intentionally kept as (a) the control condition for the
BERT-vs-baseline comparison and (b) a zero-dependency fallback for the API.

## 3. Detection (mAP)

Worked example in `scripts/run_demo.py`: 3 ground-truth elements (logo,
signature, table); the detector matches logo and table with high IoU but
mis-localises the signature (IoU below 0.5). Result: **mAP@0.5 = 0.667**
(per-class AP: logo 1.0, table 1.0, signature 0.0) — demonstrating the metric
correctly penalises localisation failures even when the class is predicted.

## 4. NER (span-level)

Worked example: 3 gold entities (ORG, LOC, DATE); the prediction gets ORG and
LOC exactly but truncates the DATE span. Result: **P/R/F1 = 0.667**, confirming
the metric penalises boundary errors (a token-level metric would over-credit
the partially-correct DATE).

## 5. BERT vs DistilBERT latency (methodology)

`docai/optimize/benchmark.py` measures p50/p95/p99 latency and throughput for
any `str → prediction` callable, so BERT, DistilBERT, and the ONNX-INT8 build
are timed identically. Expected pattern (on GPU/CPU with model access):
DistilBERT ≈ −40% p50 latency vs BERT-base at <2% F1 loss; ONNX INT8 compounds
the reduction further. Run once models are downloaded:

```python
from docai.optimize import benchmark, compare
from docai.nlp import TransformerClassifier
bert  = TransformerClassifier(NLPSettings(classifier_model="bert-base-uncased"))
distil= TransformerClassifier(NLPSettings(classifier_model="distilbert-base-uncased"))
r1 = benchmark(lambda t: bert.predict(t),  samples, name="bert")
r2 = benchmark(lambda t: distil.predict(t),samples, name="distilbert")
print(compare([r1, r2], baseline="bert"))
```
