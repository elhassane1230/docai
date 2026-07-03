"""Fine-tune BERT / DistilBERT for document classification (Hugging Face).

Reads a CSV/JSONL with ``text`` and ``label`` columns, fine-tunes a sequence
classifier, and reports Precision / Recall / F1 on a held-out split.

Run::

    python -m docai.nlp.train_classifier \
        --data data/reports.jsonl \
        --model distilbert-base-uncased \
        --epochs 3 --out models/report-classifier

DistilBERT is the recommended default: it keeps ~97% of BERT-base accuracy
while roughly halving parameters/latency — see the ablation study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_dataset(path: str):
    import pandas as pd

    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(p)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("dataset must have 'text' and 'label' columns")
    return df


def train(data: str, model_name: str = "distilbert-base-uncased",
          epochs: int = 3, out: str = "models/classifier",
          max_length: int = 256, batch: int = 16, lr: float = 2e-5):
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments,
    )

    df = load_dataset(data)
    labels = sorted(df["label"].unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    ds = Dataset.from_pandas(df[["text", "labels"]]).train_test_split(test_size=0.2, seed=42)
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length",
                   max_length=max_length)

    ds = ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    def metrics(eval_pred):
        logits, y = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f, _ = precision_recall_fscore_support(
            y, preds, average="macro", zero_division=0)
        return {"precision": p, "recall": r, "f1": f}

    args = TrainingArguments(
        output_dir=out, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", logging_steps=20,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"], eval_dataset=ds["test"],
                      compute_metrics=metrics)
    trainer.train()
    print("Eval:", trainer.evaluate())
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"Saved classifier -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", default="models/classifier")
    p.add_argument("--max-length", type=int, default=256)
    a = p.parse_args()
    train(a.data, a.model, a.epochs, a.out, a.max_length)
