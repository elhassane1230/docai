"""Fine-tune BERT for Named-Entity Recognition (token classification).

Expects a CoNLL-style / HF ``token-classification`` dataset with ``tokens`` and
``ner_tags`` columns (BIO scheme). Uses ``seqeval`` for entity-level P/R/F1,
which is the metric the report tracks for NER.

Run::

    python -m docai.nlp.train_ner \
        --data data/ner.jsonl --model bert-base-cased \
        --epochs 3 --out models/ner
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    # Each row: {"tokens": [...], "ner_tags": ["B-ORG", "O", ...]}
    return rows


def train(data: str, model_name: str = "bert-base-cased", epochs: int = 3,
          out: str = "models/ner", max_length: int = 256):
    import numpy as np
    from datasets import Dataset
    from seqeval.metrics import (
        f1_score, precision_score, recall_score,
    )
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments,
    )

    rows = load(data)
    label_list = sorted({t for r in rows for t in r["ner_tags"]})
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    ds = Dataset.from_list([
        {"tokens": r["tokens"],
         "labels": [label2id[t] for t in r["ner_tags"]]}
        for r in rows
    ]).train_test_split(test_size=0.2, seed=42)

    tok = AutoTokenizer.from_pretrained(model_name)

    def align(batch):
        enc = tok(batch["tokens"], truncation=True, is_split_into_words=True,
                  max_length=max_length)
        labels = []
        for i, lab in enumerate(batch["labels"]):
            word_ids = enc.word_ids(batch_index=i)
            prev, aligned = None, []
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)          # special tokens ignored
                elif wid != prev:
                    aligned.append(lab[wid])
                else:
                    aligned.append(-100)          # only label first sub-token
                prev = wid
            labels.append(aligned)
        enc["labels"] = labels
        return enc

    ds = ds.map(align, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list),
        id2label=id2label, label2id=label2id,
    )

    def metrics(eval_pred):
        logits, y = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_labels, true_preds = [], []
        for p_row, y_row in zip(preds, y):
            tl, tp = [], []
            for p_i, y_i in zip(p_row, y_row):
                if y_i == -100:
                    continue
                tl.append(id2label[y_i])
                tp.append(id2label[p_i])
            true_labels.append(tl)
            true_preds.append(tp)
        return {
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds),
            "f1": f1_score(true_labels, true_preds),
        }

    args = TrainingArguments(
        output_dir=out, num_train_epochs=epochs, learning_rate=2e-5,
        per_device_train_batch_size=16, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", logging_steps=20,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], compute_metrics=metrics,
        data_collator=DataCollatorForTokenClassification(tok),
    )
    trainer.train()
    print("Eval:", trainer.evaluate())
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"Saved NER model -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="bert-base-cased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", default="models/ner")
    a = p.parse_args()
    train(a.data, a.model, a.epochs, a.out)
