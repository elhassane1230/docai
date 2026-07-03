"""Export a Transformer classifier to ONNX and apply dynamic INT8 quantization.

This is the concrete mechanism behind the "real-time inference" requirement and
the latency ablation: DistilBERT already halves BERT's depth; exporting to
ONNX Runtime and quantizing weights to INT8 compounds the win, typically
cutting CPU latency further with negligible accuracy loss.

Pipeline: HF model -> torch.onnx export -> onnxruntime dynamic quantization.
"""
from __future__ import annotations

from pathlib import Path


def export_to_onnx(model_name: str, out_dir: str, max_length: int = 256) -> str:
    """Export a sequence-classification model to ONNX. Returns the .onnx path."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    tok.save_pretrained(out)

    dummy = tok("dummy text", return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_length)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    return onnx_path.as_posix()


def quantize_dynamic_int8(onnx_path: str) -> str:
    """Apply ONNX Runtime dynamic INT8 quantization. Returns the quantized path."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quant_path = onnx_path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    return quant_path


class OnnxClassifier:
    """Minimal ONNX Runtime inference wrapper for a quantized classifier."""

    def __init__(self, onnx_path: str, tokenizer_dir: str, labels: list[str],
                 max_length: int = 256):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.labels = labels
        self.max_length = max_length

    def predict(self, text: str) -> dict:
        import numpy as np

        enc = self.tok(text, return_tensors="np", truncation=True,
                       padding="max_length", max_length=self.max_length)
        logits = self.sess.run(
            ["logits"],
            {"input_ids": enc["input_ids"].astype("int64"),
             "attention_mask": enc["attention_mask"].astype("int64")},
        )[0][0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        idx = int(probs.argmax())
        return {"label": self.labels[idx], "score": float(probs[idx])}
