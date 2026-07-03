"""Fine-tune a YOLO detector on an annotated document-layout dataset.

Expects an Ultralytics-style ``data.yaml``::

    path: data/layout
    train: images/train
    val: images/val
    names:
      0: text
      1: title
      2: table
      3: figure
      4: logo
      5: signature
      6: stamp

Run::

    python -m docai.vision.train_yolo --data data/layout/data.yaml \
        --weights yolov5s.pt --epochs 100 --imgsz 640

Transfer learning from COCO weights converges fast because low-level features
(edges, corners, texture) transfer well to document elements; only the head
needs to specialise. Mosaic + copy-paste augmentation help the rare classes
(signatures, stamps) that are under-represented in most archives.
"""
from __future__ import annotations

import argparse


def train(data: str, weights: str = "yolov5s.pt", epochs: int = 100,
          imgsz: int = 640, batch: int = 16, device: str = "0",
          project: str = "runs/layout", name: str = "exp") -> str:
    from ultralytics import YOLO  # lazy heavy import

    model = YOLO(weights)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,           # early stopping
        cos_lr=True,           # cosine LR schedule
        mosaic=1.0,
        copy_paste=0.3,        # helps rare classes (signatures / stamps)
        close_mosaic=10,
    )
    metrics = model.val()
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    return str(model.trainer.best)  # path to best.pt


def _cli():
    p = argparse.ArgumentParser(description="Fine-tune YOLO for doc layout")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default="yolov5s.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    best = train(args.data, args.weights, args.epochs, args.imgsz,
                 args.batch, args.device)
    print("Best checkpoint:", best)
