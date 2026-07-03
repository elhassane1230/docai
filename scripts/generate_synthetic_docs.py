"""Generate synthetic 'scanned' documents with known ground-truth text.

We render clean paragraphs to an image (so we know the exact reference text),
then apply realistic scanner degradations: skew, Gaussian + salt-and-pepper
noise, blur, low contrast, and JPEG-style artefacts. This gives us a labelled
corpus to measure OCR CER/WER and — crucially — to quantify how much the
OpenCV preprocessing recovers.

Outputs, per document i:
    data/synthetic/clean_{i}.png       # rendered, minimal degradation
    data/synthetic/noisy_{i}.png       # heavily degraded 'scan'
    data/synthetic/gt_{i}.txt          # ground-truth text
    data/synthetic/manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# A small pool of realistic business-document sentences (labelled by doc type).
CORPUS = {
    "invoice": [
        "Invoice number 2024-00817 issued on March 14, 2024.",
        "Please remit payment of 4,250.00 EUR within thirty days.",
        "Billed to Acme Logistics, 42 Harbor Road, Rotterdam.",
        "VAT identification number NL8123456789B01 applies.",
        "Late payments are subject to a 2 percent monthly interest charge.",
    ],
    "contract": [
        "This agreement is entered into by and between the parties below.",
        "The term of this contract shall commence on the effective date.",
        "Either party may terminate with sixty days written notice.",
        "Confidential information shall not be disclosed to third parties.",
        "This contract is governed by the laws of the Netherlands.",
    ],
    "report": [
        "Quarterly revenue increased by twelve percent year over year.",
        "The engineering team shipped four major releases this quarter.",
        "Customer churn declined following the onboarding redesign.",
        "Operating margin improved to eighteen percent in the period.",
        "We recommend expanding capacity ahead of the holiday season.",
    ],
    "letter": [
        "Dear Mr. Anderson, thank you for your recent correspondence.",
        "We are writing to confirm receipt of your application.",
        "Our office will review the documents within five working days.",
        "Should you have questions, please contact our support desk.",
        "Yours sincerely, the Customer Relations Department.",
    ],
}


def _font(size: int = 28) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_document(lines: list[str], width: int = 1000) -> tuple[np.ndarray, str]:
    """Render text lines onto a white page. Returns (BGR image, ground truth)."""
    font = _font(28)
    margin, line_h = 60, 46
    height = margin * 2 + line_h * len(lines)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = margin
    for ln in lines:
        draw.text((margin, y), ln, fill="black", font=font)
        y += line_h
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr, "\n".join(lines)


def degrade(img: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    """Apply realistic scanner degradations. severity in [0,1].

    The centrepiece is an *uneven illumination gradient* (a soft shadow across
    the page). This is the failure mode that breaks both raw OCR and a global
    threshold, and the one that CLAHE + adaptive thresholding are designed to
    fix — which is exactly what makes the preprocessing ablation meaningful.
    Combined with defocus blur, fade, Gaussian noise and skew, it mimics a
    poorly-lit phone photo or an old flatbed scan.
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # 1) Uneven illumination: darker on one side, dimmer towards the bottom.
    yy, xx = np.mgrid[0:h, 0:w]
    side = rng.choice([1.0, -1.0])
    grad = (0.55 + 0.45 * (xx / w)) if side > 0 else (1.0 - 0.45 * (xx / w))
    grad = grad * (0.72 + 0.28 * (yy / h))
    out = out * (1.0 - severity + severity * grad[..., None])

    # 2) Global fade / low contrast.
    out = 90 * severity + out * (1.0 - 0.45 * severity)

    # 3) Defocus blur.
    out = cv2.GaussianBlur(out, (5, 5), 0)

    # 4) Gaussian sensor noise.
    out = out + np.random.normal(0, 14 * severity, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 5) Small page skew.
    angle = rng.uniform(-3, 3) * severity
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, m, (w, h), borderValue=(255, 255, 255))

    # 6) A little salt-and-pepper speckle.
    n_sp = int(0.0015 * severity * h * w)
    flat = out.reshape(-1, out.shape[2])
    for _ in range(n_sp):
        idx = rng.randrange(flat.shape[0])
        flat[idx] = 0 if rng.random() < 0.5 else 255

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12, help="documents to generate")
    p.add_argument("--severity", type=float, default=0.8)
    p.add_argument("--out", default=str(Path(__file__).parents[1] / "data" / "synthetic"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    doc_types = list(CORPUS.keys())
    for i in range(args.n):
        dtype = doc_types[i % len(doc_types)]
        n_lines = rng.randint(3, 5)
        lines = rng.sample(CORPUS[dtype], k=min(n_lines, len(CORPUS[dtype])))
        clean, gt = render_document(lines)
        noisy = degrade(clean, args.severity, rng)

        cv2.imwrite(str(out_dir / f"clean_{i}.png"), clean)
        cv2.imwrite(str(out_dir / f"noisy_{i}.png"), noisy)
        (out_dir / f"gt_{i}.txt").write_text(gt)
        manifest.append({
            "id": i, "type": dtype,
            "clean": f"clean_{i}.png", "noisy": f"noisy_{i}.png",
            "gt": f"gt_{i}.txt",
        })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {args.n} documents in {out_dir}")


if __name__ == "__main__":
    main()
