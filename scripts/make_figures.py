"""Regenerate the results figure (before/after preprocessing + ablation chart).

Run after the ablation:
    python scripts/make_figures.py
Outputs docs/figures/ocr_ablation.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from docai.config import PreprocessSettings  # noqa: E402
from docai.preprocessing.opencv_ops import preprocess  # noqa: E402

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "synthetic"
FIG = ROOT / "docs" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    abl = json.loads((DATA / "ocr_ablation_results.json").read_text())

    plt.figure(figsize=(13, 5.5))
    noisy = cv2.cvtColor(cv2.imread(str(DATA / "noisy_0.png")), cv2.COLOR_BGR2RGB)
    clean = preprocess(cv2.imread(str(DATA / "noisy_0.png")), PreprocessSettings())

    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(noisy)
    ax1.set_title("Input: degraded scan (uneven light + noise + skew)", fontsize=10)
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 3)
    ax2.imshow(clean, cmap="gray")
    ax2.set_title("After OpenCV preprocessing", fontsize=10)
    ax2.axis("off")

    ax3 = plt.subplot(1, 2, 2)
    steps = abl["per_step"]
    names = list(steps.keys())
    cers = [steps[n]["cer"] for n in names]
    colors = ["#c44"] + ["#48c"] * (len(names) - 2) + ["#2a2"]
    ax3.barh(range(len(names)), cers, color=colors)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.invert_yaxis()
    ax3.set_xlabel("Character Error Rate (lower is better)")
    ax3.set_title(
        f"OCR preprocessing ablation (n={abl['n_documents']} docs)\n"
        f"Raw {abl['raw_grayscale']['cer']:.3f} → "
        f"Full {abl['full_preprocessing']['cer']:.3f}  "
        f"({abl['cer_relative_improvement_pct']:.1f}% CER reduction)",
        fontsize=10,
    )
    for i, c in enumerate(cers):
        ax3.text(c + 0.005, i, f"{c:.3f}", va="center", fontsize=9)
    ax3.set_xlim(0, max(cers) * 1.18)

    plt.tight_layout()
    out = FIG / "ocr_ablation.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
