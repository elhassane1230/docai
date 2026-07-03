"""OpenCV preprocessing that runs *before* Tesseract.

Scanned documents are noisy: skew, coffee-stain speckle, uneven lighting,
bleed-through, JPEG artefacts. Cleaning the raster first is one of the highest
ROI things you can do for OCR — the ablation study in ``scripts/run_ocr_ablation.py``
quantifies exactly how much CER/WER this buys you.

Each function is a small, composable, unit-testable step. ``preprocess`` chains
them according to :class:`docai.config.PreprocessSettings`.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessSettings


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img


def denoise(img: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise. Non-local means keeps glyph strokes crisp
    while removing speckle — better than a naive Gaussian blur for text."""
    # h=7 is a deliberately light setting: strong enough to kill sensor speckle
    # but gentle enough not to erode thin glyph strokes (which hurts OCR).
    return cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7,
                                    searchWindowSize=21)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalisation. Fixes uneven
    lighting / faded scans without blowing out already-dark regions."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def deskew(img: np.ndarray) -> np.ndarray:
    """Estimate page skew from the dominant text angle and rotate it flat.

    Uses the minimum-area rectangle of foreground pixels — robust for the
    small skews (±15°) typical of flatbed/ADF scanners.
    """
    # Foreground = dark ink on light paper -> invert so ink is white.
    inv = cv2.bitwise_not(img)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:  # not enough ink to trust an estimate
        return img
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in [-90, 0); normalise to a small rotation.
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.1:
        return img
    (h, w) = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive (Sauvola-like) thresholding. Beats a global Otsu on documents
    with gradient lighting, which is the common failure case for scans."""
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=41, C=12,
    )


def remove_borders(img: np.ndarray) -> np.ndarray:
    """Crop the black scan border / page shadow by keeping the largest
    contour's bounding rectangle. No-op if nothing sensible is found."""
    inv = cv2.bitwise_not(img)
    thr = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    # Guard against degenerate crops.
    if w < img.shape[1] * 0.4 or h < img.shape[0] * 0.4:
        return img
    return img[y:y + h, x:x + w]


def upscale_to_dpi(img: np.ndarray, current_dpi: int, target_dpi: int) -> np.ndarray:
    """Tesseract likes ~300 DPI. Upscale small scans with cubic interpolation."""
    if current_dpi >= target_dpi:
        return img
    factor = target_dpi / current_dpi
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def preprocess(img: np.ndarray, cfg: PreprocessSettings | None = None) -> np.ndarray:
    """Run the configured preprocessing chain and return a clean image.

    Order matters a great deal (the ablation study proves it):

        grayscale -> denoise -> CLAHE -> deskew -> border crop -> binarise

    Denoise runs *before* CLAHE: contrast enhancement amplifies whatever is
    present, so removing speckle first prevents CLAHE from turning noise into
    hard artefacts. Adaptive binarisation runs last, once the signal is clean
    and geometrically corrected, so it can beat a global threshold on the
    uneven-illumination case that is common in real scans.
    """
    cfg = cfg or PreprocessSettings()
    out = to_grayscale(img)
    if cfg.denoise:
        out = denoise(out)
    if cfg.clahe:
        out = apply_clahe(out)
    if cfg.deskew:
        out = deskew(out)
    if cfg.remove_borders:
        out = remove_borders(out)
    if cfg.binarize:
        out = binarize(out)
    return out
