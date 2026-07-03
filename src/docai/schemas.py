"""Typed data contracts shared across every stage of the pipeline.

Using explicit schemas (instead of loose dicts) makes the boundaries between
stages self-documenting and lets FastAPI validate/serialise responses for free.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixel coordinates (x1,y1,x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BBox") -> float:
        """Intersection-over-Union with another box."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Vision
# --------------------------------------------------------------------------- #
class ElementType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    LOGO = "logo"
    SIGNATURE = "signature"
    STAMP = "stamp"


class DetectedElement(BaseModel):
    """A visual element localised on the page by the detector."""

    label: ElementType
    bbox: BBox
    score: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
class OCRWord(BaseModel):
    text: str
    bbox: BBox
    confidence: float


class OCRResult(BaseModel):
    text: str
    words: list[OCRWord] = Field(default_factory=list)
    mean_confidence: float = 0.0


# --------------------------------------------------------------------------- #
# NLP
# --------------------------------------------------------------------------- #
class Entity(BaseModel):
    text: str
    label: str          # e.g. PER, ORG, LOC, DATE, MONEY
    start: int          # char offset in the document text
    end: int
    score: float = Field(ge=0.0, le=1.0)


class Classification(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# End-to-end document result
# --------------------------------------------------------------------------- #
class DocumentResult(BaseModel):
    """The full structured output produced by the pipeline for one document."""

    doc_id: str
    text: str = ""
    ocr: Optional[OCRResult] = None
    elements: list[DetectedElement] = Field(default_factory=list)
    classification: Optional[Classification] = None
    entities: list[Entity] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    meta: dict[str, str] = Field(default_factory=dict)
