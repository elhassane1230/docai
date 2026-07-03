"""Centralised, environment-overridable configuration.

Every tunable lives here so nothing is hard-coded deep in the code. Values can
be overridden with environment variables prefixed ``DOCAI_`` (e.g.
``DOCAI_OCR__LANG=eng+fra``) or a local ``.env`` file.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class OCRSettings(BaseSettings):
    lang: str = "eng"            # Tesseract language(s), e.g. "eng+fra"
    psm: int = 3                 # page segmentation mode
    oem: int = 3                 # OCR engine mode (LSTM)
    min_confidence: float = 0.0  # drop words below this confidence (0-100)


class PreprocessSettings(BaseSettings):
    # Defaults chosen empirically from scripts/run_ocr_ablation.py.
    # CLAHE + adaptive threshold compound and amplify noise, and aggressive
    # border cropping occasionally clips text, so both are OFF by default.
    target_dpi: int = 300
    deskew: bool = True
    denoise: bool = True
    binarize: bool = True
    clahe: bool = False          # contrast-limited adaptive hist. equalisation
    remove_borders: bool = False


class VisionSettings(BaseSettings):
    weights: str = "yolov5s.pt"  # or a fine-tuned checkpoint path
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "cpu"          # "cpu", "cuda:0", ...


class NLPSettings(BaseSettings):
    classifier_model: str = "distilbert-base-uncased"
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    max_length: int = 256
    device: str = "cpu"
    # candidate document classes for the report classifier
    labels: list[str] = Field(
        default_factory=lambda: [
            "invoice", "contract", "report", "letter", "form", "email",
        ]
    )


class SemanticSettings(BaseSettings):
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = str(ROOT / "data" / "faiss.index")
    dim: int = 384


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCAI_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    root: Path = ROOT
    data_dir: Path = ROOT / "data"

    ocr: OCRSettings = Field(default_factory=OCRSettings)
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    vision: VisionSettings = Field(default_factory=VisionSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    api: APISettings = Field(default_factory=APISettings)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached singleton accessor."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
