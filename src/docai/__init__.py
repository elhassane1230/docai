"""DocAI — Intelligent document processing pipeline (Computer Vision + NLP).

An end-to-end system that turns unstructured scanned documents into
structured, queryable data by combining:
  * OpenCV image preprocessing
  * Tesseract OCR
  * YOLO layout/element detection
  * BERT / DistilBERT classification & NER
  * FAISS semantic indexing
  * A FastAPI real-time inference service
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
