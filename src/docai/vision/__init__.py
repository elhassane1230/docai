"""Vision components (lazy — require ultralytics)."""
__all__ = ["YOLODetector"]


def __getattr__(name):
    if name == "YOLODetector":
        from .yolo_detector import YOLODetector
        return YOLODetector
    raise AttributeError(name)
