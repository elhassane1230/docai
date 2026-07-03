from .benchmark import benchmark, compare, LatencyReport  # noqa: F401

__all__ = ["benchmark", "compare", "LatencyReport",
           "export_to_onnx", "quantize_dynamic_int8", "OnnxClassifier"]


def __getattr__(name):
    if name in {"export_to_onnx", "quantize_dynamic_int8", "OnnxClassifier"}:
        from . import onnx_export
        return getattr(onnx_export, name)
    raise AttributeError(name)
