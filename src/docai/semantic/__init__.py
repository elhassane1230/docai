__all__ = ["SemanticIndex"]


def __getattr__(name):
    if name == "SemanticIndex":
        from .index import SemanticIndex
        return SemanticIndex
    raise AttributeError(name)
