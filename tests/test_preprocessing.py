import numpy as np

from docai.preprocessing.opencv_ops import (
    binarize, deskew, preprocess, to_grayscale,
)
from docai.config import PreprocessSettings


def _sample_page():
    img = np.full((200, 400, 3), 255, np.uint8)
    img[80:120, 40:360] = 0  # a black bar standing in for a text line
    return img


def test_grayscale_reduces_channels():
    g = to_grayscale(_sample_page())
    assert g.ndim == 2


def test_binarize_is_binary():
    g = to_grayscale(_sample_page())
    b = binarize(g)
    assert set(np.unique(b)).issubset({0, 255})


def test_deskew_returns_same_shape():
    g = to_grayscale(_sample_page())
    assert deskew(g).shape == g.shape


def test_preprocess_pipeline_runs():
    out = preprocess(_sample_page(), PreprocessSettings())
    assert out.ndim == 2
    assert out.dtype == np.uint8
