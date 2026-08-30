import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import torch
from model import get_model


def test_model_shape():
    out = get_model("cnn", 10)(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_model_rejects_unknown_architecture():
    try:
        get_model("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown architecture should fail")
