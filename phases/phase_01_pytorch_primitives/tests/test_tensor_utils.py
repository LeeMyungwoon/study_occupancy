import torch

from phases.phase_01_pytorch_primitives.src.tensor_utils import (
    flatten_hw,
    flatten_xyz,
    unflatten_hw,
    unflatten_xyz,
)


def test_flatten_unflatten_hw_roundtrip():
    x = torch.randn(2, 3, 4, 5)

    tokens = flatten_hw(x)
    restored = unflatten_hw(tokens, h=4, w=5)

    assert tokens.shape == (2, 20, 3)
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)


def test_flatten_unflatten_xyz_roundtrip():
    x = torch.randn(2, 3, 4, 5, 6)

    tokens = flatten_xyz(x)
    restored = unflatten_xyz(tokens, x_size=4, y_size=5, z_size=6)

    assert tokens.shape == (2, 120, 3)
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)
