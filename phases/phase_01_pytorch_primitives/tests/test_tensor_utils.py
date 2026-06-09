import torch
import pytest

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

def test_permute_then_view_fails_without_contiguous():
    x = torch.randn(2, 3, 4)
    y = x.permute(1, 0, 2)

    assert y.shape == (3, 2, 4)
    assert not y.is_contiguous()
    with pytest.raises(RuntimeError):
        y.view(3, 8)

def test_contiguous_then_view_succeeds_after_permute():
    x = torch.randn(2, 3, 4)
    y = x.permute(1, 0, 2)
    z = y.contiguous().view(3, 8)
        
    assert z.shape == (3, 8)

def test_reshape_handles_non_contiguous_tensor():
    x = torch.randn(2, 3, 4)
    y = x.permute(1, 0, 2)
    z = y.reshape(3, 8)

    assert z.shape == (3, 8)
