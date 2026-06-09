import torch
import pytest
from phases.phase_01_pytorch_primitives.src.voxel_blocks import Conv3DBlock

def test_conv3d_block():
    model = Conv3DBlock(16, 32)
    x = torch.randn(2, 16, 8, 4, 2, requires_grad = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    output = model(x)
    optimizer.zero_grad()
    loss = output.mean()
    loss.backward()
    optimizer.step()

    assert output.shape == (2, 32, 8, 4, 2)
    assert x.grad is not None
