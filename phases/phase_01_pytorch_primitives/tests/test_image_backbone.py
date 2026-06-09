import torch
from phases.phase_01_pytorch_primitives.src.image_backbone import TinyImageBackbone

def test_image_backbone():
    model = TinyImageBackbone()
    x = torch.randn(2, 3, 128, 256)

    y_hat = model(x)

    assert y_hat.shape == (2, 32, 16, 32)