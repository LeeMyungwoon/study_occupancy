import torch

def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)

def unflatten_hw(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, hw, c = tokens.shape
    assert hw == h * w
    # permute로 인해 Non-Contiguous상태가 되므로 contiguous()로 Contiguous 로 복구
    return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def flatten_xyz(x: torch.Tensor) -> torch.Tensor:
    b, c, x_size, y_size, z_size = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b, x_size * y_size * z_size, c)

def unflatten_xyz(tokens: torch.Tensor, x_size: int, y_size: int, z_size: int) -> torch.Tensor:
    b, xyz, c = tokens.shape
    assert xyz == x_size * y_size * z_size
    return tokens.reshape(b, x_size, y_size, z_size, c).permute(0, 4, 1, 2, 3).contiguous()
