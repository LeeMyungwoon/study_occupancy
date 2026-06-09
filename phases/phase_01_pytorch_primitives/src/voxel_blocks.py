import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layers(x)

    