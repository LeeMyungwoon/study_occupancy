import torch.nn as nn

class TinyImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 16, 
                kernel_size = 3,
                stride = 2,
                padding = 1
                ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 16, 
                out_channels = 32, 
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3,
                stride = 2,
                padding = 1
            ),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.net(x)
        return x
