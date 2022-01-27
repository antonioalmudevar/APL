from torch import nn
from torch import Tensor
from ..modules.adain import AdaIN

class ResidualBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        nconv: int = 1,
    ) -> None:
        super().__init__()
        # Residual Path
        self.channel_matching = nn.Identity()
        if in_channels != out_channels:
            self.channel_matching =  nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Convolutional layers
        padding = (kernel_size-1)//2
        channels = [in_channels] + [out_channels for _ in range(nconv-1)]
        self.conv_dist, self.conv = [], []
        for i in range(nconv):
            self.conv_dist.append(
                nn.Conv1d(channels[i], out_channels[i+1], kernel_size, padding=padding)
            )
            self.conv.append(
                nn.Conv2d(channels[i], out_channels[i+1], kernel_size, padding=padding)
            )
        self.adain = AdaIN()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        nconv = len(self.conv)
        out, d_out = x, d
        for i in range(len(nconv)):
            d_out = self.conv_dist[i](d_out)
            out = self.conv[i](out)
            out = self.adain(out, d_out)
            if i!=(nconv-1): out = self.relu(out)
        # Residual path
        residual = self.channel_matching(x)
        out += residual
        out = self.relu(out)
        return out, d_out