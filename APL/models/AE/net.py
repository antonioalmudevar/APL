from torch import nn
from torch import Tensor
from typing import List, Tuple
from ..modules.cbam import CBAMBlock


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
            self.channel_matching =  nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        # Convolutional layers
        padding = (kernel_size-1)//2
        channels = [in_channels] + [out_channels for _ in range(nconv)]
        self.conv = []
        for i in range(nconv):
            self.conv.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size, padding=padding),
                nn.BatchNorm2d(channels[i+1])
            ])
            if i!=(nconv-1): self.conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*self.conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        residual = self.channel_matching(x)
        out += residual
        out = self.relu(out)
        return out


class MainBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        nconv_residual: int = 1,
        pooling: List[int] = [2,2],
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size, nconv_residual)
        self.cbam = CBAMBlock(out_channels) if cbam else nn.Identity()
        self.avgpooling = nn.AvgPool2d(pooling)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.resconv(x)
        out = self.cbam(out)
        out = self.dropout(out)
        out = self.avgpooling(out)
        return out


class MainBlockTranspose(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        nconv_residual: int = 1,
        scale: Tuple[int] = (2,2),
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size, nconv_residual)
        self.cbam = CBAMBlock(out_channels) if cbam else nn.Identity()
        self.upsampling = nn.Upsample(scale_factor=scale, mode='nearest')
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsampling(x)
        out = self.resconv(out)
        out = self.cbam(out)
        out = self.dropout(out)
        return out


class Net(nn.Module):

    def __init__(
        self,
        channels: List[int] = [16,32,64],
        kernel_size: List[int] = [3,3,3],
        nconv_residual: int = 1,
        time_pooling: List[int] = [2,2,2],
        distance_pooling: List[int] = [2,2,2],
        dropout: int = 0,
        cbam: bool = False,
    ) -> None:
        super(Net,self).__init__()
        total_channels = [1] + channels
        n_layers = len(channels)
        # Encoders
        encoder = []
        for i in range(n_layers):
            encoder.append(
                MainBlock( 
                    total_channels[i],
                    total_channels[i+1],
                    kernel_size[i],
                    nconv_residual,
                    [time_pooling[i], distance_pooling[i]],
                    dropout,
                    cbam,
                )
            )
        self.encoder = nn.Sequential(*encoder)
        # Decoder
        decoder = []
        for i in range(n_layers-1,0,-1):
            decoder.append(
                MainBlockTranspose(
                    total_channels[i+1],
                    total_channels[i],
                    kernel_size[i],
                    nconv_residual,
                    (time_pooling[i], distance_pooling[i]),
                    dropout,
                    cbam,
                )
            )
        decoder.append(nn.Upsample(scale_factor=(time_pooling[0], distance_pooling[0]), mode='nearest'))
        decoder.append(nn.Conv2d(channels[0], 1, kernel_size[0], padding=(kernel_size[0]-1)//2))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        return y