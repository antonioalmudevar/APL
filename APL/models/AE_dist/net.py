from torch import nn
from torch import Tensor
from typing import List, Tuple
from ..modules.adain import AdaIN
from ..modules.cbam import CBAMBlock


class MySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


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
        channels = [in_channels] + [out_channels for _ in range(nconv)]
        self.conv_dist, self.conv = [], []
        for i in range(nconv):
            self.conv_dist.append(
                nn.Conv1d(channels[i], channels[i+1], kernel_size=1, padding=padding)
            )
            self.conv.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size, padding=padding)
            )
        self.conv = nn.ModuleList(self.conv)
        self.conv_dist = nn.ModuleList(self.conv_dist)
        self.adain = AdaIN()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        nconv = len(self.conv)
        out, d_out = x, d
        for i in range(nconv):
            d_out = self.conv_dist[i](d_out)
            out = self.conv[i](out)
            out = self.adain(out, d_out)
            if i!=(nconv-1): out = self.relu(out)
        # Residual path
        residual = self.channel_matching(x)
        out += residual
        out = self.relu(out)
        return out, d_out


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

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        out, d_out = self.resconv(x, d)
        out = self.cbam(out)
        out = self.dropout(out)
        out = self.avgpooling(out)
        return out, d_out


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

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        out = self.upsampling(x)
        out, d_out = self.resconv(out, d)
        out = self.cbam(out)
        out = self.dropout(out)
        return out, d_out


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
        # Encoder
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
        self.encoder = MySequential(*encoder)
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
        self.decoder = MySequential(*decoder)
        self.final_block = nn.Sequential(
            nn.Upsample(scale_factor=(time_pooling[0], distance_pooling[0]), mode='nearest'),
            nn.Conv2d(channels[0], 1, kernel_size[0], padding=(kernel_size[0]-1)//2),
        )

    def forward(self, x: Tensor, d: Tensor) -> Tensor:
        z, d_out = self.encoder(*(x, d))
        out, _ = self.decoder(*(z, d_out))
        out = self.final_block(out)
        return out