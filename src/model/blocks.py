import math

import torch
from torch import nn


def same_padding(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


def downsample_padding(stride: int) -> int:
    return math.ceil(stride / 2)


def upsample_output_padding(stride: int) -> int:
    return stride % 2


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                padding=same_padding(7, dilation),
                dilation=dilation,
            ),
            nn.ELU(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
            nn.ELU(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=downsample_padding(stride),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=downsample_padding(stride),
                output_padding=upsample_output_padding(stride),
            ),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
