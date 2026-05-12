"""SoundStream encoder."""

import torch
from torch import nn

from src.model.blocks import EncoderBlock, same_padding


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 512,
        strides: tuple[int, ...] = (2, 4, 5, 5),
    ) -> None:
        super().__init__()

        self.strides = strides
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        layers = [
            nn.Conv1d(
                kernel_size=7,
                padding=same_padding(7),
                in_channels=in_channels,
                out_channels=base_channels,
            )
        ]

        channels = base_channels
        for stride in self.strides:
            layers.append(
                EncoderBlock(
                    in_channels=channels,
                    out_channels=channels * 2,
                    stride=stride,
                )
            )
            channels *= 2

        layers.append(nn.ELU())
        layers.append(
            nn.Conv1d(
                in_channels=channels,
                out_channels=latent_dim,
                kernel_size=3,
                padding=same_padding(3),
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
