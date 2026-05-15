import torch
from torch import nn

from src.model.blocks import DecoderBlock, same_padding


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 512,
        strides: tuple[int, ...] = (2, 4, 5, 5),
    ) -> None:
        super().__init__()

        self.strides = strides
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        channels = base_channels * (2 ** len(strides))
        layers = [
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=channels,
                kernel_size=7,
                padding=same_padding(7),
            )
        ]

        for stride in reversed(strides):
            layers.append(
                DecoderBlock(
                    in_channels=channels,
                    out_channels=channels // 2,
                    stride=stride,
                )
            )
            channels //= 2

        layers.append(nn.ELU())
        layers.append(
            nn.Conv1d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=same_padding(7),
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
