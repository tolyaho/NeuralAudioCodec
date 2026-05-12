"""Multi-scale waveform discriminator."""

import torch
import torch.nn.functional as F
from torch import nn


class WaveformDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        c = base_channels

        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=c,
                    kernel_size=15,
                    stride=1,
                    padding=7,
                ),
                nn.Conv1d(
                    in_channels=c,
                    out_channels=4 * c,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=4,
                ),
                nn.Conv1d(
                    in_channels=4 * c,
                    out_channels=16 * c,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=16,
                ),
                nn.Conv1d(
                    in_channels=16 * c,
                    out_channels=64 * c,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=64,
                ),
                nn.Conv1d(
                    in_channels=64 * c,
                    out_channels=64 * c,
                    kernel_size=41,
                    stride=4,
                    padding=20,
                    groups=256,
                ),
                nn.Conv1d(
                    in_channels=64 * c,
                    out_channels=64 * c,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
            ]
        )

        self.final = nn.Conv1d(
            in_channels=64 * c,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> dict:
        features = []

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            features.append(x)

        return {
            "logits": self.final(x),
            "features": features,
        }


class MultiScaleWaveformDiscriminator(nn.Module):
    def __init__(
        self,
        num_scales: int = 3,
        base_channels: int = 16,
        downsample_kernel_size: int = 4,
        downsample_stride: int = 2,
    ) -> None:
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [
                WaveformDiscriminator(
                    in_channels=1,
                    base_channels=base_channels,
                )
                for _ in range(num_scales)
            ]
        )
        self.downsample_kernel_size = downsample_kernel_size
        self.downsample_stride = downsample_stride

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(
            x,
            kernel_size=self.downsample_kernel_size,
            stride=self.downsample_stride,
            padding=self.downsample_kernel_size // 2,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> dict:
        all_logits = []
        all_features = []

        for i, discriminator in enumerate(self.discriminators):
            out = discriminator(x)
            all_logits.append(out["logits"])
            all_features.append(out["features"])

            if i != self.num_scales - 1:
                x = self.downsample(x)

        return {
            "logits": all_logits,
            "features": all_features,
        }
