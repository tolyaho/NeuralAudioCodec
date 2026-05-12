"""STFT-based discriminator."""

import torch
from torch import nn


class STFTResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        multiplier: int,
        stride: tuple[int, int],
    ) -> None:
        super().__init__()

        out_channels = hidden_channels * multiplier
        st, sf = stride

        self.main = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(st + 2, sf + 2),
                stride=stride,
                padding=(1, 1),
            ),
        )

        self.skip = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        skip = self.skip(x)

        h = min(y.shape[-2], skip.shape[-2])
        w = min(y.shape[-1], skip.shape[-1])

        return y[..., :h, :w] + skip[..., :h, :w]


class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        window_length: int = 1024,
        hop_length: int = 256,
    ) -> None:
        super().__init__()

        self.base_channels = base_channels
        self.window_length = window_length
        self.hop_length = hop_length

        self.register_buffer(
            "window",
            torch.hann_window(window_length),
            persistent=False,
        )

        c = self.base_channels

        self.first = nn.Conv2d(
            in_channels=2,
            out_channels=c,
            kernel_size=(7, 7),
            padding=(3, 3),
        )

        self.blocks = nn.ModuleList(
            [
                STFTResidualUnit(c, c, 2, (1, 2)),
                STFTResidualUnit(2 * c, 2 * c, 2, (2, 2)),
                STFTResidualUnit(4 * c, 4 * c, 1, (1, 2)),
                STFTResidualUnit(4 * c, 4 * c, 2, (2, 2)),
                STFTResidualUnit(8 * c, 8 * c, 1, (1, 2)),
                STFTResidualUnit(8 * c, 8 * c, 2, (2, 2)),
            ]
        )

        final_freq = self._final_freq_bins(window_length // 2 + 1)

        self.final = nn.Conv2d(
            in_channels=16 * c,
            out_channels=1,
            kernel_size=(1, final_freq),
        )

    def _final_freq_bins(self, freq_bins: int) -> int:
        for _ in range(6):
            freq_bins = freq_bins // 2
        return freq_bins

    def make_stft(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)

        spec = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            return_complex=True,
        )

        spec = torch.stack([spec.real, spec.imag], dim=1)
        return spec.permute(0, 1, 3, 2)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.make_stft(x)
        x = self.first(x)

        features = [x]

        for block in self.blocks:
            x = block(x)
            features.append(x)

        return {
            "logits": self.final(x),
            "features": features,
        }
