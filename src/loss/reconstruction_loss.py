import math

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class MultiScaleMelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        scales: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
        n_mels: int = 64,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        self.scales = scales
        self.eps = eps

        self.mels = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=s,
                    win_length=s,
                    hop_length=s // 4,
                    n_mels=n_mels,
                    power=1.0,
                )
                for s in scales
            ]
        )

    def forward(self, x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        reconstruction = reconstruction.squeeze(1)

        loss = 0.0

        for scale, mel in zip(self.scales, self.mels):
            x_mel = mel(x)
            rec_mel = mel(reconstruction)

            mag_loss = F.l1_loss(rec_mel, x_mel)

            alpha = math.sqrt(scale / 2)
            diff = torch.log(rec_mel + self.eps) - torch.log(x_mel + self.eps)
            log_loss = torch.linalg.vector_norm(diff, ord=2, dim=1).mean()
            log_loss = log_loss / math.sqrt(diff.shape[1])

            loss = loss + mag_loss + alpha * log_loss

        return loss / len(self.scales)


class CodecReconstructionLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        reconstruction_weight: float = 1.0,
        codebook_weight: float = 1.0,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.codebook_weight = codebook_weight
        self.commitment_weight = commitment_weight

        self.reconstruction_loss = MultiScaleMelSpectrogramLoss(
            sample_rate=sample_rate,
        )

    def forward(self, x: torch.Tensor, out: dict) -> dict:
        reconstruction = out["reconstruction"]

        if reconstruction.shape != x.shape:
            raise ValueError(f"Expected reconstruction shape {x.shape}, got {reconstruction.shape}")

        rec_loss = self.reconstruction_loss(x, reconstruction)
        codebook_loss = out["codebook_loss"]
        commitment_loss = out["commitment_loss"]

        loss = (
            self.reconstruction_weight * rec_loss
            + self.codebook_weight * codebook_loss
            + self.commitment_weight * commitment_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": rec_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": out["perplexity"],
        }
