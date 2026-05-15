# residual VQ with EMA codebook updates and a straight-through estimator.

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, dim: int, codebook_size: int, ema_decay: float = 0.99) -> None:
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -(dim**-0.5), dim**-0.5)

        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_cluster_sum", self.codebook.weight.data.clone())

    def compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(
            indices.reshape(-1),
            minlength=self.codebook_size,
        ).float()

        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        return torch.exp(entropy)

    def forward(self, x: torch.Tensor) -> dict:
        b, d, t = x.shape

        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        x_flat = x.permute(0, 2, 1).reshape(-1, self.dim)

        distances = (
            x_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * x_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
        )

        indices = distances.argmin(dim=1)

        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).to(x_flat.dtype)
            cluster_size = one_hot.sum(0)
            cluster_sum = one_hot.t() @ x_flat.detach()

            self.ema_cluster_size.mul_(self.ema_decay).add_(
                cluster_size,
                alpha=1 - self.ema_decay,
            )
            self.ema_cluster_sum.mul_(self.ema_decay).add_(
                cluster_sum,
                alpha=1 - self.ema_decay,
            )

            n = self.ema_cluster_size.sum()
            smoothed = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.codebook_size * 1e-5)
                * n
            )

            self.codebook.weight.data.copy_(self.ema_cluster_sum / smoothed.unsqueeze(1))

        quantized = self.codebook.weight[indices].view(b, t, d).permute(0, 2, 1)
        indices_2d = indices.view(b, t)

        commitment_loss = F.mse_loss(x, quantized.detach())
        perplexity = self.compute_perplexity(indices_2d)

        return {
            "quantized": quantized,
            "indices": indices_2d,
            "codebook_loss": torch.zeros((), device=x.device),
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int = 1024,
        num_quantizers: int = 8,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    dim=dim,
                    codebook_size=codebook_size,
                    ema_decay=ema_decay,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z: torch.Tensor) -> dict:
        residual = z
        quantized_sum = torch.zeros_like(z)

        all_indices = []
        commitment_loss = 0.0
        perplexities = []

        for quantizer in self.quantizers:
            out = quantizer(residual)
            q = out["quantized"]

            quantized_sum = quantized_sum + q
            residual = residual - q.detach()

            all_indices.append(out["indices"])
            commitment_loss = commitment_loss + out["commitment_loss"]
            perplexities.append(out["perplexity"])

        z_q = z + (quantized_sum - z).detach()

        return {
            "z_q": z_q,
            "indices": torch.stack(all_indices, dim=0),
            "codebook_loss": torch.zeros((), device=z.device),
            "commitment_loss": commitment_loss / self.num_quantizers,
            "perplexity": torch.stack(perplexities).mean(),
        }
