import torch
from torch import nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.rvq import ResidualVectorQuantizer


def match_length(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] > length:
        return x[..., :length]

    if x.shape[-1] < length:
        pad_size = length - x.shape[-1]
        pad = x[..., -1:].expand(*x.shape[:-1], pad_size)
        return torch.cat([x, pad], dim=-1)

    return x


class SoundStream(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 512,
        strides: tuple[int, ...] = (2, 4, 5, 5),
        codebook_size: int = 1024,
        num_quantizers: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.strides = strides
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
        )

        self.quantizer = ResidualVectorQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
        )

    def forward(self, x: torch.Tensor) -> dict:
        length = x.shape[-1]

        z = self.encoder(x)
        vq_out = self.quantizer(z)
        reconstruction = self.decoder(vq_out["z_q"])
        reconstruction = match_length(reconstruction, length)

        return {
            "reconstruction": reconstruction,
            "z": z,
            "z_q": vq_out["z_q"],
            "indices": vq_out["indices"],
            "codebook_loss": vq_out["codebook_loss"],
            "commitment_loss": vq_out["commitment_loss"],
            "perplexity": vq_out["perplexity"],
        }

    def encode(self, x: torch.Tensor) -> dict:
        z = self.encoder(x)
        return self.quantizer(z)

    def decode(self, z_q: torch.Tensor, length: int | None = None) -> torch.Tensor:
        reconstruction = self.decoder(z_q)

        if length is not None:
            reconstruction = match_length(reconstruction, length)

        return reconstruction
