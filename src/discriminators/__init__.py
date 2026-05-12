from src.discriminators.stft_discriminator import STFTDiscriminator
from src.discriminators.waveform_discriminator import (
    MultiScaleWaveformDiscriminator,
    WaveformDiscriminator,
)

__all__ = [
    "STFTDiscriminator",
    "WaveformDiscriminator",
    "MultiScaleWaveformDiscriminator",
]
