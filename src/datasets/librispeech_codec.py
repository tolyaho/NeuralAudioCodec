"""LibriSpeech dataset utilities for codec training."""

import json
import random
from pathlib import Path

import soundfile as sf
import torch
import torch.utils.data as data


def read_jsonl(path: Path) -> list[dict]:
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    return items


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    try:
        audio = torch.from_numpy(audio).transpose(0, 1)
    except RuntimeError:
        audio = torch.tensor(audio.tolist(), dtype=torch.float32).transpose(0, 1)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio, sample_rate


def crop_or_pad(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    length = audio.shape[1]

    if length == target_length:
        return audio
    if length > target_length:
        start = random.randint(0, length - target_length)
        return audio[:, start : start + target_length]

    pad_size = target_length - length
    pad = audio[:, -1:].expand(-1, pad_size)
    return torch.cat([audio, pad], dim=1)


class LibriSpeechCodecDataset(data.Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | Path = ".",
        sample_rate: int = 16000,
        crop_seconds: float = 0.5,
        train: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.crop_size = int(sample_rate * crop_seconds)
        self.train = train

        self.items = read_jsonl(self.manifest_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]

        audio_path = self.root / item["audio_path"]
        audio, sample_rate = load_audio(audio_path)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.sample_rate}, got {sample_rate} for {audio_path}"
            )

        if self.train:
            audio = crop_or_pad(audio, self.crop_size)

        return {
            "audio": audio,
            "utt_id": item["utt_id"],
            "audio_path": item["audio_path"],
            "sample_rate": sample_rate,
        }
