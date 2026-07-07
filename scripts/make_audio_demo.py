"""Export original/reconstruction pairs from the released checkpoint for the README audio table.

Every clip goes through the same per-utterance path as `scripts/inference.py`: the full
utterance, the reconstruction clamped to [-1, 1], torchmetrics STOI against the original, and
NISQA v2.0 on the reconstruction. NISQA is also run on the original, so you can see how much of a
low score was already in the source. The script writes 16 kHz mono FLAC pairs and `metrics.json`
to `assets/audio/`.

Inputs: the first N utterances of the test-clean manifest (`scripts/prepare_data.sh` +
`scripts/process_data.py`) and every clip in `data/external/russian/`.

    python scripts/make_audio_demo.py --device cpu
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchmetrics
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from inference import build_metrics, load_codec  # noqa: E402

from src.datasets.librispeech_codec import load_audio, read_jsonl  # noqa: E402
from src.model.soundstream import SoundStream  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = load_audio(path)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/final_soundstream.pt")
    parser.add_argument("--manifest", default="data/manifests/test-clean.jsonl")
    parser.add_argument("--num-test-clean", type=int, default=4)
    parser.add_argument("--extra-dir", default="data/external/russian")
    parser.add_argument("--out-dir", default="assets/audio")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = OmegaConf.load(ROOT / "src/configs/codec_eval.yaml")
    sample_rate = int(cfg.sample_rate)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    model = SoundStream(
        in_channels=1,
        base_channels=cfg.base_channels,
        latent_dim=cfg.latent_dim,
        codebook_size=cfg.codebook_size,
        num_quantizers=cfg.num_quantizers,
    ).to(device)
    ckpt_path = ROOT / args.checkpoint
    ckpt = load_codec(ckpt_path, model, device)
    model.eval()
    stoi_metric, nisqa_metric = build_metrics(sample_rate, device)

    clips = []
    manifest = read_jsonl(ROOT / args.manifest)
    for index, item in enumerate(manifest[: args.num_test_clean]):
        clips.append(
            {
                "key": item["utt_id"],
                "source": "LibriSpeech test-clean",
                "manifest_index": index,
                "path": ROOT / item["audio_path"],
            }
        )
    for path in sorted((ROOT / args.extra_dir).iterdir()):
        if path.suffix.lower() in AUDIO_SUFFIXES:
            clips.append(
                {"key": path.stem, "source": str(path.relative_to(ROOT)), "path": path}
            )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for clip in clips:
            audio = (
                load_mono(clip["path"], sample_rate).unsqueeze(0).to(device)
            )  # (1, 1, T)
            rec = model(audio)["reconstruction"].clamp(-1.0, 1.0)
            target, pred = audio.squeeze(1), rec.squeeze(1)
            row = {k: v for k, v in clip.items() if k != "path"}
            row["duration_s"] = round(target.shape[-1] / sample_rate, 3)
            row["stoi"] = float(stoi_metric(pred, target))
            row["nisqa_reconstruction"] = float(nisqa_metric(pred)[0])
            row["nisqa_original"] = float(nisqa_metric(target)[0])
            for side, signal in (("original", target), ("reconstruction", pred)):
                name = f"{clip['key']}_{side}.flac"
                sf.write(
                    out_dir / name,
                    signal[0].cpu().numpy(),
                    sample_rate,
                    format="FLAC",
                    subtype="PCM_16",
                )
                row[side] = f"{args.out_dir}/{name}"
            rows.append(row)
            print(
                f"{row['key']:>22}  {row['duration_s']:6.2f}s  STOI {row['stoi']:.4f}  "
                f"NISQA {row['nisqa_original']:.2f} -> {row['nisqa_reconstruction']:.2f}"
            )

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "device": str(device),
        "torch": torch.__version__,
        "torchmetrics": torchmetrics.__version__,
        "clips": rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} pairs and metrics.json to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
