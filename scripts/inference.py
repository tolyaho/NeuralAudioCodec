import argparse
import json
import sys
import warnings
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.librispeech_codec import LibriSpeechCodecDataset
from src.model.soundstream import SoundStream

warnings.filterwarnings("ignore", category=UserWarning)


def build_metrics(sample_rate: int, device: torch.device):
    from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    stoi_metric = ShortTimeObjectiveIntelligibility(
        fs=sample_rate,
        extended=False,
    ).to(device)

    nisqa_metric = NonIntrusiveSpeechQualityAssessment(
        fs=sample_rate,
    ).to(device)

    return stoi_metric, nisqa_metric


def config_to_args(config: DictConfig) -> argparse.Namespace:
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return argparse.Namespace(**cfg)


def sync_ema_buffers(model: SoundStream) -> None:
    for vq in model.quantizer.quantizers:
        vq.ema_cluster_sum.data.copy_(vq.codebook.weight.data)
        vq.ema_cluster_size.fill_(1.0)


def load_codec(path: Path, model: SoundStream, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "codec" in ckpt:
        state = ckpt["codec"]
    else:
        state = ckpt

    missing, _ = model.load_state_dict(state, strict=False)
    if any("ema_" in k for k in missing):
        sync_ema_buffers(model)

    return ckpt if isinstance(ckpt, dict) else {}


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    parent_name = ckpt_path.parent.name
    run_label = ckpt_path.stem if parent_name in {"checkpoints", "", "."} else parent_name

    output_dir = ROOT / args.output_dir / run_label
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = LibriSpeechCodecDataset(
        manifest_path=ROOT / args.manifest,
        root=ROOT,
        sample_rate=args.sample_rate,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SoundStream(
        in_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
    ).to(device)

    ckpt = load_codec(ROOT / args.checkpoint, model, device)
    model.eval()

    stoi_metric, nisqa_metric = build_metrics(args.sample_rate, device)

    stoi_values = []
    nisqa_values = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="evaluating")):
            if args.limit is not None and i >= args.limit:
                break

            audio = batch["audio"].to(device)

            out = model(audio)
            rec = out["reconstruction"].clamp(-1.0, 1.0)

            target = audio.squeeze(1)
            pred = rec.squeeze(1)

            stoi = stoi_metric(pred, target)
            nisqa = nisqa_metric(pred)

            stoi_values.append(float(stoi.detach().cpu()))
            nisqa_values.append(float(nisqa[0].detach().cpu()))

            if i < args.save_audio:
                utt_id = batch["utt_id"][0]

                original = target[0].detach().cpu().numpy()
                reconstructed = pred[0].detach().cpu().numpy()

                sf.write(audio_dir / f"{i:03d}_{utt_id}_original.wav", original, args.sample_rate)
                sf.write(audio_dir / f"{i:03d}_{utt_id}_reconstruction.wav", reconstructed, args.sample_rate)

    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": ckpt.get("step"),
        "num_samples": len(stoi_values),
        "stoi_mean": sum(stoi_values) / len(stoi_values),
        "nisqa_mean": sum(nisqa_values) / len(nisqa_values),
        "stoi_values": stoi_values,
        "nisqa_values": nisqa_values,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = {k: v for k, v in results.items() if not k.endswith("_values")}
    print(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path="../src/configs", config_name="codec_eval")
def main(config: DictConfig) -> None:
    run(config_to_args(config))


if __name__ == "__main__":
    main()
