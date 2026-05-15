# reconstruction-only training entry point. GAN stage lives in train_gan.py

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.librispeech_codec import LibriSpeechCodecDataset
from src.logger import CometMLWriter
from src.loss.reconstruction_loss import CodecReconstructionLoss
from src.model.soundstream import SoundStream

warnings.filterwarnings("ignore", category=UserWarning)


def config_to_args(config: DictConfig) -> argparse.Namespace:
    cfg = OmegaConf.to_container(config, resolve=True)
    if cfg["device"] == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return argparse.Namespace(**cfg)


def make_run_name(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        return f"{ts}_{args.run_name.replace(' ', '_')}"
    return f"{ts}_soundstream_bs{args.batch_size}_lr{args.lr}_q{args.num_quantizers}_cb{args.codebook_size}"


def create_experiment(args: argparse.Namespace, run_name: str):
    if not args.use_comet:
        return None

    api_key = None
    workspace = args.comet_workspace
    try:
        import private_tokens

        api_key = getattr(private_tokens, "COMET_API_KEY", None)
        workspace = args.comet_workspace or getattr(private_tokens, "COMET_WORKSPACE", None)
    except ImportError:
        pass

    exp = CometMLWriter(
        project_name=args.comet_project,
        project_config=vars(args),
        workspace=workspace,
        run_name=run_name,
        api_key=api_key,
    )

    if args.run_description:
        exp.log_other("description", args.run_description)

    for tag in args.tag:
        exp.add_tag(tag)

    return exp


def update_lr(optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, step / warmup_steps)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def sync_ema_buffers(model: SoundStream) -> None:
    for vq in model.quantizer.quantizers:
        vq.ema_cluster_sum.data.copy_(vq.codebook.weight.data)
        vq.ema_cluster_size.fill_(1.0)


def save_checkpoint(
    path: Path,
    model: SoundStream,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: SoundStream,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt.get("codec", ckpt)
    missing, _ = model.load_state_dict(state, strict=False)
    if any("ema_" in k for k in missing):
        sync_ema_buffers(model)
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    if args.resume:
        resume_path = (ROOT / args.resume).resolve()
        run_dir = resume_path.parent
        run_name = run_dir.name
    else:
        run_name = make_run_name(args)
        run_dir = ROOT / args.save_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    exp = create_experiment(args, run_name)

    print(f"run: {run_name}")
    print(f"checkpoints: {run_dir.relative_to(ROOT)}")

    dataset = LibriSpeechCodecDataset(
        manifest_path=ROOT / args.manifest,
        root=ROOT,
        sample_rate=args.sample_rate,
        crop_seconds=args.crop_seconds,
        train=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    batches = infinite_loader(loader)

    model = SoundStream(
        in_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
    ).to(device)

    criterion = CodecReconstructionLoss(sample_rate=args.sample_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(resume_path, model, optimizer, device)
        print(f"resumed from step {start_step}")

    model.train()
    pbar = tqdm(range(start_step + 1, args.steps + 1), desc="training")

    last_metrics = {}

    for step in pbar:
        batch = next(batches)
        audio = batch["audio"].to(device, non_blocking=True)

        lr = update_lr(optimizer, args.lr, step, args.warmup_steps)

        optimizer.zero_grad(set_to_none=True)

        out = model(audio)
        losses = criterion(audio, out)
        loss = losses["loss"]

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step % args.log_every == 0:
            last_metrics = {
                "train/loss": losses["loss"].item(),
                "train/reconstruction_loss": losses["reconstruction_loss"].item(),
                "train/codebook_loss": losses["codebook_loss"].item(),
                "train/commitment_loss": losses["commitment_loss"].item(),
                "train/perplexity": losses["perplexity"].item(),
                "train/grad_norm": float(grad_norm),
                "train/lr": lr,
            }

            pbar.set_postfix(
                loss=f"{last_metrics['train/loss']:.4f}",
                rec=f"{last_metrics['train/reconstruction_loss']:.4f}",
                cb=f"{last_metrics['train/codebook_loss']:.4f}",
                commit=f"{last_metrics['train/commitment_loss']:.4f}",
                ppl=f"{last_metrics['train/perplexity']:.2f}",
            )

            if exp is not None:
                exp.log_metrics(last_metrics, step=step)

        if exp is not None and args.comet_audio_every > 0 and step % args.comet_audio_every == 0:
            exp.log_audio(
                audio[0, 0].detach().cpu().numpy(),
                sample_rate=args.sample_rate,
                file_name=f"step_{step:06d}_original.wav",
                step=step,
            )
            exp.log_audio(
                out["reconstruction"][0, 0].detach().cpu().numpy(),
                sample_rate=args.sample_rate,
                file_name=f"step_{step:06d}_reconstruction.wav",
                step=step,
            )

        if step % args.save_every == 0:
            metrics = {
                "loss": losses["loss"].item(),
                "reconstruction_loss": losses["reconstruction_loss"].item(),
                "codebook_loss": losses["codebook_loss"].item(),
                "commitment_loss": losses["commitment_loss"].item(),
                "perplexity": losses["perplexity"].item(),
            }

            step_path = run_dir / f"step_{step:06d}.pt"
            latest_path = run_dir / "latest.pt"

            save_checkpoint(step_path, model, optimizer, step, args, metrics)
            save_checkpoint(latest_path, model, optimizer, step, args, metrics)

            if exp is not None:
                exp.log_model(f"soundstream_step_{step:06d}", str(step_path))

    final_path = run_dir / "final.pt"
    latest_path = run_dir / "latest.pt"

    save_checkpoint(final_path, model, optimizer, args.steps, args, last_metrics)
    save_checkpoint(latest_path, model, optimizer, args.steps, args, last_metrics)

    if exp is not None:
        exp.log_model("soundstream_final", str(final_path))
        exp.end()


@hydra.main(version_base=None, config_path="../src/configs", config_name="codec_train")
def main(config: DictConfig) -> None:
    run(config_to_args(config))


if __name__ == "__main__":
    main()
