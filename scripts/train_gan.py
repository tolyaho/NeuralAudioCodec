"""Adversarial fine-tuning for SoundStream on LibriSpeech."""

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
from src.discriminators.stft_discriminator import STFTDiscriminator
from src.discriminators.waveform_discriminator import MultiScaleWaveformDiscriminator
from src.logger import CometMLWriter
from src.loss.adversarial_loss import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)
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
    return f"{ts}_soundstream_gan_bs{args.batch_size}_lr{args.lr}_dlr{args.disc_lr}"


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


def update_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    warmup_steps: int,
) -> float:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, step / warmup_steps)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def loss_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(value)


def mean_logits(out: dict) -> torch.Tensor:
    logits = out["logits"]
    if isinstance(logits, torch.Tensor):
        return logits.mean()
    return sum(l.mean() for l in logits) / len(logits)


def save_checkpoint(
    path: Path,
    codec: SoundStream,
    stft_disc: STFTDiscriminator,
    wave_disc: MultiScaleWaveformDiscriminator | None,
    codec_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": step,
            "codec": codec.state_dict(),
            "stft_disc": stft_disc.state_dict(),
            "wave_disc": wave_disc.state_dict() if wave_disc is not None else None,
            "codec_optimizer": codec_optimizer.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics or {},
        },
        path,
    )


def sync_ema_buffers(codec: SoundStream) -> None:
    for vq in codec.quantizer.quantizers:
        vq.ema_cluster_sum.data.copy_(vq.codebook.weight.data)
        vq.ema_cluster_size.fill_(1.0)


def load_pretrained_codec(path: Path, codec: SoundStream, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)

    if "model" in ckpt:
        state = ckpt["model"]
    elif "codec" in ckpt:
        state = ckpt["codec"]
    else:
        state = ckpt

    missing, _ = codec.load_state_dict(state, strict=False)
    if any("ema_" in k for k in missing):
        sync_ema_buffers(codec)

    print(f"loaded pretrained codec from {path}")


def load_gan_checkpoint(
    path: Path,
    codec: SoundStream,
    stft_disc: STFTDiscriminator,
    wave_disc: MultiScaleWaveformDiscriminator | None,
    codec_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["codec"] if "codec" in ckpt else ckpt.get("model", ckpt)

    missing, _ = codec.load_state_dict(state, strict=False)
    if any("ema_" in k for k in missing):
        sync_ema_buffers(codec)

    if "stft_disc" in ckpt:
        stft_disc.load_state_dict(ckpt["stft_disc"])
    elif "discriminator" in ckpt:
        stft_disc.load_state_dict(ckpt["discriminator"])

    if wave_disc is not None and ckpt.get("wave_disc") is not None:
        wave_disc.load_state_dict(ckpt["wave_disc"])

    if "codec_optimizer" in ckpt:
        codec_optimizer.load_state_dict(ckpt["codec_optimizer"])
    elif "optimizer" in ckpt:
        codec_optimizer.load_state_dict(ckpt["optimizer"])

    if "disc_optimizer" in ckpt:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])

    return int(ckpt.get("step", 0))


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

    codec = SoundStream(
        in_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
    ).to(device)

    stft_disc = STFTDiscriminator(
        base_channels=args.disc_base_channels,
    ).to(device)

    wave_disc = (
        MultiScaleWaveformDiscriminator(
            base_channels=args.wave_disc_base_channels,
        ).to(device)
        if args.use_wave_disc
        else None
    )

    discriminators = [stft_disc] + ([wave_disc] if wave_disc is not None else [])
    print("discriminators: stft" + (", wave (3-scale)" if wave_disc is not None else ""))

    rec_criterion = CodecReconstructionLoss(sample_rate=args.sample_rate).to(device)

    disc_params = list(stft_disc.parameters())
    if wave_disc is not None:
        disc_params += list(wave_disc.parameters())

    codec_optimizer = torch.optim.Adam(codec.parameters(), lr=args.lr)
    disc_optimizer = torch.optim.Adam(disc_params, lr=args.disc_lr)

    start_step = 0
    if args.resume:
        start_step = load_gan_checkpoint(
            resume_path,
            codec,
            stft_disc,
            wave_disc,
            codec_optimizer,
            disc_optimizer,
            device,
        )
        print(f"resumed GAN training from step {start_step}")
    elif args.pretrained_codec:
        load_pretrained_codec((ROOT / args.pretrained_codec).resolve(), codec, device)

    codec.train()
    for d in discriminators:
        d.train()

    pbar = tqdm(range(start_step + 1, args.steps + 1), desc="gan training")
    last_metrics = {}

    for step in pbar:
        batch = next(batches)
        audio = batch["audio"].to(device, non_blocking=True)

        codec_lr = update_lr(codec_optimizer, args.lr, step, args.warmup_steps)
        disc_lr = update_lr(disc_optimizer, args.disc_lr, step, args.warmup_steps)

        gan_step = max(0, step - args.disc_start_step)

        if step < args.disc_start_step:
            gan_scale = 0.0
        else:
            gan_scale = loss_scale(gan_step, args.gan_warmup_steps)

        adv_weight = args.adv_weight * gan_scale
        feat_weight = args.feat_weight * gan_scale

        d_loss = torch.tensor(0.0, device=device)
        d_losses_per_disc = [torch.tensor(0.0, device=device)] * len(discriminators)
        d_grad_norm = torch.tensor(0.0, device=device)
        real_logit_mean = torch.tensor(0.0, device=device)
        fake_logit_mean = torch.tensor(0.0, device=device)

        do_disc_update = (
            step >= args.disc_start_step
            and (step - args.disc_start_step) % args.disc_every == 0
        )

        if do_disc_update:
            for d in discriminators:
                set_requires_grad(d, True)
            disc_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_audio = codec(audio)["reconstruction"]

            real_outs = [d(audio) for d in discriminators]
            fake_outs = [d(fake_audio.detach()) for d in discriminators]

            d_losses_per_disc = [
                discriminator_hinge_loss(r["logits"], f["logits"])
                for r, f in zip(real_outs, fake_outs)
            ]
            d_loss = sum(d_losses_per_disc) / len(d_losses_per_disc)

            d_loss.backward()
            d_grad_norm = torch.nn.utils.clip_grad_norm_(disc_params, max_norm=10.0)
            disc_optimizer.step()

            real_logit_mean = sum(mean_logits(o) for o in real_outs).detach() / len(real_outs)
            fake_logit_mean = sum(mean_logits(o) for o in fake_outs).detach() / len(fake_outs)

        for d in discriminators:
            set_requires_grad(d, False)
        codec_optimizer.zero_grad(set_to_none=True)

        out = codec(audio)
        reconstruction = out["reconstruction"]
        rec_losses = rec_criterion(audio, out)

        if gan_scale > 0.0:
            fake_outs_g = [d(reconstruction) for d in discriminators]

            with torch.no_grad():
                real_outs_fm = [d(audio) for d in discriminators]

            g_adv_losses = [generator_hinge_loss(o["logits"]) for o in fake_outs_g]
            g_adv_loss = sum(g_adv_losses) / len(g_adv_losses)

            feat_losses = [
                feature_matching_loss(r["features"], f["features"])
                for r, f in zip(real_outs_fm, fake_outs_g)
            ]
            feat_loss = sum(feat_losses) / len(feat_losses)

            real_logit_mean = sum(mean_logits(o) for o in real_outs_fm).detach() / len(real_outs_fm)
            fake_logit_mean = sum(mean_logits(o) for o in fake_outs_g).detach() / len(fake_outs_g)
        else:
            g_adv_loss = torch.tensor(0.0, device=device)
            feat_loss = torch.tensor(0.0, device=device)

        g_loss = rec_losses["loss"] + adv_weight * g_adv_loss + feat_weight * feat_loss

        g_loss.backward()
        g_grad_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=10.0)
        codec_optimizer.step()

        for d in discriminators:
            set_requires_grad(d, True)

        if step % args.log_every == 0:
            last_metrics = {
                "train/g_loss": g_loss.item(),
                "train/d_loss": d_loss.item(),
                "train/d_loss_stft": d_losses_per_disc[0].item(),
                "train/reconstruction_loss": rec_losses["reconstruction_loss"].item(),
                "train/codebook_loss": rec_losses["codebook_loss"].item(),
                "train/commitment_loss": rec_losses["commitment_loss"].item(),
                "train/perplexity": rec_losses["perplexity"].item(),
                "train/g_adv_loss": g_adv_loss.item(),
                "train/feature_loss": feat_loss.item(),
                "train/g_grad_norm": float(g_grad_norm),
                "train/d_grad_norm": float(d_grad_norm),
                "train/lr": codec_lr,
                "train/disc_lr": disc_lr,
                "train/adv_weight": adv_weight,
                "train/feat_weight": feat_weight,
                "train/gan_scale": gan_scale,
                "train/real_logits": real_logit_mean.item(),
                "train/fake_logits": fake_logit_mean.item(),
            }

            if wave_disc is not None:
                last_metrics["train/d_loss_wave"] = d_losses_per_disc[1].item()

            pbar.set_postfix(
                g=f"{last_metrics['train/g_loss']:.3f}",
                d=f"{last_metrics['train/d_loss']:.3f}",
                rec=f"{last_metrics['train/reconstruction_loss']:.3f}",
                adv=f"{last_metrics['train/g_adv_loss']:.3f}",
                feat=f"{last_metrics['train/feature_loss']:.3f}",
                ppl=f"{last_metrics['train/perplexity']:.1f}",
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
                reconstruction[0, 0].detach().cpu().numpy(),
                sample_rate=args.sample_rate,
                file_name=f"step_{step:06d}_reconstruction.wav",
                step=step,
            )

        if step % args.save_every == 0:
            metrics = {
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "reconstruction_loss": rec_losses["reconstruction_loss"].item(),
                "codebook_loss": rec_losses["codebook_loss"].item(),
                "commitment_loss": rec_losses["commitment_loss"].item(),
                "perplexity": rec_losses["perplexity"].item(),
                "g_adv_loss": g_adv_loss.item(),
                "feature_loss": feat_loss.item(),
            }

            step_path = run_dir / f"step_{step:06d}.pt"
            latest_path = run_dir / "latest.pt"

            save_checkpoint(
                step_path,
                codec,
                stft_disc,
                wave_disc,
                codec_optimizer,
                disc_optimizer,
                step,
                args,
                metrics,
            )
            save_checkpoint(
                latest_path,
                codec,
                stft_disc,
                wave_disc,
                codec_optimizer,
                disc_optimizer,
                step,
                args,
                metrics,
            )

            if exp is not None:
                exp.log_model(f"soundstream_gan_step_{step:06d}", str(step_path))

    final_path = run_dir / "final.pt"
    latest_path = run_dir / "latest.pt"

    save_checkpoint(
        final_path,
        codec,
        stft_disc,
        wave_disc,
        codec_optimizer,
        disc_optimizer,
        args.steps,
        args,
        last_metrics,
    )
    save_checkpoint(
        latest_path,
        codec,
        stft_disc,
        wave_disc,
        codec_optimizer,
        disc_optimizer,
        args.steps,
        args,
        last_metrics,
    )

    if exp is not None:
        exp.log_model("soundstream_gan_final", str(final_path))
        exp.end()


@hydra.main(version_base=None, config_path="../src/configs", config_name="codec_gan")
def main(config: DictConfig) -> None:
    run(config_to_args(config))


if __name__ == "__main__":
    main()
