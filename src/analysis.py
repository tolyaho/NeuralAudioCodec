import json
import urllib.request
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from IPython.display import Audio, display

from src.datasets.librispeech_codec import load_audio, read_jsonl
from src.model.soundstream import SoundStream


ROOT = Path(__file__).resolve().parents[1]
SR = 16000
N_FFT, HOP, N_MELS = 1024, 256, 80

CKPT = ROOT / "checkpoints/final_soundstream.pt"
METRICS = ROOT / "reports/eval/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/metrics.json"
TEST_MANIFEST = ROOT / "data/manifests/test-clean.jsonl"
EXT_DIR = ROOT / "data/external"

DEFAULT_IDXS = [0, 1, 50, 600]

# (url, local filename) — kept in sync with scripts/download_external.sh
EN_FILES = [
    ("https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav", "lj025-0076.wav"),
    ("https://github.com/microsoft/MS-SNSD/raw/master/clean_test/clnsp1.wav", "microsoft_clean.wav"),
]
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


_cached = None


def _model():
    global _cached
    if _cached is not None:
        return _cached
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SoundStream(
        in_channels=1, base_channels=32, latent_dim=512,
        codebook_size=1024, num_quantizers=8,
    ).to(device).eval()
    ckpt = torch.load(CKPT, map_location=device)
    state = ckpt.get("codec", ckpt.get("model", ckpt))
    missing, _ = m.load_state_dict(state, strict=False)
    # older checkpoints don't persist rvq EMA buffers — re-seed from the codebook weights
    if any("ema_" in k for k in missing):
        for vq in m.quantizer.quantizers:
            vq.ema_cluster_sum.data.copy_(vq.codebook.weight.data)
            vq.ema_cluster_size.fill_(1.0)
    _cached = (m, device)
    return _cached


def _run(path):
    wav, sr = load_audio(path)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    m, dev = _model()
    with torch.no_grad():
        rec = m(wav.unsqueeze(0).to(dev))["reconstruction"].clamp(-1, 1)
    return wav.squeeze().numpy(), rec.squeeze().cpu().numpy()


def _fetch(url, out):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print("fetching", out.name)
    with urllib.request.urlopen(url, timeout=30) as r, out.open("wb") as f:
        f.write(r.read())


def _ext_paths(folder, files=None):
    cache = EXT_DIR / folder
    cache.mkdir(parents=True, exist_ok=True)
    for url, name in files or []:
        try:
            _fetch(url, cache / name)
        except Exception as e:
            print(f"skipping {name}: {e}")
    return sorted(p for p in cache.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)


def _mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max)


def _show(samples):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(14, 2.7 * n), squeeze=False)
    for row, (title, o, r) in zip(axes, samples):
        t = np.arange(len(o)) / SR
        row[0].plot(t, o, lw=0.5, label="orig")
        row[0].plot(t, r, lw=0.5, alpha=0.7, label="rec")
        row[0].set_ylim(-1, 1)
        row[0].set_title(title, fontsize=9)
        row[0].legend(fontsize=7)
        for ax, sig in [(row[1], o), (row[2], r)]:
            librosa.display.specshow(_mel(sig), sr=SR, hop_length=HOP, ax=ax,
                                     y_axis="mel", x_axis="time", cmap="magma")
        row[1].set_title("original", fontsize=9)
        row[2].set_title("reconstruction", fontsize=9)
    fig.tight_layout()

    for title, o, r in samples:
        print(title)
        display(Audio(o, rate=SR))
        display(Audio(r, rate=SR))


def in_domain(indices=None):
    items = read_jsonl(TEST_MANIFEST)
    samples = []
    for i in indices or DEFAULT_IDXS:
        it = items[i]
        o, r = _run(ROOT / it["audio_path"])
        samples.append((f"{it['utt_id']} ({it['duration']:.1f}s)", o, r))
    _show(samples)


def external_english(files=EN_FILES):
    paths = _ext_paths("english", files)
    if not paths:
        print("no clips in data/external/english/")
        return
    _show([(p.stem, *_run(p)) for p in paths])


def russian():
    paths = _ext_paths("russian")
    if not paths:
        print("no clips in data/external/russian/")
        return
    _show([(p.stem, *_run(p)) for p in paths])


def metrics_summary():
    m = json.loads(METRICS.read_text())
    rows = []
    for name, key in [("STOI", "stoi_values"), ("NISQA v2.0", "nisqa_values")]:
        x = np.array(m[key])
        rows.append([name, len(x), x.mean(), x.std(), x.min(),
                     np.quantile(x, .25), np.median(x), np.quantile(x, .75), x.max()])
    cols = ["metric", "n", "mean", "std", "min", "p25", "median", "p75", "max"]
    return pd.DataFrame(rows, columns=cols).set_index("metric").round(3)


def metric_distributions():
    m = json.loads(METRICS.read_text())
    stoi = np.array(m["stoi_values"])
    nisqa = np.array(m["nisqa_values"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))

    axes[0].hist(stoi, bins=40, color="#4477aa", edgecolor="white")
    axes[0].axvline(stoi.mean(), color="k", ls="--", lw=1, label=f"mean {stoi.mean():.3f}")
    axes[0].set_title(f"STOI on test-clean (n={len(stoi)})")
    axes[0].set_xlabel("STOI")
    axes[0].legend(fontsize=8)

    axes[1].hist(nisqa, bins=40, color="#cc6677", edgecolor="white")
    axes[1].axvline(nisqa.mean(), color="k", ls="--", lw=1, label=f"mean {nisqa.mean():.3f}")
    axes[1].set_title(f"NISQA v2.0 on test-clean (n={len(nisqa)})")
    axes[1].set_xlabel("NISQA v2.0")
    axes[1].legend(fontsize=8)

    fig.tight_layout()


def spectral_stats(n=100, seed=0):
    rng = np.random.default_rng(seed)
    items = read_jsonl(TEST_MANIFEST)
    pick = rng.choice(len(items), size=min(n, len(items)), replace=False)

    co, cr, ho, hr, l1 = [], [], [], [], []
    hf = librosa.fft_frequencies(sr=SR, n_fft=N_FFT) > 4000

    for i in pick:
        o, r = _run(ROOT / items[int(i)]["audio_path"])
        co.append(librosa.feature.spectral_centroid(y=o, sr=SR).mean())
        cr.append(librosa.feature.spectral_centroid(y=r, sr=SR).mean())
        So = np.abs(librosa.stft(o, n_fft=N_FFT)) ** 2
        Sr = np.abs(librosa.stft(r, n_fft=N_FFT)) ** 2
        ho.append(So[hf].sum() / (So.sum() + 1e-9))
        hr.append(Sr[hf].sum() / (Sr.sum() + 1e-9))
        Mo, Mr = _mel(o), _mel(r)
        k = min(Mo.shape[1], Mr.shape[1])
        l1.append(np.abs(Mo[:, :k] - Mr[:, :k]).mean())

    co, cr, ho, hr, l1 = map(np.array, (co, cr, ho, hr, l1))
    return pd.DataFrame([
        ["spectral centroid (Hz)",
         f"{co.mean():.1f} ± {co.std():.1f}",
         f"{cr.mean():.1f} ± {cr.std():.1f}",
         f"{np.abs(co - cr).mean():.1f}"],
        [">4 kHz energy share",
         f"{ho.mean():.3f} ± {ho.std():.3f}",
         f"{hr.mean():.3f} ± {hr.std():.3f}",
         f"{np.abs(ho - hr).mean():.3f}"],
        ["log-mel L1 (dB)", "—", "—", f"{l1.mean():.2f}"],
    ], columns=["statistic", "original", "reconstruction", "|orig − rec|"])


def logmel_diff(index=0):
    it = read_jsonl(TEST_MANIFEST)[index]
    o, r = _run(ROOT / it["audio_path"])
    Mo, Mr = _mel(o), _mel(r)
    k = min(Mo.shape[1], Mr.shape[1])
    Mo, Mr = Mo[:, :k], Mr[:, :k]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    librosa.display.specshow(Mo, sr=SR, hop_length=HOP, ax=axes[0],
                             y_axis="mel", x_axis="time", cmap="magma")
    axes[0].set_title("original")
    librosa.display.specshow(Mr, sr=SR, hop_length=HOP, ax=axes[1],
                             y_axis="mel", x_axis="time", cmap="magma")
    axes[1].set_title("reconstruction")
    im = librosa.display.specshow(Mo - Mr, sr=SR, hop_length=HOP, ax=axes[2],
                                  y_axis="mel", x_axis="time",
                                  cmap="coolwarm", vmin=-20, vmax=20)
    axes[2].set_title(f"diff — {it['utt_id']}")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    fig.tight_layout()
