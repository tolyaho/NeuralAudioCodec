# Neural Audio Codec

SoundStream-style neural audio codec for LibriSpeech speech reconstruction.

Final checkpoint reference: `checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt`.

Final full-test metrics: STOI `0.9409373478580068`, NISQA `2.3272093147039414`.

For exact run commands and results, see [`README_SOUNDSTREAM.md`](README_SOUNDSTREAM.md).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements.txt
```

CPU-only setup can skip `requirements-torch-cu126.txt`.

## Data

```bash
bash scripts/prepare_data.sh
python scripts/process_data.py
```

Expected manifests:

```text
data/manifests/train-clean-100.jsonl
data/manifests/test-clean.jsonl
```

## How To Use

Reconstruction training:

```bash
python scripts/train.py steps=45000 batch_size=12 use_comet=true
```

GAN fine-tuning:

```bash
python scripts/train_gan.py steps=45000 use_wave_disc=true use_comet=true
```

Inference / evaluation:

```bash
python scripts/inference.py checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt
```

Hydra overrides use `key=value` syntax. The default configs live in `src/configs/codec_train.yaml`, `src/configs/codec_gan.yaml`, and `src/configs/codec_eval.yaml`.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
