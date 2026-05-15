# Neural Audio Codec

SoundStream-style neural audio codec for LibriSpeech speech reconstruction.

The final model is trained from scratch with EMA RVQ, an STFT discriminator
and a small waveform discriminator, both turned on late with hinge loss and
feature matching. Discriminator starts at step 20k, wave discriminator base
channels 8, feature matching weight 3.0.

Final checkpoint:

```text
checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt
```

A stable path `checkpoints/final_soundstream.pt` is used everywhere else in
the docs. Fetch it with:

```bash
bash scripts/download_checkpoint.sh
```

`CHECKPOINT_URL` overrides the default URL baked into the script; `CHECKPOINT_DEST`
overrides the output path. Yandex Disk public share URLs
(`https://disk.yandex.ru/d/<id>`) are resolved to a direct download link through
the cloud-api; HuggingFace `resolve/main` URLs and other direct links work as-is.

Final full `test-clean` results:

| model                                          |   STOI |  NISQA |
| ---------------------------------------------- | -----: | -----: |
| reconstruction only                            | 0.9144 | 1.9335 |
| two-stage STFT-GAN (from reconstruction)       | 0.9317 | 2.3278 |
| scratch EMA RVQ + delayed STFT/wave GAN (final)| 0.9399 | 2.5212 |

Final numbers in full precision: STOI `0.9398752048725391`,
NISQA `2.5212083049857887`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements.txt
```

For CPU-only experiments `requirements.txt` alone is enough.

## Data

```bash
bash scripts/prepare_data.sh
python scripts/process_data.py
```

Produces:

```text
data/raw/LibriSpeech/
data/manifests/train-clean-100.jsonl
data/manifests/test-clean.jsonl
```

## Comet

Logging is optional. Either set the environment variables or drop a local
`private_tokens.py` (it is gitignored):

```bash
export COMET_API_KEY="..."
export COMET_WORKSPACE="..."
```

Hydra does not like commas in unquoted values. If a description has commas,
wrap it like so:

```bash
'run_description="Final scratch EMA RVQ run with delayed STFT/wave GAN"'
```

## Training

All scripts use Hydra-style `key=value` overrides on top of the YAML configs
under `src/configs/`.

### Reconstruction baseline

```bash
python scripts/train.py \
  steps=45000 \
  batch_size=12 \
  num_workers=2 \
  log_every=100 \
  save_every=5000 \
  use_comet=true \
  comet_audio_every=2500 \
  warmup_steps=1000 \
  lr=1e-4 \
  run_name=rec_only_mel_45k \
  tag='[reconstruction,baseline,mel-loss,rvq,45k]'
```

### Two-stage STFT-GAN (fine-tune from reconstruction)

```bash
python scripts/train_gan.py \
  steps=45000 \
  batch_size=12 \
  num_workers=2 \
  log_every=100 \
  save_every=5000 \
  use_comet=true \
  comet_audio_every=2500 \
  pretrained_codec=checkpoints/20260502_060459_rec_only_mel_45k_logfix_warmup/final.pt \
  warmup_steps=1000 \
  gan_warmup_steps=2000 \
  lr=5e-5 \
  disc_lr=2e-6 \
  disc_base_channels=16 \
  adv_weight=0.03 \
  feat_weight=3.0 \
  run_name=gan_stft_soft_45k_from_rec45k \
  tag='[gan,two-stage,stft-discriminator,feature-matching,45k]'
```

### Final scratch EMA + STFT/wave GAN

```bash
python scripts/train_gan.py \
  steps=45000 \
  batch_size=12 \
  num_workers=2 \
  log_every=100 \
  save_every=5000 \
  use_comet=true \
  comet_audio_every=2500 \
  warmup_steps=1000 \
  lr=1e-4 \
  disc_lr=5e-7 \
  disc_start_step=20000 \
  gan_warmup_steps=15000 \
  disc_every=4 \
  disc_base_channels=16 \
  use_wave_disc=true \
  wave_disc_base_channels=8 \
  adv_weight=0.02 \
  feat_weight=3.0 \
  run_name=gan_scratch_ema_disc20k_wave8_fm3_45k \
  tag='[gan,scratch,final,ema-rvq,stft-discriminator,waveform-discriminator,45k]'
```

## Evaluation

Full `test-clean` evaluation, using the stable checkpoint path:

```bash
python scripts/inference.py checkpoint=checkpoints/final_soundstream.pt save_audio=20
```

Fast smoke (two utterances, no audio dumped):

```bash
python scripts/inference.py checkpoint=checkpoints/final_soundstream.pt limit=2 save_audio=0
```

Metrics and audio land under `reports/eval/<run-or-stem>/`.

## Demo

The mandatory `notebooks/demo.ipynb` clones the repo, installs deps,
downloads the checkpoint and synthesizes a user-provided URL through the
codec. Open it in a fresh Google Colab session and run all cells.

## Repo layout

```text
scripts/   training, evaluation, data prep
src/       model, datasets, losses, discriminators, logger
docs/      assignment specification
tests/     light interface tests
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
