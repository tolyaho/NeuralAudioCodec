# Neural Audio Codec

SoundStream-style neural audio codec for 16 kHz speech at 6.4 kbps. Encoder + RVQ + decoder, trained from scratch on LibriSpeech `train-clean-100` with EMA codebook updates, an STFT discriminator, and a small waveform discriminator switched on at step 20 000 with hinge loss and feature matching on top of multi-scale mel reconstruction.

Final model on full `test-clean` (n = 2620):

| model                                          |   STOI |  NISQA |
| ---------------------------------------------- | -----: | -----: |
| reconstruction only                            | 0.9144 | 1.9335 |
| two-stage STFT-GAN (from reconstruction)       | 0.9317 | 2.3278 |
| scratch EMA RVQ + delayed STFT/wave GAN (final)| 0.9399 | 2.5212 |

Full precision: STOI `0.9398752048725391`, NISQA `2.5212083049857887`. Clears the grade-5 thresholds (STOI > 0.80, NISQA > 2.25) on both metrics.

## Report

Full Comet ML Report — problem framing, all experiments with curves and audio, ablations, GAN vs no-GAN bonus, per-step loss terms, codebook perplexity:

→ https://www.comet.com/tolya-ho/neural-audio-codec/reports/bjGhn710Lq4THJBv3eqoCNKnE

Supplementary analysis (qualitative in-domain, external English, Russian, quantitative) lives in `notebooks/analysis.ipynb`.

## Checkpoint

Final checkpoint: `checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt`. Everywhere else (this README, the demo, the analysis notebook) the stable alias `checkpoints/final_soundstream.pt` is used.

```bash
bash scripts/download_checkpoint.sh
```

`CHECKPOINT_URL` and `CHECKPOINT_DEST` override the URL and output path. HuggingFace `resolve/main` URLs and other direct links work as-is; Yandex Disk shares are resolved through the cloud-api.

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

## Logging

Comet logging is optional but used throughout training. Either set the environment variables or drop a local `private_tokens.py` (gitignored):

```bash
export COMET_API_KEY="..."
export COMET_WORKSPACE="..."
```

Hydra does not like commas in unquoted values. Wrap descriptions with commas like so:

```bash
'run_description="Final scratch EMA RVQ run with delayed STFT/wave GAN"'
```

## Training

All scripts use Hydra-style `key=value` overrides on top of the YAML configs in `src/configs/`. The three runs below match the three anchor runs in the report.

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

### Two-stage STFT-GAN (finetune from reconstruction)

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

### Final scratch EMA + delayed STFT/wave GAN

The configuration that ships. Reproduces the final checkpoint.

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

`src/configs/codec_gan.yaml` is pinned to these values, so an override-free `python scripts/train_gan.py` reproduces the final run.

## Evaluation

Full `test-clean` evaluation:

```bash
python scripts/inference.py checkpoint=checkpoints/final_soundstream.pt save_audio=20
```

Fast smoke (two utterances, no audio dumped):

```bash
python scripts/inference.py checkpoint=checkpoints/final_soundstream.pt limit=2 save_audio=0
```

Metrics (per-utterance STOI / NISQA and summary statistics) and audio land under `reports/eval/<run-or-stem>/`. The full per-utterance distribution for the final checkpoint is committed at `data/metrics_final.json` and consumed by the analysis notebook.

## Demo

`notebooks/demo.ipynb` clones the repo, installs dependencies, downloads the checkpoint, and re-synthesizes any user-provided audio URL through the codec. Open it in a fresh Google Colab session and run all cells.

## Analysis

`notebooks/analysis.ipynb` covers the qualitative, external-dataset, and quantitative analyses required by the assignment. Code lives in `src/analysis.py`; the notebook contains only markdown and thin calls into that module. Run `bash scripts/download_external.sh` once before opening the notebook to fetch the two external English clips (the Russian clips are bundled).

## Repo layout

```text
scripts/    training, evaluation, data prep
src/        model, datasets, losses, discriminators, logger, analysis
notebooks/  analysis + colab demo
tests/      light interface tests
```

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
