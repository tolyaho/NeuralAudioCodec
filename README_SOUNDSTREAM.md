# Neural Audio Codec

SoundStream-style neural audio codec for LibriSpeech speech reconstruction.  
Final model: scratch EMA RVQ + delayed STFT/waveform GAN.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements.txt
```

CPU/non-CUDA setup:

```bash
python -m pip install -r requirements.txt
```

## Comet

```bash
export COMET_API_KEY="your_key"
export COMET_WORKSPACE="your_workspace"
```

or create `private_tokens.py`:

```python
COMET_API_KEY = "your_key"
COMET_WORKSPACE = "your_workspace"
```

Do not commit `private_tokens.py`.

## Data

```bash
bash scripts/prepare_data.sh
python scripts/process_data.py
```

Expected:

```text
data/raw/LibriSpeech/
data/manifests/train-clean-100.jsonl
data/manifests/test-clean.jsonl
```

## Training

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
  run_name=rec_only_mel_45k_logfix_warmup \
  'run_description=Reconstruction-only SoundStream baseline with multi-scale mel loss and RVQ losses.' \
  'tag=[reconstruction,baseline,mel-loss,rvq,45k]'
```

### Two-stage STFT-GAN

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
  'run_description=STFT-GAN fine-tuning from the reconstruction baseline.' \
  'tag=[gan,two-stage,stft-discriminator,feature-matching,45k]'
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
  disc_lr=1e-6 \
  disc_start_step=15000 \
  gan_warmup_steps=10000 \
  disc_every=4 \
  disc_base_channels=16 \
  use_wave_disc=true \
  wave_disc_base_channels=16 \
  adv_weight=0.03 \
  feat_weight=5.0 \
  run_name=gan_scratch_ema_disc15k_fm5 \
  'run_description=Final scratch run with EMA RVQ, delayed STFT/waveform GAN, hinge loss, and feature matching.' \
  'tag=[gan,scratch,final,ema-rvq,stft-discriminator,waveform-discriminator,delayed-discriminator,feature-matching,45k]'
```

## Inference / evaluation

Full test-clean evaluation:

```bash
python scripts/inference.py \
  checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  save_audio=20
```

Quick check:

```bash
python scripts/inference.py \
  checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  limit=2 \
  save_audio=0
```

## Results

| Model | STOI | NISQA |
| --- | ---: | ---: |
| Reconstruction-only | 0.9144 | 1.9335 |
| Two-stage STFT-GAN | 0.9317 | 2.3278 |
| Scratch EMA + STFT/wave GAN | 0.9409 | 2.3272 |

Final checkpoint:

```text
checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt
STOI  = 0.9409373478580068
NISQA = 2.3272093147039414
```

## Sanity checks

```bash
python -m compileall src scripts
```

```bash
python scripts/train.py \
  steps=2 \
  batch_size=2 \
  num_workers=0 \
  log_every=1 \
  save_every=2 \
  run_name=debug_rec
```

```bash
python scripts/train_gan.py \
  steps=2 \
  batch_size=2 \
  num_workers=0 \
  log_every=1 \
  save_every=2 \
  use_wave_disc=true \
  run_name=debug_gan
```

```bash
python scripts/inference.py \
  checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  limit=2 \
  save_audio=0
```
