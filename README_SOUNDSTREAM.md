# Neural Audio Codec

My SoundStream-ish codec experiments on LibriSpeech. The final model is trained
from scratch with EMA RVQ, STFT discriminator, delayed waveform discriminator,
hinge loss and feature matching.

Final checkpoint:

```text
checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt
```

## Notes to Reproduce

I used the CUDA torch requirements first, then the normal requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements.txt
```

For CPU-only stuff, `requirements.txt` is enough.

Data prep:

```bash
bash scripts/prepare_data.sh
python scripts/process_data.py
```

This should give:

```text
data/raw/LibriSpeech/
data/manifests/train-clean-100.jsonl
data/manifests/test-clean.jsonl
```

Comet is optional. I used either env vars or local `private_tokens.py`:

```bash
export COMET_API_KEY="..."
export COMET_WORKSPACE="..."
```

Hydra can be annoying with commas. If I pass a description with commas, I use
inner quotes like this:

```bash
'run_description="Final scratch run with EMA RVQ, delayed STFT/waveform GAN, hinge loss, and feature matching."'
```

## Runs

First I trained the reconstruction-only codec:

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
  tag='[reconstruction,baseline,mel-loss,rvq,45k]'
```

I also tried a two-stage STFT-GAN from the reconstruction checkpoint:

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

The final run was trained from scratch. Discriminator starts late; wave
discriminator is small because the stronger version was not worth it here.

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
  disc_start_step=20000 \
  gan_warmup_steps=10000 \
  disc_every=4 \
  disc_base_channels=16 \
  use_wave_disc=true \
  wave_disc_base_channels=8 \
  adv_weight=0.03 \
  feat_weight=3.0 \
  run_name=gan_scratch_ema_disc20k_wave8_fm3_45k \
  tag='[gan,scratch,final,ema-rvq,stft-discriminator,waveform-discriminator,45k]'
```

## Evaluation

Full test-clean evaluation:

```bash
python scripts/inference.py \
  checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  save_audio=20
```

Fast smoke check:

```bash
python scripts/inference.py \
  checkpoint=checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  limit=2 \
  save_audio=0
```

## Results

```text
reconstruction only              STOI 0.9144   NISQA 1.9335
two-stage STFT-GAN               STOI 0.9317   NISQA 2.3278
scratch EMA + STFT/wave GAN      STOI 0.9409   NISQA 2.3272
```

Final checkpoint full eval:

```text
STOI  = 0.9409373478580068
NISQA = 2.3272093147039414
```

## Small Checks

```bash
python -m compileall src scripts
```

```bash
python scripts/train.py steps=2 batch_size=2 num_workers=0 log_every=1 save_every=2 run_name=debug_rec
```

```bash
python scripts/train_gan.py steps=2 batch_size=2 num_workers=0 log_every=1 save_every=2 use_wave_disc=true run_name=debug_gan
```
