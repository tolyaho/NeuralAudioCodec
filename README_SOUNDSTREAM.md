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
  --steps 45000 \
  --batch-size 12 \
  --num-workers 2 \
  --log-every 100 \
  --save-every 5000 \
  --use-comet \
  --comet-audio-every 2500 \
  --warmup-steps 1000 \
  --lr 1e-4 \
  --run-name rec_only_mel_45k_logfix_warmup \
  --run-description "Reconstruction-only SoundStream baseline with multi-scale mel loss and RVQ losses." \
  --tag reconstruction \
  --tag baseline \
  --tag mel-loss \
  --tag rvq \
  --tag 45k
```

### Two-stage STFT-GAN

```bash
python scripts/train_gan.py \
  --steps 45000 \
  --batch-size 12 \
  --num-workers 2 \
  --log-every 100 \
  --save-every 5000 \
  --use-comet \
  --comet-audio-every 2500 \
  --pretrained-codec checkpoints/20260502_060459_rec_only_mel_45k_logfix_warmup/final.pt \
  --warmup-steps 1000 \
  --gan-warmup-steps 2000 \
  --lr 5e-5 \
  --disc-lr 2e-6 \
  --disc-base-channels 16 \
  --adv-weight 0.03 \
  --feat-weight 3.0 \
  --run-name gan_stft_soft_45k_from_rec45k \
  --run-description "STFT-GAN fine-tuning from the reconstruction baseline." \
  --tag gan \
  --tag two-stage \
  --tag stft-discriminator \
  --tag feature-matching \
  --tag 45k
```

### Final scratch EMA + STFT/wave GAN

```bash
python scripts/train_gan.py \
  --steps 45000 \
  --batch-size 12 \
  --num-workers 2 \
  --log-every 100 \
  --save-every 5000 \
  --use-comet \
  --comet-audio-every 2500 \
  --warmup-steps 1000 \
  --lr 1e-4 \
  --disc-lr 1e-6 \
  --disc-start-step 15000 \
  --gan-warmup-steps 10000 \
  --disc-every 4 \
  --disc-base-channels 16 \
  --use-wave-disc \
  --wave-disc-base-channels 16 \
  --adv-weight 0.03 \
  --feat-weight 5.0 \
  --run-name gan_scratch_ema_disc15k_fm5 \
  --run-description "Final scratch run with EMA RVQ, delayed STFT/waveform GAN, hinge loss, and feature matching." \
  --tag gan \
  --tag scratch \
  --tag final \
  --tag ema-rvq \
  --tag stft-discriminator \
  --tag waveform-discriminator \
  --tag delayed-discriminator \
  --tag feature-matching \
  --tag 45k
```

## Evaluation

Full test-clean evaluation:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/20260509_191350_gan_scratch_ema_disc20k_wave8_fm3_45k/final.pt \
  --save-audio 20
```

Quick check:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/20260507_082649_gan_scratch_ema_disc15k_fm5/final.pt \
  --limit 2 \
  --save-audio 0
```

Stable final checkpoint path:

```bash
mkdir -p checkpoints
cp checkpoints/20260507_082649_gan_scratch_ema_disc15k_fm5/final.pt checkpoints/final_soundstream.pt
```

## Results

| Model | STOI | NISQA |
| --- | ---: | ---: |
| Reconstruction-only | 0.9144 | 1.9335 |
| Two-stage STFT-GAN | 0.9317 | 2.3278 |
| Scratch EMA + STFT/wave GAN | 0.9409 | 2.3272 |

Final checkpoint:

```text
checkpoints/20260507_082649_gan_scratch_ema_disc15k_fm5/final.pt
STOI  = 0.9409373478580068
NISQA = 2.3272093147039414
```

## Inference

```bash
python scripts/infer.py \
  --checkpoint checkpoints/final_soundstream.pt \
  --input path/to/input.wav \
  --output reconstructed.wav
```

## Sanity checks

```bash
python -m compileall src scripts
```

```bash
python scripts/train.py \
  --steps 2 \
  --batch-size 2 \
  --num-workers 0 \
  --log-every 1 \
  --save-every 2 \
  --run-name debug_rec
```

```bash
python scripts/train_gan.py \
  --steps 2 \
  --batch-size 2 \
  --num-workers 0 \
  --log-every 1 \
  --save-every 2 \
  --use-wave-disc \
  --run-name debug_gan
```

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/final_soundstream.pt \
  --limit 2 \
  --save-audio 0
```
