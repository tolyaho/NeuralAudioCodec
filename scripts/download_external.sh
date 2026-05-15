#!/usr/bin/env bash
# pulls the external speech samples used by notebooks/analysis.ipynb.
# russian samples are not bundled — drop your own wavs into data/external/russian/.
# from mp3 (mono 16 khz wav): copy russian-dialogue-intro.mp3 there, then
#   bash scripts/prepare_russian_sample.sh
set -euo pipefail

DEST_EN="data/external/english"
mkdir -p "$DEST_EN" data/external/russian

fetch() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "[skip] $out"
    return
  fi
  echo "[get]  $url"
  curl -sSL --fail --retry 3 --retry-delay 3 -o "$out" "$url"
}

fetch "https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav"          "$DEST_EN/lj025-0076.wav"
fetch "https://github.com/microsoft/MS-SNSD/raw/master/clean_test/clnsp1.wav"  "$DEST_EN/microsoft_clean.wav"

echo "[done] english samples in $DEST_EN"
