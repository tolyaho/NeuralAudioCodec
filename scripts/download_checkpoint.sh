#!/usr/bin/env bash
# Yandex Disk shares resolved via cloud-api; other URLs fetched as-is.
set -euo pipefail

DEFAULT_URL="https://huggingface.co/tolyho/soundstream-librispeech-16khz/resolve/main/final-soundstream.pt"

URL="${CHECKPOINT_URL:-$DEFAULT_URL}"
DEST="${CHECKPOINT_DEST:-checkpoints/final_soundstream.pt}"

if [[ "$URL" == *"<"*">"* ]]; then
  echo "error: edit DEFAULT_URL in $0 or set CHECKPOINT_URL=..." >&2
  exit 1
fi

if [[ -f "$DEST" || -L "$DEST" ]]; then
  echo "[skip] $DEST already present"
  exit 0
fi

resolve_yandex() {
  python3 - "$1" <<'PY'
import json, sys, urllib.parse, urllib.request
share = sys.argv[1]
api = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
q = urllib.parse.quote(share, safe="")
with urllib.request.urlopen(f"{api}?public_key={q}", timeout=30) as r:
    print(json.load(r)["href"])
PY
}

if [[ "$URL" == *"disk.yandex"* ]]; then
  echo "[resolve] yandex share: $URL"
  URL="$(resolve_yandex "$URL")"
fi

mkdir -p "$(dirname "$DEST")"
echo "[download] $URL"
echo "[dest]     $DEST"
curl -L --fail --retry 3 --retry-delay 5 -o "$DEST" "$URL"

size=$(stat -c '%s' "$DEST" 2>/dev/null || stat -f '%z' "$DEST")
echo "[done] wrote $size bytes"
