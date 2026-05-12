set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"
cd "${RAW}"

BASE_URL="https://www.openslr.org/resources/12"

download() {
  local name="$1"
  if [[ -d "${name}" ]]; then
    echo "[skip] ${name}/ already exists"
    return 0
  fi
  if [[ -f "${name}.tar.gz" ]]; then
    echo "[skip] ${name}.tar.gz already present"
  else
    echo "[download] ${name}.tar.gz ..."
    curl -L --fail --retry 3 --retry-delay 5 -o "${name}.tar.gz" "${BASE_URL}/${name}.tar.gz"
  fi
  echo "[extract] ${name}.tar.gz"
  tar -xzf "${name}.tar.gz"
}

download "train-clean-100"
download "test-clean"

echo "done: audio is under ${RAW}/LibriSpeech/"
