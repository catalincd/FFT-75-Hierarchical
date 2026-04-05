#!/usr/bin/env bash
# Download the FFT-75 binary dataset from HuggingFace Hub.
#
# Usage (from repo root):
#   bash scripts/download_data.sh YOUR_HF_USERNAME
#
# The dataset repo should contain the binary split files:
#   train_fragments.bin, train_labels.bin, train_meta.json
#   val_fragments.bin,   val_labels.bin,   val_meta.json
#   test_fragments.bin,  test_labels.bin,  test_meta.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${1:-}"
if [[ -z "$HF_USER" ]]; then
  echo "Usage: bash scripts/download_data.sh YOUR_HF_USERNAME"
  exit 1
fi

mkdir -p data/4k_1/binary

echo "Downloading dataset: $HF_USER/fft75-dataset -> data/4k_1/binary/"
huggingface-cli download \
  "$HF_USER/fft75-dataset" \
  --repo-type dataset \
  --local-dir data/4k_1/binary \
  --local-dir-use-symlinks False

echo "Done. Files in data/4k_1/binary/:"
ls -lh data/4k_1/binary/
