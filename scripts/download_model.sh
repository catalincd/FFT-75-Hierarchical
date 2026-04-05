#!/usr/bin/env bash
# Download trained checkpoints from HuggingFace Hub for local use.
#
# Usage (from repo root, on your local machine):
#   bash scripts/download_model.sh YOUR_HF_USERNAME [DEST_DIR]
#
# DEST_DIR defaults to checkpoints/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${1:-}"
DEST_DIR="${2:-checkpoints}"

if [[ -z "$HF_USER" ]]; then
  echo "Usage: bash scripts/download_model.sh YOUR_HF_USERNAME [DEST_DIR]"
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "Downloading checkpoints: $HF_USER/fft75-models -> $DEST_DIR/"
huggingface-cli download \
  "$HF_USER/fft75-models" \
  --repo-type model \
  --local-dir "$DEST_DIR" \
  --local-dir-use-symlinks False

echo "Done. Files in $DEST_DIR/:"
ls -lh "$DEST_DIR/"
