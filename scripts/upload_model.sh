#!/usr/bin/env bash
# Upload trained checkpoints to HuggingFace Hub after a training run.
#
# Usage (from repo root, on the vast.ai instance after training):
#   bash scripts/upload_model.sh YOUR_HF_USERNAME [CHECKPOINT_DIR]
#
# CHECKPOINT_DIR defaults to src/phase1_archive/ (where train_phase1.py saves).
# Set HF_TOKEN env var or run `huggingface-cli login` first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER="${1:-}"
CKPT_DIR="${2:-src/phase1_archive}"

if [[ -z "$HF_USER" ]]; then
  echo "Usage: bash scripts/upload_model.sh YOUR_HF_USERNAME [CHECKPOINT_DIR]"
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Error: checkpoint directory '$CKPT_DIR' not found."
  echo "Run training first, or pass the correct path as the second argument."
  exit 1
fi

echo "Uploading checkpoints: $CKPT_DIR -> $HF_USER/fft75-models"
huggingface-cli upload \
  "$HF_USER/fft75-models" \
  "$CKPT_DIR" \
  --repo-type model

echo "Done. View at: https://huggingface.co/$HF_USER/fft75-models"
