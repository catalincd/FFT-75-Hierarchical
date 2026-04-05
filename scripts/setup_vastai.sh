#!/usr/bin/env bash
# One-shot setup for a fresh vast.ai instance.
#
# Usage (from repo root):
#   bash scripts/setup_vastai.sh [--hf-user YOUR_HF_USERNAME] [--skip-data] [--resume-checkpoint]
#
# What it does:
#   1. Installs Python dependencies (torch is usually pre-installed on vast.ai images)
#   2. Downloads the dataset from HuggingFace Hub -> data/4k_1/binary/
#   3. Optionally downloads the latest checkpoint from HuggingFace Hub -> checkpoints/
#
# Prerequisites:
#   - Python 3.10+, pip
#   - git clone of this repo, run from repo root
#   - HF_TOKEN env var set (or run `huggingface-cli login` first) if repo is private

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_USER=""
SKIP_DATA=0
PULL_CHECKPOINT=0

for arg in "$@"; do
  case $arg in
    --hf-user=*) HF_USER="${arg#*=}" ;;
    --hf-user)   shift; HF_USER="$1" ;;
    --skip-data) SKIP_DATA=1 ;;
    --resume-checkpoint) PULL_CHECKPOINT=1 ;;
  esac
done

if [[ -z "$HF_USER" ]]; then
  echo "Usage: bash scripts/setup_vastai.sh --hf-user YOUR_HF_USERNAME [--skip-data] [--resume-checkpoint]"
  exit 1
fi

echo "=== FFT-75 vast.ai Setup ==="
echo "HF user: $HF_USER"
echo "Repo root: $REPO_ROOT"
echo ""

# --- 1. Python dependencies ---
echo "[1/3] Installing Python dependencies..."
pip install -q -r requirements.txt
echo "Done."

# --- 2. Dataset ---
if [[ $SKIP_DATA -eq 0 ]]; then
  echo ""
  echo "[2/3] Downloading dataset from HuggingFace Hub..."
  echo "      Source: $HF_USER/fft75-dataset"
  echo "      Destination: data/4k_1/binary/"
  mkdir -p data/4k_1/binary

  huggingface-cli download \
    "$HF_USER/fft75-dataset" \
    --repo-type dataset \
    --local-dir data/4k_1/binary \
    --local-dir-use-symlinks False

  echo "Dataset downloaded."
else
  echo "[2/3] Skipping dataset download (--skip-data)."
fi

# --- 3. Checkpoint (optional) ---
if [[ $PULL_CHECKPOINT -eq 1 ]]; then
  echo ""
  echo "[3/3] Downloading latest checkpoint from HuggingFace Hub..."
  echo "      Source: $HF_USER/fft75-models"
  echo "      Destination: checkpoints/"
  mkdir -p checkpoints

  huggingface-cli download \
    "$HF_USER/fft75-models" \
    --repo-type model \
    --local-dir checkpoints \
    --local-dir-use-symlinks False

  echo "Checkpoint downloaded."
else
  echo "[3/3] Skipping checkpoint download (pass --resume-checkpoint to pull latest)."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start Phase 1 training:"
echo "  bash scripts/train_phase1.sh"
echo ""
echo "To resume from a checkpoint:"
echo "  bash scripts/train_phase1.sh --resume"
