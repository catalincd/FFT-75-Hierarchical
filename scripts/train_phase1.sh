#!/usr/bin/env bash
# Train Phase 1 coarse classifier (CoarseEncoder + CoarseClassifier).
# CoarseEncoder = ByteEncoder (CNN) + BigramBranch + EntropyBranch.
#
# Usage (from repo root):
#   bash scripts/train_phase1.sh [extra args passed to train_phase1.py]
#
# Examples:
#   bash scripts/train_phase1.sh                          # full dataset, 30 epochs
#   bash scripts/train_phase1.sh --resume                 # resume from latest checkpoint
#   bash scripts/train_phase1.sh --max-per-class 500 --epochs 2   # quick smoke test
#
# Checkpoints are saved to checkpoints/phase1_archive/ by default.
# After training, upload them with: bash scripts/upload_model.sh YOUR_HF_USERNAME

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BINARY_DIR="data/4k_1/binary"
ARCHIVE_DIR="checkpoints/phase1_archive"

if [[ ! -d "$BINARY_DIR" ]]; then
  echo "Error: data not found at $BINARY_DIR"
  echo "Run: bash scripts/download_data.sh YOUR_HF_USERNAME"
  exit 1
fi

mkdir -p "$ARCHIVE_DIR"

echo "=== Phase 1 Training ==="
echo "Data:     $BINARY_DIR"
echo "Archive:  $ARCHIVE_DIR"
echo ""

# --epochs 30: conservative so the cosine LR schedule completes inside ~5h even
#   if epochs run slow. Truncating a 50-epoch schedule leaves the LR high and the
#   model worse than a fully-annealed shorter run.
# --lr 1e-3 + --min-lr 1e-5: matches the regime that produced the 87% baseline.
# --grad-accum 2 + --grad-checkpoint: effective batch 1024 with memory headroom
#   for the added bigram branch at 4096-byte fragments.
# --cutmix-alpha 0.2: mild byte CutMix; byte-noise augmentation is on by default.
PYTHONPATH=src python src/train_phase1.py \
  --binary-dir     "$BINARY_DIR" \
  --archive-dir    "$ARCHIVE_DIR" \
  --epochs         30 \
  --batch-size     512 \
  --lr             1e-3 \
  --min-lr         1e-5 \
  --grad-accum     2 \
  --grad-checkpoint \
  --compile \
  --cutmix-alpha   0.2 \
  --lazy \
  "$@"
