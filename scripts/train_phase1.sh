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

# --epochs 30: conservative so the cosine LR schedule completes inside ~5h.
#   Truncating the schedule leaves the LR high and the model worse than a
#   fully-annealed shorter run.
# --lr 1e-3 + --min-lr 1e-5: matches the regime that produced the 87% baseline.
# --batch-size 1024: one real batch, no gradient accumulation. Fits comfortably
#   in 95 GB VRAM and keeps the effective batch at the proven 1024, so lr needs
#   no rescaling. No --grad-checkpoint: with this much VRAM, recomputing encoder
#   activations in backward would only be a ~30% compute tax for memory we have.
# --cutmix-alpha 0.2: mild byte CutMix; byte-noise augmentation is on by default.
# --disk-image-weight 0.6: disk_image (iso/img/vmdk) is a container format whose
#   fragments are byte-identical to embedded content. A mild CE down-weight keeps
#   Phase 1 from over-claiming it without collapsing its recall (the epoch-2
#   matrix showed the over-prediction largely self-corrects as features sharpen).
PYTHONPATH=src python src/train_phase1.py \
  --binary-dir        "$BINARY_DIR" \
  --archive-dir       "$ARCHIVE_DIR" \
  --epochs            30 \
  --batch-size        1024 \
  --lr                1e-3 \
  --min-lr            1e-5 \
  --compile \
  --cutmix-alpha      0.2 \
  --disk-image-weight 0.6 \
  --lazy \
  "$@"
