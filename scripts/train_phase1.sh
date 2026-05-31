#!/usr/bin/env bash
# Train Phase 1 coarse classifier (CoarseEncoder + CoarseClassifier).
#
# CoarseEncoder = ByteEncoder (CNN + 2-layer Bi-GRU + attention pool) +
#                 BigramBranch + EntropyBranch + StructuralBranch.
#
# Loss = ConfusionWeightedCE: per-class label smoothing (heavier on
# disk_image / database) + off-diagonal confusion penalty seeded from
# a prior training_log.json.
#
# Augmentation: GBFlip (per-bit Gaussian noise) + optional byte CutMix.
#
# Usage (from repo root):
#   bash scripts/train_phase1.sh                          # full dataset, 30 epochs
#   bash scripts/train_phase1.sh --resume                 # resume from latest checkpoint
#   bash scripts/train_phase1.sh --fraction 0.1 --epochs 5  # quick smoke test
#
# Checkpoints are saved to checkpoints/phase1_archive/ by default.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BINARY_DIR="data/4k_1/binary"
ARCHIVE_DIR="checkpoints/phase1_archive"

# Auto-detect a prior training_log.json to seed the confusion cost matrix.
CONFUSION_DEFAULT=""
for cand in \
    "checkpoints/phase1_archive/training_log.json" \
    "phase1_full/training_log.json"; do
  if [[ -f "$cand" ]]; then
    CONFUSION_DEFAULT="$cand"
    break
  fi
done

if [[ ! -d "$BINARY_DIR" ]]; then
  echo "Error: data not found at $BINARY_DIR"
  echo "Run: bash scripts/download_data.sh YOUR_HF_USERNAME"
  exit 1
fi

mkdir -p "$ARCHIVE_DIR"

echo "=== Phase 1 Training ==="
echo "Data:      $BINARY_DIR"
echo "Archive:   $ARCHIVE_DIR"
echo "Confusion: ${CONFUSION_DEFAULT:-<uniform>}"
echo ""

# Hyperparameter notes:
# --epochs 30 + --warmup-pct 0.05 + --min-lr 1e-5:
#     Cosine schedule lasts the full run; LR ends at 1e-5 instead of crashing
#     to zero too early.
# --batch-size 512 + --grad-accum 2 -> effective batch 1024:
#     Matches the proven setup. Lower per-step VRAM than batch=1024 directly
#     (the new BiGRU adds activation memory).
# --gbflip-sigma 0.01 + --gbflip-max-rate 0.05:
#     XMP-recipe per-bit augmentation. Replaces the old byte-noise scheme.
# --cutmix-alpha 0.2:
#     Mild byte CutMix; valid for integer byte sequences (splices windows).
# --label-smoothing 0.1 + --container-smoothing 0.2:
#     Default smoothing 0.1; heavier 0.2 on disk_image and database where
#     fragments are genuinely ambiguous.
# --confusion-lambda 0.5:
#     Off-diagonal penalty weight. If text/archive recall is the bottleneck
#     in the resulting confusion matrix, raise to 1.0 and rerun.
# --disk-image-weight 0.6:
#     CE class weight down-weights disk_image so it stops absorbing other
#     classes' fragments at the coarse level.
CONFUSION_ARG=()
if [[ -n "$CONFUSION_DEFAULT" ]]; then
  CONFUSION_ARG=(--confusion-source "$CONFUSION_DEFAULT")
fi

PYTHONPATH=src python src/train_phase1.py \
  --binary-dir          "$BINARY_DIR" \
  --archive-dir         "$ARCHIVE_DIR" \
  --epochs              30 \
  --batch-size          512 \
  --grad-accum          2 \
  --lr                  1e-3 \
  --min-lr              1e-5 \
  --warmup-pct          0.05 \
  --compile \
  --grad-checkpoint \
  --lazy \
  --gbflip-sigma        0.01 \
  --gbflip-max-rate     0.05 \
  --cutmix-alpha        0.2 \
  --label-smoothing     0.1 \
  --container-smoothing 0.2 \
  --confusion-lambda    0.5 \
  --disk-image-weight   0.6 \
  "${CONFUSION_ARG[@]}" \
  "$@"
