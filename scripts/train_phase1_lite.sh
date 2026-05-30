#!/usr/bin/env bash
# Train the LITE Phase 1 coarse classifier (LiteCoarseClassifier).
#
# Lighter than the original CoarseEncoder (~150k vs ~10M params).
# Uses dark knowledge distillation from an existing Phase 1 checkpoint and a
# confusion-matrix-weighted CE term that directly penalises the dominant
# text<->archive mistakes from the previous run.
#
# Usage (from repo root):
#   bash scripts/train_phase1_lite.sh                        # 25% per class, 10 epochs (smoke test)
#   bash scripts/train_phase1_lite.sh --teacher data-old/best.pt
#   bash scripts/train_phase1_lite.sh --fraction 1.0 --epochs 30  # full run
#
# Extra args after the script name are forwarded to train.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BINARY_DIR="data/4k_1/binary"
ARCHIVE_DIR="checkpoints/phase1_lite_archive"

# Auto-detect a teacher checkpoint if one exists.
TEACHER_DEFAULT=""
for cand in \
    "checkpoints/phase1_archive/best.pt" \
    "phase1_full/best.pt" \
    "data-old/best.pt"; do
  if [[ -f "$cand" ]]; then
    TEACHER_DEFAULT="$cand"
    break
  fi
done

# Auto-detect a prior training_log.json to seed the confusion-cost matrix.
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

echo "=== Phase 1 LITE Training ==="
echo "Data:      $BINARY_DIR"
echo "Archive:   $ARCHIVE_DIR"
echo "Teacher:   ${TEACHER_DEFAULT:-<none>}"
echo "Confusion: ${CONFUSION_DEFAULT:-<uniform>}"
echo ""

# Defaults baked in:
#   --fraction 0.25 --epochs 10  : fast smoke test (~4 min/epoch on an A100)
#   --batch-size 256 --lr 2e-3   : a small model wants a higher LR
#   --kd-alpha 0.5 --kd-temp 4   : standard Hinton recipe
#   --confusion-lambda 0.5       : moderate; raise if text/archive recall stays low
#   --disk-image-weight 0.6      : preserves the original project decision
TEACHER_ARG=()
if [[ -n "$TEACHER_DEFAULT" ]]; then
  TEACHER_ARG=(--teacher "$TEACHER_DEFAULT")
fi

CONFUSION_ARG=()
if [[ -n "$CONFUSION_DEFAULT" ]]; then
  CONFUSION_ARG=(--confusion-source "$CONFUSION_DEFAULT")
fi

# Defaults tuned for "best accuracy at reasonable training time" (~1-2 h on
# a single mid-range GPU). Model now includes a 2-layer BiGRU + attention pool
# on top of the DS-CNN — adds ~250k params but is the key fix for the
# text<->archive bottleneck.
#
#   --fraction 1.0             full data
#   --epochs 15                long enough for cosine LR to fully anneal
#   --batch-size 512           AMP keeps this comfortable on a 16-24 GB card
#   --lr 2e-3 -> 1e-5          AdamW + linear warmup + cosine
#   --gbflip-sigma 0.01        XMP-style bit-flip aug
#   --label-smoothing 0.1      base
#   --container-smoothing 0.2  heavier smoothing on disk_image/database
#   --kd-alpha 0.5 / T=4       Hinton KD from cached teacher logits
#   --confusion-lambda 0.5     off-diagonal-confusion penalty
#
# Teacher logits are precomputed once at startup and cached to
# $ARCHIVE_DIR/teacher_logits_train_<fp>.pt; subsequent runs reuse the cache.
PYTHONPATH=src python src/phase1_lite/train.py \
  --binary-dir            "$BINARY_DIR" \
  --archive-dir           "$ARCHIVE_DIR" \
  --fraction              1.0 \
  --epochs                15 \
  --batch-size            512 \
  --lr                    2e-3 \
  --min-lr                1e-5 \
  --kd-alpha              0.5 \
  --kd-temp               4.0 \
  --confusion-lambda      0.5 \
  --gbflip-sigma          0.01 \
  --gbflip-max-rate       0.05 \
  --label-smoothing       0.1 \
  --container-smoothing   0.2 \
  --disk-image-weight     0.6 \
  --workers               6 \
  --lazy \
  "${TEACHER_ARG[@]}" \
  "${CONFUSION_ARG[@]}" \
  "$@"
