#!/usr/bin/env bash
# Train Phase 1 coarse classifier (ByteEncoder + CoarseClassifier).
#
# Usage (from repo root):
#   bash scripts/train_phase1.sh [extra args passed to train_phase1.py]
#
# Examples:
#   bash scripts/train_phase1.sh                          # full dataset, 50 epochs
#   bash scripts/train_phase1.sh --resume                 # resume from latest checkpoint
#   bash scripts/train_phase1.sh --max-per-class 5000     # cap per class, fast iteration
#   bash scripts/train_phase1.sh --compile                # torch.compile for 2-3x throughput
#
# Checkpoints are saved to src/phase1_archive/ by default.
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

PYTHONPATH=src python src/train_phase1.py \
  --binary-dir  "$BINARY_DIR" \
  --archive-dir "$ARCHIVE_DIR" \
  --epochs      50 \
  --batch-size  512 \
  --lr          1e-4 \
  --lazy \
  "$@"
