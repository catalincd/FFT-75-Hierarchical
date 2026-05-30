"""
phase1_lite — lightweight reimplementation of the Phase 1 coarse classifier.

Three design pillars:
  - Tiny depthwise-separable CNN backbone + cheap position-free stats branches
    (entropy, byte histogram, structural-char freqs). ~100x fewer params than
    the original CoarseEncoder.
  - Dark knowledge distillation from the existing (heavy) Phase 1 checkpoint
    when one is available; falls back to plain CE when no teacher is given.
  - Confusion-matrix-weighted cross entropy: directly penalises predictions
    that fall into the high-confusion off-diagonal cells (text<->archive in
    particular) using statistics from the existing training_log.json.

Fast-iteration default: 25% per class, 10 epochs, batch 256.
"""

from .model    import LiteCoarseClassifier
from .losses   import ConfusionWeightedCE, distillation_kl
from .teacher  import load_teacher

__all__ = [
    "LiteCoarseClassifier",
    "ConfusionWeightedCE",
    "distillation_kl",
    "load_teacher",
]
