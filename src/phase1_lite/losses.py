"""
Losses for phase1_lite.

ConfusionWeightedCE
-------------------
Standard cross entropy + an off-diagonal penalty term:

    loss(x, y) = CE(logits, y) + lam * <C[y, :], softmax(logits)>

where C is a (num_classes, num_classes) confusion-rate matrix derived from an
earlier training run (rows normalised, diagonal zeroed). Off-diagonal cells are
large exactly where the previous run kept making mistakes — adding them as a
weighted dot-product with the predicted probabilities pulls mass away from the
historically confused alternatives.

This is cheaper and more directly targeted than focal loss: focal asks
"how confident were you about the true class?", we ask "are you about to fall
into the same trap you fell into before?".

distillation_kl
---------------
Vanilla Hinton KD: KL between softened student/teacher logits at temperature T,
scaled by T^2 to preserve gradient magnitude when T > 1.
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_label_smoothing_vector(
    num_classes:      int,
    default_smoothing: float,
    class_overrides:   Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """
    Build a per-class smoothing tensor. Container/ambiguous classes get heavier
    smoothing via class_overrides (a dict of class index -> epsilon).
    """
    vec = torch.full((num_classes,), float(default_smoothing), dtype=torch.float32)
    if class_overrides:
        for idx, eps in class_overrides.items():
            vec[idx] = float(eps)
    return vec


# ---------------------------------------------------------------------------
# Confusion matrix → cost matrix
# ---------------------------------------------------------------------------

def _normalise_confusion(conf: np.ndarray) -> np.ndarray:
    """Row-normalise (recall view) and zero the diagonal."""
    conf = conf.astype(np.float32)
    row  = conf.sum(axis=1, keepdims=True).clip(1)
    cost = conf / row
    np.fill_diagonal(cost, 0.0)
    return cost


def cost_matrix_from_log(
    log_path: Union[str, Path],
    num_classes: int,
) -> torch.Tensor:
    """
    Read training_log.json (Phase 1 archive) and return a (C, C) cost tensor.

    The log layout (see train_phase1.py) stores the matrix only when a new best
    epoch is found, under key "confusion_matrix". When the key is absent we
    return a uniform off-diagonal cost so the training loop still runs.
    """
    log = json.loads(Path(log_path).read_text())
    raw = log.get("confusion_matrix")
    if raw is None:
        cost = np.ones((num_classes, num_classes), dtype=np.float32)
        np.fill_diagonal(cost, 0.0)
        cost /= cost.sum(axis=1, keepdims=True).clip(1)
        return torch.from_numpy(cost)
    return torch.from_numpy(_normalise_confusion(np.asarray(raw)))


def cost_matrix_from_array(conf: np.ndarray) -> torch.Tensor:
    """Same as cost_matrix_from_log but takes the raw confusion array."""
    return torch.from_numpy(_normalise_confusion(conf))


# ---------------------------------------------------------------------------
# ConfusionWeightedCE
# ---------------------------------------------------------------------------

def _per_class_smooth_ce(
    logits:          torch.Tensor,         # (B, C)
    target:          torch.Tensor,         # (B,)
    smoothing:       torch.Tensor,         # (C,) per-class epsilon
    class_weights:   Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Cross entropy with a per-class smoothing vector.

    Standard label smoothing places (1-eps) on the true class and eps/C on
    every class (including the true one in the "uniform" form). The loss
    becomes:

        L = (1 - eps_y) * NLL(y) + eps_y * (-mean log p)

    where the second term is the KL-with-uniform contribution.

    Implementing this manually (rather than calling F.cross_entropy with a
    scalar smoothing) lets us assign eps_y per-class — heavier smoothing on
    container/ambiguous classes (disk_image, database) where hard targets
    are misleading because a single fragment really is sometimes ambiguous.
    """
    log_p = F.log_softmax(logits, dim=-1)                              # (B, C)
    nll   = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)           # (B,)
    smooth_term = -log_p.mean(dim=-1)                                  # (B,)
    eps_y = smoothing[target]                                          # (B,)
    per_sample = (1.0 - eps_y) * nll + eps_y * smooth_term             # (B,)
    if class_weights is not None:
        w = class_weights[target]
        return (per_sample * w).sum() / w.sum().clamp(min=1e-8)
    return per_sample.mean()


class ConfusionWeightedCE(nn.Module):
    """
    Per-class-smoothed CE + an off-diagonal-confusion penalty.

    Parameters
    ----------
    cost_matrix     : (C, C) tensor. Row i, col j is the penalty multiplier for
                      placing probability on class j when the true label is i.
                      Diagonal should be 0.
    confusion_lambda: scalar weight on the penalty term. 0 disables it.
    class_weights   : optional (C,) tensor for per-class CE weights.
    label_smoothing : either a scalar (legacy) or a (C,) tensor with the
                      smoothing epsilon per class. Heavier values on
                      container classes prevent over-confident predictions
                      on genuinely-ambiguous fragments.
    """

    def __init__(
        self,
        cost_matrix:      torch.Tensor,
        confusion_lambda: float = 1.0,
        class_weights:    Optional[torch.Tensor]            = None,
        label_smoothing:  Union[float, torch.Tensor]        = 0.1,
    ):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix.float())
        self.confusion_lambda = float(confusion_lambda)

        # Always store smoothing as a (C,) tensor — uniform repeats the scalar.
        C = cost_matrix.shape[0]
        if isinstance(label_smoothing, torch.Tensor):
            assert label_smoothing.shape == (C,), \
                f"label_smoothing shape {tuple(label_smoothing.shape)} != ({C},)"
            smoothing_vec = label_smoothing.float()
        else:
            smoothing_vec = torch.full((C,), float(label_smoothing), dtype=torch.float32)
        self.register_buffer("label_smoothing", smoothing_vec)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        ce = _per_class_smooth_ce(
            logits, target,
            smoothing     = self.label_smoothing,
            class_weights = self.class_weights,
        )
        if self.confusion_lambda == 0.0:
            return ce

        probs   = F.softmax(logits, dim=-1)                          # (B, C)
        penalty = (self.cost_matrix[target] * probs).sum(dim=-1).mean()
        return ce + self.confusion_lambda * penalty


# ---------------------------------------------------------------------------
# Dark knowledge distillation (Hinton et al., 2015)
# ---------------------------------------------------------------------------

def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Symmetric T^2 scaling: KL(soft_teacher || soft_student) * T^2.

    Using batchmean reduction matches the standard Hinton recipe. The T^2
    factor keeps the gradient magnitude comparable to a T=1 cross entropy so
    you can combine this with CE on a single shared learning rate.
    """
    T = temperature
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T,     dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)
