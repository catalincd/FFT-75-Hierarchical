"""
GBFlip (Gaussian Bit-Flip) augmentation + a unified training Dataset wrapper.

Why GBFlip over plain byte noise
--------------------------------
Plain byte-noise (the original FragmentDataset's augmentation) replaces ~2%
of bytes with uniformly random values. This is a heavy edit per affected
byte (one byte's whole identity changes) but cheap globally — it doesn't
simulate any real corruption process.

GBFlip flips individual bits with a per-fragment rate ~ |N(0, sigma)|, clipped
at max_rate. This:
  - is the actual error model of the storage media we ultimately read from;
  - perturbs more bytes (any byte hit even once is "changed") but each change
    is small in Hamming distance, so structural signatures (magic headers,
    delimiter rhythm) survive while the model is forced to be invariant to
    low-order bit noise.
This is the de-facto standard augmentation in the file-fragment literature
(XMP, CarveFormer, ByteSCAN).

AugmentedDataset
----------------
Wraps a base (fragment, label) Dataset. Optionally:
  * applies GBFlip to the byte tensor (train split only)
  * appends per-sample cached teacher logits (so KD is a constant-time index)

Yields:
  (x, y)                       — base case
  (x, y, teacher_logits[idx])  — when teacher_logits is provided
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# GBFlip primitive (numpy uint8 in, numpy uint8 out)
# ---------------------------------------------------------------------------

def gbflip_uint8(
    block: np.ndarray,
    sigma:     float,
    max_rate:  float,
    rng:       np.random.Generator,
) -> np.ndarray:
    """
    Flip a Gaussian-distributed fraction of bits.

    Per-fragment flip rate = clip(|N(0, sigma)|, 0, max_rate). With sigma=0.01
    and max_rate=0.05, mean ≈ 0.008, ceiling 5%. Bit positions are sampled
    with replacement: faster than np.random.choice(..., replace=False) and
    in practice the duplicate rate is sub-1% at these densities.
    """
    if sigma <= 0.0:
        return block
    rate = float(min(abs(rng.normal(0.0, sigma)), max_rate))
    L    = block.size
    n_bits = int(rate * L * 8)
    if n_bits == 0:
        return block

    out  = block.copy()
    bit  = rng.integers(0, L * 8, size=n_bits, dtype=np.int64)
    byte = (bit >> 3).astype(np.int64)
    msk  = (np.uint8(1) << (bit & 7).astype(np.uint8))
    # bitwise_xor.at handles repeated indices correctly (a double-flip cancels).
    np.bitwise_xor.at(out, byte, msk)
    return out


# ---------------------------------------------------------------------------
# Unified training dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedDataset(Dataset):
    """
    Wraps a base Dataset whose __getitem__ returns (x_int64_tensor, y_int).

    Parameters
    ----------
    base            : FragmentDataset built with augment=False (we own the
                      augmentation policy here).
    gbflip_sigma    : >0 enables GBFlip. Set on the train wrapper, leave 0 for val.
    gbflip_max_rate : ceiling on the flip rate (0.05 in the XMP recipe).
    teacher_logits  : optional (N, C) tensor; appended to each sample so KD
                      is a tensor index instead of a teacher forward pass.
    seed            : seed for the per-worker numpy generator. Workers fork
                      from the parent, so we re-seed in __init__ by combining
                      base seed with a per-worker offset retrieved at call
                      time — keeps reproducibility without cross-worker
                      correlation.
    """

    def __init__(
        self,
        base:            Dataset,
        gbflip_sigma:    float = 0.0,
        gbflip_max_rate: float = 0.05,
        teacher_logits:  Optional[torch.Tensor] = None,
        seed:            int   = 0,
    ):
        if teacher_logits is not None:
            assert len(base) == len(teacher_logits), (
                f"dataset/logits mismatch: {len(base)} vs {len(teacher_logits)}"
            )
        self.base            = base
        self.gbflip_sigma    = float(gbflip_sigma)
        self.gbflip_max_rate = float(gbflip_max_rate)
        self.teacher_logits  = teacher_logits
        self._seed           = int(seed)
        self._rng:           Optional[np.random.Generator] = None  # lazy per-worker

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            try:
                info = torch.utils.data.get_worker_info()
                wid  = info.id if info is not None else 0
            except Exception:
                wid = 0
            self._rng = np.random.default_rng(self._seed + 1009 * wid)
        return self._rng

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if self.gbflip_sigma > 0.0:
            # x comes out of FragmentDataset as int64 — convert to uint8 for
            # the bit-fiddling, back to int64 for the embedding lookup.
            arr = x.numpy().astype(np.uint8, copy=False)
            arr = gbflip_uint8(
                arr, self.gbflip_sigma, self.gbflip_max_rate, self._get_rng()
            )
            x = torch.from_numpy(arr.astype(np.int64))
        if self.teacher_logits is not None:
            return x, y, self.teacher_logits[idx]
        return x, y
