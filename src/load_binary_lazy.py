"""
Lazy-loading dataset for FFT-75 binary fragment data.

Fragments are read on demand via file seeks; labels (~N bytes) are kept in RAM.
Use this instead of load_binary.load_split when the fragment file is too large
to fit in RAM (e.g. the full 25 GB FFT-75 train split).
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from load_binary import BINARY_DIR, label_indices_to_strings
from hierarchical_cascade import (
    TYPE_TO_GROUP, GROUP_TO_IDX, GROUP_LOCAL_IDX, GROUPS,
    _MAX_MAGIC_LEN, _is_header_for_label,
)


class LazyFragmentDataset(Dataset):
    """
    Reads 512-byte fragment sectors from disk on demand.

    Only label strings and file-row indices are kept in RAM.  The fragment
    file is opened lazily per DataLoader worker (fork-safe: the handle is
    dropped before pickling and re-opened on first access in the worker).

    See hierarchical_cascade.FragmentDataset for the meaning of with_position
    and rejection_prob — semantics are identical here, with the only
    difference being that out-of-group samples are read from disk on demand.

    Args:
        frag_path:    path to the raw binary fragment file
        sector_size:  bytes per fragment (typically 512)
        file_indices: (M,) int64 array — row positions inside frag_path
                      to use (supports subsampling without loading fragments)
        labels:       list[str] of fine-grained type names, length M
        mode:         "coarse" | "specialist:<group>"
        augment:      enable byte-noise augmentation (training only)
        noise_prob:   fraction of bytes to corrupt per sample
        with_position: emit per-sample is_header pseudo-label (coarse only)
        rejection_prob: probability of swapping in an out-of-group sample
                       labelled with the rejection class index (specialist only)
    """

    def __init__(
        self,
        frag_path:      Path,
        sector_size:    int,
        file_indices:   np.ndarray,
        labels:         list[str],
        mode:           str   = "coarse",
        augment:        bool  = False,
        noise_prob:     float = 0.02,
        with_position:  bool  = False,
        rejection_prob: float = 0.0,
    ):
        assert mode == "coarse" or mode.startswith("specialist:")
        assert len(file_indices) == len(labels)
        if with_position and mode != "coarse":
            raise ValueError("with_position only supported in coarse mode")
        if rejection_prob > 0 and mode == "coarse":
            raise ValueError("rejection_prob only supported in specialist mode")

        self.frag_path      = Path(frag_path)
        self.sector_size    = sector_size
        self.mode           = mode
        self.augment        = augment
        self.noise_prob     = noise_prob
        self.with_position  = with_position
        self.rejection_prob = rejection_prob
        self._fh            = None     # opened lazily per worker

        target_group = mode.split(":")[1] if ":" in mode else None

        if target_group is not None:
            keep = [
                i for i, lbl in enumerate(labels)
                if TYPE_TO_GROUP.get(lbl) == target_group
            ]
            self.file_indices = file_indices[keep]
            self.labels       = [labels[i] for i in keep]
            self.label_map    = GROUP_LOCAL_IDX[target_group]
            self.num_local    = len(GROUPS[target_group])
            self.target_group = target_group

            if rejection_prob > 0:
                # Out-of-group pool: file indices for every sample whose
                # group ≠ target.  Stored as int64 array so a random draw
                # is O(1).  No fragment bytes are loaded until __getitem__.
                out_keep = [
                    i for i, lbl in enumerate(labels)
                    if TYPE_TO_GROUP.get(lbl) != target_group
                ]
                if not out_keep:
                    raise RuntimeError(
                        f"rejection_prob>0 but no out-of-group samples available "
                        f"for group {target_group}"
                    )
                self._out_indices = np.asarray(
                    [int(file_indices[i]) for i in out_keep], dtype=np.int64
                )
            else:
                self._out_indices = None
        else:
            self.file_indices = np.asarray(file_indices, dtype=np.int64)
            self.labels       = list(labels)
            self.label_map    = GROUP_TO_IDX
            self.num_local    = None
            self.target_group = None
            self._out_indices = None

    def __len__(self) -> int:
        return len(self.file_indices)

    def _read(self, file_idx: int) -> bytes:
        if self._fh is None or self._fh.closed:
            self._fh = open(self.frag_path, "rb")
        self._fh.seek(file_idx * self.sector_size)
        return self._fh.read(self.sector_size)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.augment and self.noise_prob > 0:
            mask  = torch.rand(x.shape) < self.noise_prob
            noise = torch.randint_like(x, 0, 256)
            x     = torch.where(mask, noise, x)
        return x

    def __getitem__(self, idx: int):
        # Rejection-class injection (specialist mode only).
        if self._out_indices is not None and random.random() < self.rejection_prob:
            out_file_idx = int(self._out_indices[random.randrange(len(self._out_indices))])
            raw = self._read(out_file_idx)
            x = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
            x = self._augment(x)
            return x, self.num_local

        raw = self._read(int(self.file_indices[idx]))
        x = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
        x = self._augment(x)

        if self.mode == "coarse":
            label_str = self.labels[idx]
            y = self.label_map[TYPE_TO_GROUP[label_str]]
            if self.with_position:
                is_header = _is_header_for_label(raw[:_MAX_MAGIC_LEN], label_str)
                return x, y, is_header
            return x, y

        y = self.label_map[self.labels[idx]]
        return x, y

    def __getstate__(self):
        """Drop the file handle before pickling so DataLoader workers are safe."""
        state = self.__dict__.copy()
        state["_fh"] = None
        return state


def load_split_lazy(
    split: str,
    max_per_class: Optional[int] = None,
    fraction: Optional[float]    = None,
    binary_dir: Path             = BINARY_DIR,
    seed: int                    = 42,
) -> tuple[Path, np.ndarray, list[str], int]:
    """
    Load metadata and labels only; fragment data stays on disk.

    Subsampling (max_per_class / fraction) operates on the label array,
    which is tiny (~N bytes), so no fragment data is read at all.

    Returns:
        frag_path:    path to the binary fragment file
        file_indices: (N,) int64 array of row positions to use for seeks
        labels:       list[str] of fine-grained type names, length N
        sector_size:  bytes per fragment (typically 512)
    """
    meta_path  = binary_dir / f"{split}_meta.json"
    frag_path  = binary_dir / f"{split}_fragments.bin"
    label_path = binary_dir / f"{split}_labels.bin"

    for p in (meta_path, frag_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run convert_npz_to_binary.py first."
            )

    meta      = json.loads(meta_path.read_text())
    n         = meta["n_samples"]
    sector    = meta["sector_size"]
    all_types = meta["all_types"]

    label_indices = np.fromfile(label_path, dtype=np.uint8)
    file_indices  = np.arange(n, dtype=np.int64)

    if max_per_class is None and fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        counts        = np.bincount(label_indices)
        min_count     = int(counts[counts > 0].min())
        max_per_class = max(1, round(min_count * fraction))

    if max_per_class is not None:
        rng  = np.random.default_rng(seed)
        keep: list[int] = []
        for cls_idx in np.unique(label_indices):
            idx = np.where(label_indices == cls_idx)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep.extend(idx.tolist())
        keep_arr      = np.array(sorted(keep), dtype=np.int64)
        file_indices  = keep_arr
        label_indices = label_indices[keep_arr]

    labels = label_indices_to_strings(label_indices, all_types)
    return frag_path, file_indices, labels, sector
