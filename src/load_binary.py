"""
Load binary fragment data produced by convert_npz_to_binary.py.

Usage:
    from load_binary import load_split, load_all_splits

    train_frags, train_labels = load_split("train")
    splits = load_all_splits()   # {"train": (...), "val": (...), "test": (...)}

np.fromfile on a raw binary is typically 3-5x faster than np.load on a
compressed NPZ, and supports memory-mapping for zero-copy access.
"""

import json
import numpy as np
from pathlib import Path
from typing import Literal

BINARY_DIR = Path("4k_1/binary")

Split = Literal["train", "val", "test"]


def load_split(
    split: Split,
    mmap: bool = False,
    binary_dir: Path = BINARY_DIR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a single split from binary files.

    Args:
        split:      "train", "val", or "test"
        mmap:       If True, memory-map the fragment array (zero RAM copy,
                    useful when the array is larger than available RAM).
        binary_dir: Directory containing the binary files.

    Returns:
        fragments:  (N, 512) uint8 array of raw byte sectors
        labels:     (N,)     uint8 array of label indices
        all_types:  list[str] mapping index -> type string (len 75)
    """
    meta_path  = binary_dir / f"{split}_meta.json"
    frag_path  = binary_dir / f"{split}_fragments.bin"
    label_path = binary_dir / f"{split}_labels.bin"

    for p in (meta_path, frag_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run convert_npz_to_binary.py first."
            )

    meta = json.loads(meta_path.read_text())
    n         = meta["n_samples"]
    sector    = meta["sector_size"]
    all_types = meta["all_types"]

    if mmap:
        fragments = np.memmap(frag_path, dtype=np.uint8, mode="r", shape=(n, sector))
    else:
        fragments = np.fromfile(frag_path, dtype=np.uint8).reshape(n, sector)

    labels = np.fromfile(label_path, dtype=np.uint8)

    return fragments, labels, all_types


def load_all_splits(
    mmap: bool = False,
    binary_dir: Path = BINARY_DIR,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """
    Load all available splits (train / val / test).

    Returns a dict keyed by split name. Missing splits are silently skipped.
    """
    result = {}
    for split in ("train", "val", "test"):
        meta_path = binary_dir / f"{split}_meta.json"
        if not meta_path.exists():
            continue
        result[split] = load_split(split, mmap=mmap, binary_dir=binary_dir)
    return result


def label_indices_to_strings(
    indices: np.ndarray,
    all_types: list[str],
) -> list[str]:
    """Convert uint8 index array back to type strings."""
    return [all_types[i] for i in indices.tolist()]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    for split in ("train", "val", "test"):
        meta_path = BINARY_DIR / f"{split}_meta.json"
        if not meta_path.exists():
            continue

        t0 = time.perf_counter()
        fragments, labels, all_types = load_split(split)
        elapsed = time.perf_counter() - t0

        unique, counts = np.unique(labels, return_counts=True)
        print(
            f"[{split}] "
            f"fragments={fragments.shape}  "
            f"labels={labels.shape}  "
            f"classes={len(unique)}  "
            f"loaded in {elapsed*1000:.1f} ms"
        )
        sample_labels = label_indices_to_strings(labels[:5], all_types)
        print(f"  first 5 labels: {sample_labels}")
