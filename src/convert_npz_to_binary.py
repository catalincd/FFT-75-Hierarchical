"""
Convert NPZ dataset files to raw binary format for faster loading.

Output per split (train/val/test):
  <split>_fragments.bin  - raw uint8 C-contiguous array, shape (N, 512)
  <split>_labels.bin     - uint8 label indices, shape (N,)
  <split>_meta.json      - shape info and label mapping

Loading with np.fromfile is significantly faster than np.load on large arrays
because it skips format parsing and avoids decompression.

Usage:
    python convert_npz_to_binary.py --source-dir data/4k_1 --out-dir data/4k_1/binary
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Label vocabulary from hierarchical_cascade.py
GROUPS = {
    "image_raster":  ["jpg", "png", "bmp", "gif", "tiff", "webp"],
    "image_raw":     ["cr2", "nef", "arw", "dng", "orf"],
    "video":         ["mp4", "avi", "mkv", "mov", "wmv", "flv"],
    "audio":         ["mp3", "wav", "flac", "aac", "ogg", "m4a"],
    "document":      ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
    "text":          ["txt", "csv", "xml", "json", "html", "log"],
    "archive":       ["zip", "gz", "bz2", "7z", "tar", "rar"],
    "executable":    ["exe", "elf", "dll", "so", "class"],
    "database":      ["sqlite", "db", "mdb"],
    "disk_image":    ["iso", "img", "vmdk"],
    "other":         ["swf", "ttf", "ps", "eps", "psd"],
}
ALL_TYPES = [t for types in GROUPS.values() for t in types]
TYPE_TO_IDX = {t: i for i, t in enumerate(ALL_TYPES)}


def detect_keys(data: np.lib.npyio.NpzFile) -> tuple[str, str]:
    """Detect fragment and label array keys in NPZ."""
    keys = list(data.keys())
    print(f"  NPZ keys: {keys}")

    frag_key = None
    label_key = None

    for candidate in ["x", "X", "fragments", "data", "samples", "features"]:
        if candidate in keys:
            frag_key = candidate
            break

    for candidate in ["y", "Y", "labels", "label", "targets", "target"]:
        if candidate in keys:
            label_key = candidate
            break

    if frag_key is None or label_key is None:
        arrays = {k: data[k] for k in keys}
        sorted_by_size = sorted(arrays.items(), key=lambda kv: kv[1].nbytes, reverse=True)
        if frag_key is None:
            frag_key = sorted_by_size[0][0]
        if label_key is None:
            label_key = sorted_by_size[-1][0]

    return frag_key, label_key


def labels_to_indices(raw_labels: np.ndarray) -> np.ndarray:
    """Convert raw label array (string or int) to uint8 TYPE indices."""
    if raw_labels.dtype.kind in ("U", "S", "O"):
        labels_list = raw_labels.tolist()
        if isinstance(labels_list[0], bytes):
            labels_list = [b.decode() for b in labels_list]
        indices = np.array([TYPE_TO_IDX[lbl] for lbl in labels_list], dtype=np.uint8)
    else:
        indices = raw_labels.astype(np.uint8)
    return indices


def convert_split(npz_path: Path, out_dir: Path) -> None:
    split = npz_path.stem  # "train", "val", "test"
    print(f"\n[{split}] loading {npz_path} ...")

    data = np.load(npz_path, allow_pickle=True)
    frag_key, label_key = detect_keys(data)

    fragments = data[frag_key]
    raw_labels = data[label_key]

    print(f"  fragments: shape={fragments.shape}, dtype={fragments.dtype}")
    print(f"  labels:    shape={raw_labels.shape}, dtype={raw_labels.dtype}")

    if fragments.dtype != np.uint8:
        print(f"  casting fragments {fragments.dtype} -> uint8")
        fragments = fragments.astype(np.uint8)
    if not fragments.flags["C_CONTIGUOUS"]:
        fragments = np.ascontiguousarray(fragments)

    label_indices = labels_to_indices(raw_labels)

    # Detect which types are actually present and remap to contiguous 0..N-1
    unique_indices = np.unique(label_indices)
    out_of_range = [int(i) for i in unique_indices if i >= len(ALL_TYPES)]
    if out_of_range:
        print(f"  WARNING: label indices out of range for ALL_TYPES (len={len(ALL_TYPES)}): {out_of_range}")
        unique_indices = unique_indices[unique_indices < len(ALL_TYPES)]
    actual_types = [ALL_TYPES[i] for i in unique_indices]
    remap = np.zeros(256, dtype=np.uint8)
    for new_idx, old_idx in enumerate(unique_indices.tolist()):
        remap[old_idx] = new_idx
    label_indices = remap[label_indices]

    missing = set(range(len(ALL_TYPES))) - set(unique_indices.tolist())
    if missing:
        missing_names = [ALL_TYPES[i] for i in sorted(missing)]
        print(f"  types not present in data ({len(missing)}): {missing_names}")
    print(f"  classes found: {len(actual_types)} / {len(ALL_TYPES)}")

    frag_path  = out_dir / f"{split}_fragments.bin"
    label_path = out_dir / f"{split}_labels.bin"
    meta_path  = out_dir / f"{split}_meta.json"

    fragments.tofile(frag_path)
    label_indices.tofile(label_path)

    meta = {
        "split":       split,
        "n_samples":   fragments.shape[0],
        "sector_size": fragments.shape[1],
        "n_classes":   len(actual_types),
        "all_types":   actual_types,
        "frag_dtype":  "uint8",
        "label_dtype": "uint8",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    frag_mb  = frag_path.stat().st_size / 1e6
    label_kb = label_path.stat().st_size / 1e3
    print(f"  -> {frag_path.name} ({frag_mb:.1f} MB)")
    print(f"  -> {label_path.name} ({label_kb:.1f} KB)")
    print(f"  -> {meta_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NPZ dataset to binary format")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory containing train.npz, val.npz, test.npz")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for binary files (created if needed)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.source_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {args.source_dir}")

    print(f"Found {len(npz_files)} NPZ file(s): {[f.name for f in npz_files]}")

    for npz_path in npz_files:
        convert_split(npz_path, args.out_dir)

    print("\nDone. Binary files written to:", args.out_dir)


if __name__ == "__main__":
    main()
