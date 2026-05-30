"""
Teacher loader for distillation.

Wraps the existing (heavy) CoarseClassifier from hierarchical_cascade.py so
the student can train against its soft logits. The teacher is run in
inference-only mode — its parameters are frozen and gradients are disabled.

Handles two known checkpoint quirks:
  - torch.compile-wrapped state dicts (keys prefixed with "_orig_mod.")
  - per-epoch ckpts that store both "model" and "ema" — we honour the
    "best_source" field if present, otherwise prefer "ema" then "model".
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


def _strip_compile_prefix(state_dict: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def _strip_averaged_model_prefix(state_dict: dict) -> dict:
    """AveragedModel wraps the inner module as `.module` and adds n_averaged.

    Drop both so the dict slots into a plain CoarseClassifier."""
    out = {}
    for k, v in state_dict.items():
        if k == "n_averaged":
            continue
        if k.startswith("module."):
            out[k.removeprefix("module.")] = v
        else:
            out[k] = v
    return out


def _build_encoder_for_state_dict(state_dict: dict):
    """
    Pick the matching encoder class for an older or current checkpoint.

    The repo went through three encoder variants with growing out_dim:
      - ByteEncoder    : 1024 (CNN only)
      - FusedEncoder   : 1536 (CNN + Bigram)
      - CoarseEncoder  : 1664 (CNN + Bigram + Entropy + Struct)

    We detect which one produced the checkpoint by:
      1) presence of the bigram / entropy / structural sub-modules, then
      2) the size of `norm.weight` (= encoder.out_dim) as a sanity check.
    """
    from hierarchical_cascade import ByteEncoder, FusedEncoder, CoarseEncoder

    has_bigram  = any(k.startswith("encoder.bigram_enc.")  for k in state_dict)
    has_entropy = any(k.startswith("encoder.entropy_enc.") for k in state_dict)
    has_struct  = any(k.startswith("encoder.struct_enc.")  for k in state_dict)

    if has_entropy or has_struct:
        cls = CoarseEncoder
    elif has_bigram:
        cls = FusedEncoder
    else:
        cls = ByteEncoder

    encoder = cls()

    # Sanity check against norm.weight (= encoder.out_dim).
    norm_w = state_dict.get("norm.weight")
    if norm_w is not None and norm_w.shape[0] != encoder.out_dim:
        raise RuntimeError(
            f"Teacher encoder mismatch: state_dict expects out_dim={norm_w.shape[0]} "
            f"but {cls.__name__} produces {encoder.out_dim}."
        )
    return encoder, cls.__name__


def load_teacher(
    checkpoint_path: Union[str, Path],
    device:          str = "cpu",
    prefer_ema:      bool = True,
) -> nn.Module:
    """
    Reconstruct a CoarseClassifier (or whatever encoder it was trained with)
    and load weights from `checkpoint_path`. Returned model is in eval() with
    requires_grad=False on every parameter.

    The import of hierarchical_cascade is deferred so this subpackage doesn't
    fail to import when the heavy module isn't on the path yet.
    """
    # Deferred import — the lite package should still be usable when no teacher
    # is present, so we only pay the import cost when someone asks for one.
    from hierarchical_cascade import CoarseClassifier  # noqa: E402

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        source = ckpt.get("best_source")
        if source == "ema" and "ema" in ckpt:
            sd = _strip_averaged_model_prefix(ckpt["ema"])
        elif source == "raw":
            sd = ckpt["model"]
        elif prefer_ema and "ema" in ckpt:
            sd = _strip_averaged_model_prefix(ckpt["ema"])
        else:
            sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        # Raw state_dict was saved directly.
        sd = ckpt

    sd = _strip_compile_prefix(sd)

    encoder, enc_name = _build_encoder_for_state_dict(sd)
    print(f"[teacher] detected encoder: {enc_name} (out_dim={encoder.out_dim})")
    model = CoarseClassifier(encoder)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[teacher] {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"[teacher] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def teacher_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper — exists so the train loop can read clearly."""
    return model(x)


# ---------------------------------------------------------------------------
# Bulk pre-computation: run the teacher once over a fragment array and save
# the (N, NUM_GROUPS) logits to disk. Subsequent runs read the cache instead
# of paying the teacher forward cost per training step.
# ---------------------------------------------------------------------------

def _cache_path(archive_dir: Path, split: str, fingerprint: str) -> Path:
    """One cache per (split, data fingerprint) — avoids stale caches when the
    sampled subset changes between runs."""
    return archive_dir / f"teacher_logits_{split}_{fingerprint}.pt"


def precompute_logits(
    teacher: nn.Module,
    fragments,                      # numpy uint8 (N, L) — in-memory case
    device: str,
    batch_size: int = 512,
    progress: bool = True,
) -> torch.Tensor:
    """Run the teacher on the whole array and return (N, NUM_GROUPS) logits."""
    import numpy as np
    from tqdm import tqdm

    N = len(fragments)
    out_chunks = []
    teacher.eval()
    iterator = range(0, N, batch_size)
    if progress:
        iterator = tqdm(iterator, total=(N + batch_size - 1) // batch_size,
                        desc="caching teacher logits", unit="batch")
    with torch.inference_mode():
        for start in iterator:
            x_np = np.asarray(fragments[start:start + batch_size], dtype=np.int64)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            out_chunks.append(teacher(x).float().cpu())
    return torch.cat(out_chunks, dim=0)


def precompute_logits_from_loader(
    teacher: nn.Module,
    loader,                          # DataLoader yielding (x, y) in dataset order
    num_samples: int,
    device: str,
    progress: bool = True,
) -> torch.Tensor:
    """Lazy-mode equivalent of precompute_logits.

    The loader MUST iterate in dataset order (shuffle=False). Each batch's
    (x) is run through the teacher; results are stacked into a single
    (N, NUM_GROUPS) tensor that lines up with dataset indices.
    """
    from tqdm import tqdm

    out = None  # allocated on first batch once we know C
    cursor = 0
    teacher.eval()
    iterator = loader
    if progress:
        iterator = tqdm(loader, desc="caching teacher logits (lazy)", unit="batch")
    with torch.inference_mode():
        for batch in iterator:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            logits = teacher(x).float().cpu()
            if out is None:
                out = torch.zeros((num_samples, logits.shape[-1]), dtype=torch.float32)
            B = logits.shape[0]
            out[cursor:cursor + B] = logits
            cursor += B
    if cursor != num_samples:
        raise RuntimeError(f"teacher precompute saw {cursor} samples, expected {num_samples}")
    return out


def load_or_compute_logits(
    teacher: nn.Module,
    fragments,
    cache_path: Path,
    device: str,
    batch_size: int = 512,
) -> torch.Tensor:
    """In-memory variant. Reads cache or recomputes via precompute_logits."""
    if cache_path.exists():
        logits = torch.load(cache_path, map_location="cpu", weights_only=True)
        if logits.shape[0] == len(fragments):
            print(f"[teacher] cache hit: {cache_path} (shape {tuple(logits.shape)})")
            return logits
        print(f"[teacher] cache shape mismatch ({logits.shape[0]} vs {len(fragments)}) — recomputing")
    logits = precompute_logits(teacher, fragments, device, batch_size=batch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, cache_path)
    print(f"[teacher] cache saved: {cache_path} (shape {tuple(logits.shape)})")
    return logits


def load_or_compute_logits_lazy(
    teacher: nn.Module,
    base_dataset,                   # LazyFragmentDataset with augment=False
    cache_path: Path,
    device: str,
    batch_size: int = 512,
    num_workers: int = 4,
) -> torch.Tensor:
    """Lazy variant. base_dataset is iterated sequentially (no shuffle, no aug)."""
    N = len(base_dataset)
    if cache_path.exists():
        logits = torch.load(cache_path, map_location="cpu", weights_only=True)
        if logits.shape[0] == N:
            print(f"[teacher] cache hit: {cache_path} (shape {tuple(logits.shape)})")
            return logits
        print(f"[teacher] cache shape mismatch ({logits.shape[0]} vs {N}) — recomputing")

    from torch.utils.data import DataLoader
    loader = DataLoader(
        base_dataset,
        batch_size = batch_size,
        shuffle    = False,
        num_workers= num_workers,
        pin_memory = device.startswith("cuda"),
        persistent_workers = num_workers > 0,
    )
    logits = precompute_logits_from_loader(teacher, loader, N, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, cache_path)
    print(f"[teacher] cache saved: {cache_path} (shape {tuple(logits.shape)})")
    return logits


def data_fingerprint(fragments, labels: list[str]) -> str:
    """Cheap content-derived id for the cache key: covers N + a few bytes from
    the start and end of the fragment array. Doesn't need to be cryptographic
    — just stable across runs with the same subset."""
    import hashlib
    h = hashlib.md5()
    h.update(str(len(fragments)).encode())
    if len(fragments) > 0:
        h.update(bytes(fragments[0][:32]))
        h.update(bytes(fragments[-1][-32:]))
        h.update(labels[0].encode())
        h.update(labels[-1].encode())
    return h.hexdigest()[:10]


def lazy_fingerprint(
    frag_path: Path,
    file_indices,
    labels: list[str],
) -> str:
    """Fingerprint for the lazy (memmap) path. Uses file path + size + a few
    indices/labels — doesn't touch the fragment bytes themselves."""
    import hashlib
    import os
    h = hashlib.md5()
    h.update(str(frag_path).encode())
    try:
        h.update(str(os.path.getsize(frag_path)).encode())
    except OSError:
        pass
    h.update(str(len(file_indices)).encode())
    if len(file_indices) > 0:
        h.update(file_indices[:5].tobytes())
        h.update(file_indices[-5:].tobytes())
    if labels:
        h.update(labels[0].encode())
        h.update(labels[-1].encode())
    return h.hexdigest()[:10]
