"""
Multi-fragment voting smoke test.

Hypothesis: a single 4096-byte fragment is often ambiguous (text vs archive in
particular), but a *file* produces many fragments — averaging the Phase-1
model's predictions across a file's fragments should beat any single fragment.

This probes that without needing per-file metadata. It runs the trained Phase-1
model over the val split to get per-fragment group probabilities, then:

  1. Run-length check — are fragments from one file stored consecutively? If so,
     each same-label run is treated as a real file and voted directly (the
     honest number). If shuffled, real grouping would need a dataset regen.

  2. Voting curve (simulation) — simulates "files" by partitioning each group's
     fragments into chunks of K, mean-averaging their predicted probabilities,
     and scoring the voted argmax. Reports voted accuracy as K grows.

Caveat: simulated chunks draw fragments from *different* files, so their errors
are more independent than fragments of one real file would be. The simulated
curve is therefore an optimistic upper bound — flat here means real voting
won't help; steep here means it is worth building with real per-file grouping.

Usage:
    PYTHONPATH=src python multifrag/smoke_vote_sim.py \
        --binary-dir data/4k_1/binary \
        --checkpoint checkpoints/phase1_archive/best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Phase-1 model code lives in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hierarchical_cascade import (  # noqa: E402
    CoarseEncoder, CoarseClassifier, GROUPS, TYPE_TO_GROUP, GROUP_TO_IDX,
)

GROUP_NAMES = list(GROUPS.keys())
VOTE_KS = [1, 2, 4, 8, 16, 32, 64]


def _clean_state_dict(sd: dict) -> dict:
    """Strip AveragedModel ('module.') and torch.compile ('_orig_mod.') prefixes."""
    out = {}
    for k, v in sd.items():
        if k == "n_averaged":          # AveragedModel bookkeeping buffer
            continue
        k = k.removeprefix("module.").removeprefix("_orig_mod.")
        out[k] = v
    return out


def load_model(ckpt_path: Path, weights: str, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    src = state.get("best_source", "model") if weights == "auto" else weights
    sd = state["ema"] if (src == "ema" and "ema" in state) else state["model"]
    model = CoarseClassifier(CoarseEncoder()).to(device)
    model.load_state_dict(_clean_state_dict(sd))
    model.eval()
    return model, src


def load_val(binary_dir: Path, max_samples):
    meta = json.loads((binary_dir / "val_meta.json").read_text())
    n, sector, all_types = meta["n_samples"], meta["sector_size"], meta["all_types"]
    frags = np.memmap(binary_dir / "val_fragments.bin", dtype=np.uint8,
                      mode="r", shape=(n, sector))
    labels = np.fromfile(binary_dir / "val_labels.bin", dtype=np.uint8)
    if max_samples is not None and max_samples < n:
        frags, labels = frags[:max_samples], labels[:max_samples]
    # True coarse-group index per fragment.
    group_idx = np.array(
        [GROUP_TO_IDX[TYPE_TO_GROUP[all_types[l]]] for l in labels], dtype=np.int64
    )
    return frags, np.asarray(labels), group_idx


@torch.inference_mode()
def predict_probs(model, frags, device, batch_size=512):
    n = len(frags)
    probs = np.zeros((n, len(GROUP_NAMES)), dtype=np.float32)
    for s in range(0, n, batch_size):
        x = torch.from_numpy(np.asarray(frags[s:s + batch_size]).astype(np.int64)).to(device)
        probs[s:s + batch_size] = torch.softmax(model(x), dim=-1).float().cpu().numpy()
    return probs


def real_run_voting(probs, labels, group_idx):
    """Vote within consecutive same-(type)label runs — honest only if file-ordered."""
    changes = np.flatnonzero(np.diff(labels.astype(np.int16))) + 1
    bounds  = np.concatenate([[0], changes, [len(labels)]])
    runs    = np.diff(bounds)
    correct = total = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        voted = int(probs[a:b].mean(axis=0).argmax())
        correct += int(voted == group_idx[a])
        total   += 1
    return float(runs.mean()), int(runs.max()), correct / max(total, 1), total


def voting_curve(probs, group_idx, rng):
    """Partition each group's fragments into K-chunks, mean-vote, score. (overall, per-group)"""
    by_group = {g: rng.permutation(np.flatnonzero(group_idx == g))
                for g in range(len(GROUP_NAMES))}
    overall, per_group = {}, {}
    for K in VOTE_KS:
        g_acc = {}
        tot_correct = tot_files = 0
        for g, idxs in by_group.items():
            n_chunks = len(idxs) // K
            if n_chunks == 0:
                g_acc[g] = float("nan")
                continue
            chunks = idxs[:n_chunks * K].reshape(n_chunks, K)
            voted  = probs[chunks].mean(axis=1).argmax(axis=1)   # (n_chunks,)
            correct = int((voted == g).sum())
            g_acc[g] = correct / n_chunks
            tot_correct += correct
            tot_files   += n_chunks
        overall[K]   = tot_correct / max(tot_files, 1)
        per_group[K] = g_acc
    return overall, per_group


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-fragment voting smoke test")
    ap.add_argument("--binary-dir", type=Path, default=Path("data/4k_1/binary"))
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Phase-1 checkpoint (e.g. checkpoints/phase1_archive/best.pt)")
    ap.add_argument("--weights", choices=["auto", "model", "ema"], default="auto",
                    help="Which weights to load (default: auto = checkpoint's best_source)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap val fragments scanned (default: all)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    model, src = load_model(args.checkpoint, args.weights, device)
    print(f"Loaded {args.checkpoint.name}  (weights: {src}, device: {device})")

    frags, labels, group_idx = load_val(args.binary_dir, args.max_samples)
    print(f"Running Phase-1 inference on {len(frags)} val fragments ...")
    t0 = time.time()
    probs = predict_probs(model, frags, device, args.batch_size)
    print(f"  inference done in {time.time() - t0:.1f}s\n")

    # --- 1. Run-length + real-run voting ---
    mean_run, max_run, real_acc, n_runs = real_run_voting(probs, labels, group_idx)
    per_frag = float((probs.argmax(axis=1) == group_idx).mean())
    print("=== 1. Data layout + real-run voting ===")
    print(f"  per-fragment accuracy        : {per_frag:.4f}")
    print(f"  mean / max same-label run    : {mean_run:.2f} / {max_run}")
    if mean_run >= 2.0:
        print(f"  real-run voted accuracy      : {real_acc:.4f}  ({n_runs} runs)")
        print("  -> data is file-ordered: this is an honest voting number.")
    else:
        print("  -> data looks shuffled: real grouping needs a per-fragment file-id.")

    # --- 2. Simulated voting curve ---
    overall, per_group = voting_curve(probs, group_idx, rng)
    print("\n=== 2. Simulated voting curve (optimistic upper bound) ===")
    print(f"  {'K':>4}  {'voted_acc':>10}")
    for K in VOTE_KS:
        print(f"  {K:>4}  {overall[K]:>10.4f}")

    kmax = VOTE_KS[-1]
    print(f"\n  per-group accuracy  (K=1  ->  K={kmax}):")
    print(f"  {'group':<14}{'K=1':>9}{f'K={kmax}':>9}")
    for g, name in enumerate(GROUP_NAMES):
        print(f"  {name:<14}{per_group[1][g]:>9.4f}{per_group[kmax][g]:>9.4f}")

    print(f"\nTotal time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
