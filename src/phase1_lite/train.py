"""
Phase 1 lite training loop.

Defaults are tuned for the user's "fast smoke test" target: 25% per class,
10 epochs, batch 256, AMP on if CUDA is available. Per-epoch wall-clock at
that setting on a single A100 should be ~3-5 minutes.

Pipeline per step:
  1. forward through LiteCoarseClassifier
  2. (optional) forward through frozen teacher
  3. loss = CE_w(confusion-penalty) + alpha * KD(student, teacher)

The confusion-penalty cost matrix is loaded from either the original phase1
training_log.json or a saved confusion_matrix.json (see --confusion-source).
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python src/phase1_lite/train.py` to find sibling modules.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR  = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from load_binary import load_split, label_indices_to_strings, BINARY_DIR
from hierarchical_cascade import (
    FragmentDataset, GROUP_NAMES, GROUP_TO_IDX, NUM_GROUPS, TYPE_TO_GROUP,
)

from phase1_lite.model    import LiteCoarseClassifier, count_parameters
from phase1_lite.losses   import (
    ConfusionWeightedCE, cost_matrix_from_log, cost_matrix_from_array,
    distillation_kl, build_label_smoothing_vector,
)
from phase1_lite.teacher  import (
    load_teacher, load_or_compute_logits, load_or_compute_logits_lazy,
    data_fingerprint, lazy_fingerprint, _cache_path,
)
from phase1_lite.augment  import AugmentedDataset


# ---------------------------------------------------------------------------
# Data helpers (mirrors the heavy train_phase1.py; kept short)
# ---------------------------------------------------------------------------

def load_data(
    split:         str,
    max_per_class: Optional[int],
    fraction:      Optional[float],
    binary_dir:    Path,
    seed:          int = 42,
) -> tuple[np.ndarray, list[str]]:
    fragments, label_indices, all_types = load_split(split, binary_dir=binary_dir)
    labels = label_indices_to_strings(label_indices, all_types)

    if max_per_class is None and fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        counts    = np.bincount(label_indices)
        min_count = int(counts[counts > 0].min())
        max_per_class = max(1, round(min_count * fraction))

    if max_per_class is not None:
        rng  = np.random.default_rng(seed)
        keep: list[int] = []
        for cls_idx in np.unique(label_indices):
            idx = np.where(label_indices == cls_idx)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep.extend(idx.tolist())
        keep_arr  = np.array(sorted(keep))
        fragments = fragments[keep_arr]
        labels    = [labels[i] for i in keep_arr]

    return fragments, labels


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float, np.ndarray]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    conf = np.zeros((NUM_GROUPS, NUM_GROUPS), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.numel()
        for t, p in zip(y.cpu().tolist(), preds.cpu().tolist()):
            conf[t, p] += 1
    return loss_sum / total, correct / total, conf


def format_confusion(conf: np.ndarray, class_names: list[str]) -> str:
    n     = len(class_names)
    lbl_w = max(len(c) for c in class_names)
    cell  = max(lbl_w, 6)
    row_t = conf.sum(axis=1).clip(1)
    col_t = conf.sum(axis=0).clip(1)
    pct   = conf / row_t[:, None] * 100

    def fmt(i, j):
        v = pct[i, j]
        if i == j: return f"[{v:3.0f}%]".center(cell)
        if v >= 1: return f"{v:3.0f}% ".rjust(cell)
        return " " * cell

    sep    = "-" * (lbl_w + 2 + n * (cell + 1))
    header = " " * (lbl_w + 2) + " ".join(c.center(cell) for c in class_names)
    lines  = [header, sep]
    for i, name in enumerate(class_names):
        lines.append(f"{name:>{lbl_w}}  " + " ".join(fmt(i, j) for j in range(n)))
    lines.append(sep)
    prec = np.diag(conf) / col_t * 100
    lines.append(f"{'prec':>{lbl_w}}  " + " ".join(f"{p:3.0f}% ".rjust(cell) for p in prec))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- data ----
    # Lazy mode (default for full data): only labels live in RAM; fragments are
    # read on demand via per-worker file handles. Required for the 25 GB train
    # split — eager loading OOM-kills on commodity hardware.
    if args.lazy:
        from load_binary_lazy import LazyFragmentDataset, load_split_lazy

        print(f"Loading train split (lazy) from {args.binary_dir} ...")
        train_frag_path, train_file_idx, train_labels, sector = load_split_lazy(
            split=        "train",
            max_per_class=args.max_per_class,
            fraction=     args.fraction,
            binary_dir=   args.binary_dir,
        )
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print("Loading val split (lazy) ...")
        val_frag_path, val_file_idx, val_labels, _ = load_split_lazy(
            split=        "val",
            max_per_class=args.max_per_class,
            fraction=     args.fraction,
            binary_dir=   args.binary_dir,
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        # AugmentedDataset wraps a base dataset that has its own aug disabled.
        train_ds_base = LazyFragmentDataset(
            train_frag_path, sector, train_file_idx, train_labels,
            mode="coarse", augment=False,
        )
        val_ds_base = LazyFragmentDataset(
            val_frag_path, sector, val_file_idx, val_labels,
            mode="coarse", augment=False,
        )
        # In-memory `fragments` array does NOT exist in lazy mode — we set this
        # placeholder so the teacher-cache branch below picks the lazy path.
        train_frags = None
    else:
        print(f"Loading train split from {args.binary_dir} ...")
        train_frags, train_labels = load_data(
            "train", args.max_per_class, args.fraction, args.binary_dir
        )
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print("Loading val split ...")
        val_frags, val_labels = load_data(
            "val", args.max_per_class, args.fraction, args.binary_dir
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        train_ds_base = FragmentDataset(
            train_frags, train_labels, mode="coarse", augment=False,
        )
        val_ds_base   = FragmentDataset(
            val_frags, val_labels, mode="coarse", augment=False,
        )

    on_cuda = device.startswith("cuda")
    dl_kwargs = dict(
        batch_size         = args.batch_size,
        num_workers        = args.workers,
        pin_memory         = on_cuda,
        persistent_workers = args.workers > 0,
        prefetch_factor    = 2 if args.workers > 0 else None,
    )

    # ---- model ----
    model = LiteCoarseClassifier().to(device)
    print(f"LiteCoarseClassifier params: {count_parameters(model):,}")

    # ---- teacher (optional) + cached logits ----
    # Pre-computing the teacher's logits over the whole train array once and
    # streaming them from RAM (or disk) cuts per-step cost from "teacher
    # forward" to "tensor index" — typically a 5-20x speedup vs. running the
    # teacher live every batch.
    teacher = None
    train_logits = None
    if args.teacher:
        print(f"Loading teacher checkpoint: {args.teacher}")
        teacher = load_teacher(args.teacher, device=device)
        print(f"  teacher loaded ({count_parameters(teacher):,} params, frozen)")

        if args.lazy:
            fp = lazy_fingerprint(train_frag_path, train_file_idx, train_labels)
            cache = _cache_path(args.archive_dir, "train", fp)
            train_logits = load_or_compute_logits_lazy(
                teacher, train_ds_base, cache, device,
                batch_size=max(args.batch_size, 512),
                num_workers=args.workers,
            )
        else:
            fp = data_fingerprint(train_frags, train_labels)
            cache = _cache_path(args.archive_dir, "train", fp)
            train_logits = load_or_compute_logits(
                teacher, train_frags, cache, device,
                batch_size=max(args.batch_size, 512),
            )
        # Move teacher off-GPU once logits are cached — frees VRAM for the student.
        teacher = teacher.cpu()
        if on_cuda:
            torch.cuda.empty_cache()

    train_ds = AugmentedDataset(
        train_ds_base,
        gbflip_sigma    = args.gbflip_sigma,
        gbflip_max_rate = args.gbflip_max_rate,
        teacher_logits  = train_logits,
        seed            = 42,
    )
    val_ds = AugmentedDataset(val_ds_base, gbflip_sigma=0.0)  # no aug on val

    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    # ---- confusion cost matrix ----
    if args.confusion_source and Path(args.confusion_source).exists():
        cost = cost_matrix_from_log(args.confusion_source, NUM_GROUPS)
        print(f"Loaded confusion cost matrix from {args.confusion_source}")
    else:
        print("No confusion source given — using uniform off-diagonal cost.")
        cost = cost_matrix_from_array(np.ones((NUM_GROUPS, NUM_GROUPS), np.float32))

    # ---- class weights (preserve disk_image down-weight; see project memory) ----
    class_weights = torch.ones(NUM_GROUPS, dtype=torch.float32)
    class_weights[GROUP_TO_IDX["disk_image"]] = args.disk_image_weight
    print(f"  cross-entropy class weights: disk_image={args.disk_image_weight}, others=1.0")

    # Per-class label smoothing: container/ambiguous classes get heavier
    # smoothing so hard targets don't force overconfident predictions on
    # genuinely-ambiguous fragments (a disk_image sector really is whatever
    # file it contains; a database page legitimately overlaps executable).
    container_overrides = {
        GROUP_TO_IDX["disk_image"]: args.container_smoothing,
        GROUP_TO_IDX["database"]:   args.container_smoothing,
    }
    smoothing_vec = build_label_smoothing_vector(
        num_classes      = NUM_GROUPS,
        default_smoothing = args.label_smoothing,
        class_overrides   = container_overrides,
    )
    print(f"  label smoothing: default={args.label_smoothing}, "
          f"disk_image/database={args.container_smoothing}")

    loss_fn = ConfusionWeightedCE(
        cost_matrix      = cost,
        confusion_lambda = args.confusion_lambda,
        class_weights    = class_weights,
        label_smoothing  = smoothing_vec,
    ).to(device)

    # ---- optimizer + sched ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, round(total_steps * args.warmup_pct))
    cosine_steps    = max(1, total_steps - warmup_steps)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=args.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps],
    )

    scaler = torch.amp.GradScaler("cuda") if on_cuda else None
    device_type = device.split(":")[0]

    # ---- archive ----
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.archive_dir / "training_log.json"
    log = {
        "session_id":   datetime.now().strftime("%Y%m%d_%H%M%S"),
        "started_at":   datetime.now().isoformat(),
        "config":       {**vars(args), "binary_dir": str(args.binary_dir),
                          "archive_dir": str(args.archive_dir),
                          "teacher": str(args.teacher) if args.teacher else None},
        "model":        "LiteCoarseClassifier",
        "params":       count_parameters(model),
        "status":       "in_progress",
        "best":         {"epoch": None, "val_acc": None, "val_loss": None},
        "epochs":       [],
    }
    log_path.write_text(json.dumps(log, indent=2, default=str))

    best_val_acc = 0.0
    total_start  = time.time()

    for epoch in range(args.epochs):
        ep = epoch + 1
        ep_start = time.time()
        model.train()
        running = correct = total = 0
        running_loss = 0.0
        running_ce = running_kd = 0.0
        n_batches = len(train_loader)

        pbar = tqdm(total=n_batches, desc=f"Epoch {ep:>2}/{args.epochs} [train]",
                    unit="batch", bar_format="{l_bar}{bar:30}{r_bar}")
        for step, batch in enumerate(train_loader, 1):
            if train_logits is not None:
                x, y, t_logits = batch
                t_logits = t_logits.to(device, non_blocking=True)
            else:
                x, y = batch
                t_logits = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, enabled=(scaler is not None)):
                logits = model(x)
                loss_ce = loss_fn(logits, y)
                if t_logits is not None:
                    loss_kd = distillation_kl(logits, t_logits, temperature=args.kd_temp)
                    loss = (1.0 - args.kd_alpha) * loss_ce + args.kd_alpha * loss_kd
                else:
                    loss_kd = torch.tensor(0.0, device=device)
                    loss = loss_ce

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                correct += (logits.argmax(-1) == y).sum().item()
                total   += y.numel()
            running_loss += loss.item()
            running_ce   += loss_ce.item()
            running_kd   += float(loss_kd.item())
            pbar.set_postfix(
                loss=f"{running_loss / step:.4f}",
                ce  =f"{running_ce   / step:.4f}",
                kd  =f"{running_kd   / step:.4f}",
                acc =f"{correct / total:.3f}",
                lr  =f"{optimizer.param_groups[-1]['lr']:.2e}",
                refresh=False,
            )
            pbar.update(1)
        pbar.close()

        train_loss = running_loss / max(1, n_batches)
        train_acc  = correct / max(1, total)

        val_loss, val_acc, conf = evaluate(model, val_loader, device)
        elapsed = time.time() - ep_start
        print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"(best={max(best_val_acc, val_acc):.4f}, {elapsed:.1f}s)")

        # save per-epoch ckpt
        ckpt_path = args.archive_dir / f"epoch_{ep:04d}.pt"
        torch.save({
            "epoch":     ep,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss":   val_loss,   "val_acc":   val_acc,
        }, ckpt_path)

        log["epochs"].append({
            "epoch":      ep,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         round(optimizer.param_groups[-1]["lr"], 8),
            "elapsed_s":  round(elapsed, 2),
            "timestamp":  datetime.now().isoformat(),
            "checkpoint": ckpt_path.name,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log["best"] = {
                "epoch": ep,
                "val_acc":  round(val_acc, 6),
                "val_loss": round(val_loss, 6),
            }
            best_path = args.archive_dir / "best.pt"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(ckpt_path.name)

            # store the matrix when we get a new best so the next run can
            # bootstrap its confusion-cost matrix from it.
            log["confusion_matrix"]         = conf.tolist()
            log["confusion_matrix_classes"] = GROUP_NAMES
            print(f"\n  val confusion matrix (row=true, col=pred):")
            for line in format_confusion(conf, GROUP_NAMES).split("\n"):
                print(f"    {line}")
            print()

        log_path.write_text(json.dumps(log, indent=2, default=str))

    log["status"]      = "complete"
    log["finished_at"] = datetime.now().isoformat()
    log["total_time_s"] = round(time.time() - total_start, 2)
    log_path.write_text(json.dumps(log, indent=2, default=str))

    print(f"\nDone. best val_acc={best_val_acc:.4f} (archive: {args.archive_dir})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1 LITE: distilled, confusion-aware coarse classifier.",
    )
    # data
    p.add_argument("--binary-dir",    type=Path,  default=BINARY_DIR)
    p.add_argument("--max-per-class", type=int,   default=None)
    p.add_argument("--fraction",      type=float, default=1.0,
                   help="proportional cut per class (default 1.0 = full data)")
    # training
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=2e-3)
    p.add_argument("--min-lr",        type=float, default=1e-5)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--warmup-pct",    type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--lazy",          action="store_true",
                   help="memmap-based loading; keeps only labels in RAM "
                        "(required for full train set on commodity machines)")
    # GBFlip augmentation — replaces the old byte-noise scheme. sigma=0 disables.
    p.add_argument("--gbflip-sigma",  type=float, default=0.01,
                   help="per-fragment flip rate ~ |N(0, sigma)|; 0 disables augmentation")
    p.add_argument("--gbflip-max-rate", type=float, default=0.05,
                   help="hard cap on the GBFlip rate")
    # Per-class label smoothing — container classes get heavier smoothing.
    p.add_argument("--container-smoothing", type=float, default=0.2,
                   help="label smoothing for disk_image/database (default 0.2)")
    # distillation
    p.add_argument("--teacher",       type=Path,  default=None,
                   help="path to a phase1 CoarseClassifier checkpoint; enables KD")
    p.add_argument("--kd-alpha",      type=float, default=0.5,
                   help="weight on KD loss (1.0 = pure distillation, 0 = no KD)")
    p.add_argument("--kd-temp",       type=float, default=4.0)
    # confusion penalty
    p.add_argument("--confusion-source", type=Path, default=None,
                   help="path to a training_log.json containing a confusion_matrix")
    p.add_argument("--confusion-lambda", type=float, default=0.5,
                   help="weight on the off-diagonal confusion penalty (0 disables)")
    # disk image down-weight (project rule, see auto-memory)
    p.add_argument("--disk-image-weight", type=float, default=0.6)
    # archive
    p.add_argument("--archive-dir",   type=Path,
                   default=Path(__file__).resolve().parent.parent.parent / "phase1_lite_archive")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
