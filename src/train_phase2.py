"""
Phase 2 Specialist Training — one specialist per group, crash-safe.

Each specialist trains only on its group's data and predicts the fine-grained
file type within that group (local class index within the group, NOT the global
75-class index).

Archive layout:
    phase2_archive/
        {group_name}/
            epoch_{N:04d}.pt
            best.pt -> epoch_{N:04d}.pt   (symlink, updated on val_acc improvement)
            training_log.json

Encoder initialisation (--encoder-init):
    random      fresh FusedEncoder; train encoder + head together
    from-phase1 warm-start encoder from phase1 best.pt; train encoder + head
    frozen      load phase1 encoder weights and freeze them; train head only

Usage:
    python train_phase2.py [options]

    --phase1-checkpoint PATH  phase1 best.pt to warm-start / freeze encoder
    --binary-dir PATH         binary split directory (default: BINARY_DIR)
    --group NAME              train only this one group (default: all groups)
    --encoder-init MODE       random | from-phase1 | frozen  (default: from-phase1)
    --max-per-class N         hard cap on samples per fine-grained class
    --fraction F              proportional cut, e.g. 0.1 = 10 %% of each class
    --epochs N                epochs per specialist (default: 50)
    --batch-size N            batch size (default: 256)
    --lr LR                   learning rate (default: 1e-3)
    --min-lr LR               cosine annealing lower bound (default: 1e-5)
    --warmup-pct F            fraction of total steps for linear warmup (default: 0.05)
    --weight-decay F          AdamW weight decay (default: 0.01)
    --label-smoothing F       cross-entropy label smoothing (default: 0.1)
    --grad-accum N            gradient accumulation steps (default: 1)
    --grad-checkpoint         recompute encoder activations in backward (~60%% VRAM)
    --compile                 torch.compile for 2-3x CUDA throughput (PyTorch >= 2.0)
    --resume                  resume each specialist from its latest checkpoint
    --archive-dir PATH        archive root (default: ./phase2_archive)
    --gpu-mem-fraction F      max fraction of GPU memory, e.g. 0.9
    --cpu-fraction F          fraction of CPU cores to use, e.g. 0.5
"""

import json
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from load_binary import load_split, label_indices_to_strings, BINARY_DIR
from hierarchical_cascade import (
    FusedEncoder,
    ArchiveEncoder,
    TextEncoder,
    SpecialistClassifier,
    FragmentDataset,
    TYPE_TO_GROUP,
    GROUP_LOCAL_IDX,
    GROUPS,
    GROUP_NAMES,
    make_optimizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _latest_checkpoint(archive_dir: Path) -> Optional[Path]:
    ckpts = sorted(archive_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _epoch_from_path(p: Path) -> int:
    return int(p.stem.split("_")[1])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_group_data(
    split: str,
    group: str,
    max_per_class: Optional[int],
    fraction: Optional[float],
    binary_dir: Path,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Load split, filter to one group, optionally subsample per fine-grained class."""
    # mmap=True: the fragment file is memory-mapped rather than read entirely into
    # RAM.  Only the pages for the rows in `keep` are faulted in from disk, so
    # loading is proportional to the number of samples kept, not to the total
    # dataset size.  This makes smoke-test runs (--max-per-class 500) load in
    # seconds instead of waiting for a multi-GB file to copy into RAM.
    fragments, label_indices, all_types = load_split(split, mmap=True, binary_dir=binary_dir)
    labels = label_indices_to_strings(label_indices, all_types)

    # Filter to this group's fine-grained classes
    keep = [i for i, lbl in enumerate(labels) if TYPE_TO_GROUP.get(lbl) == group]
    if not keep:
        return np.empty((0, fragments.shape[1]), dtype=fragments.dtype), []
    fragments = fragments[keep]
    labels    = [labels[i] for i in keep]

    if max_per_class is None and fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        local_map   = GROUP_LOCAL_IDX[group]
        local_idxs  = np.array([local_map[lbl] for lbl in labels])
        counts      = np.bincount(local_idxs)
        min_count   = int(counts[counts > 0].min())
        max_per_class = max(1, round(min_count * fraction))

    if max_per_class is not None:
        local_map  = GROUP_LOCAL_IDX[group]
        local_idxs = np.array([local_map[lbl] for lbl in labels])
        rng  = np.random.default_rng(seed)
        keep2: list[int] = []
        for cls_idx in np.unique(local_idxs):
            idx = np.where(local_idxs == cls_idx)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep2.extend(idx.tolist())
        keep_arr  = np.array(sorted(keep2))
        fragments = fragments[keep_arr]
        labels    = [labels[i] for i in keep_arr]

    return fragments, labels


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: SpecialistClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pbar: tqdm,
    scaler=None,
    scheduler=None,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
    label_smoothing: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total_samples = 0
    running_loss = 0.0
    device_type  = device.split(":")[0]
    n_batches    = len(loader)

    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        is_last   = (step == n_batches)
        do_update = (step % accum_steps == 0) or is_last

        # Byte CutMix: splice a contiguous region from a shuffled copy of the
        # batch into x, mix labels proportionally.
        #
        # Why bytes and not embeddings: byte sequences are integers so linear
        # interpolation is meaningless. Splicing a contiguous region is valid
        # because compressed / structured files have region-level statistics
        # (e.g. a GZ header block) — swapping a region teaches the model that
        # ambiguous fragments deserve uncertain predictions.
        #
        # Applied with 50% probability per batch so half the steps still see
        # clean examples, preserving the model's ability to learn from clear signal.
        y_mix, lam_eff = y, 1.0
        if cutmix_alpha > 0 and random.random() < 0.5:
            B, L     = x.shape
            lam      = np.random.beta(cutmix_alpha, cutmix_alpha)
            cut_size = int(L * (1 - lam))
            if cut_size > 0:
                start   = random.randint(0, L - cut_size)
                j       = torch.randperm(B, device=device)
                x       = x.clone()
                x[:, start:start + cut_size] = x[j, start:start + cut_size]
                y_mix   = y[j]
                lam_eff = 1.0 - cut_size / L   # actual λ after integer truncation

        with torch.autocast(device_type=device_type, enabled=(scaler is not None)):
            logits = model(x)
            if lam_eff < 1.0:
                loss = (
                    lam_eff       * F.cross_entropy(logits, y,    label_smoothing=label_smoothing) +
                    (1 - lam_eff) * F.cross_entropy(logits, y_mix, label_smoothing=label_smoothing)
                ) / accum_steps
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if do_update:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        batch_loss     = loss.item() * accum_steps
        total_loss    += batch_loss * len(x)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += len(x)
        running_loss  += batch_loss

        pbar.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{total_correct / total_samples:.3f}",
            lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
            refresh=False,
        )
        pbar.update(1)

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def eval_one_epoch(
    model: SpecialistClassifier,
    loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_samples = 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = F.cross_entropy(logits, y, reduction="sum")
        total_loss    += loss.item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += len(x)
    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Encoder initialisation from phase1 checkpoint
# ---------------------------------------------------------------------------

def _make_encoder(group: str, grad_checkpoint: bool = False) -> torch.nn.Module:
    """Return the appropriate encoder for a specialist group."""
    if group == "archive":
        return ArchiveEncoder(grad_checkpoint=grad_checkpoint)
    if group == "text":
        return TextEncoder(grad_checkpoint=grad_checkpoint)
    return FusedEncoder(grad_checkpoint=grad_checkpoint)


def _load_encoder_from_phase1(encoder: torch.nn.Module, ckpt_path: Path, device: str) -> None:
    """
    Extract ByteEncoder weights from a phase1 CoarseClassifier checkpoint and load
    them into FusedEncoder.byte_enc.  FusedEncoder.bigram_enc has no phase1 equivalent
    and is left at its random initialisation.

    Key path translation:
        phase1 CoarseClassifier  →  FusedEncoder
        encoder.<key>            →  byte_enc.<key>
    """
    state    = torch.load(ckpt_path, map_location=device, weights_only=True)
    model_sd = state["model"]
    # torch.compile wraps keys with "_orig_mod."
    if any(k.startswith("_orig_mod.") for k in model_sd):
        model_sd = {k.removeprefix("_orig_mod."): v for k, v in model_sd.items()}

    # Remap encoder.* → byte_enc.* (ByteEncoder lives at FusedEncoder.byte_enc)
    encoder_sd = {
        "byte_enc." + k.removeprefix("encoder."): v
        for k, v in model_sd.items()
        if k.startswith("encoder.")
    }

    # strict=False: only byte_enc.* is expected to be in the phase1 checkpoint.
    # All other branches (bigram_enc, header_mlp, hist_mlp, struct_mlp, …) are new
    # and will be missing — that is correct, they are randomly initialised.
    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    unexpected       = [k for k in unexpected if not k.startswith("byte_enc.")]
    byte_enc_missing = [k for k in missing    if k.startswith("byte_enc.")]
    if byte_enc_missing or unexpected:
        raise RuntimeError(
            f"Encoder load mismatch — missing byte_enc keys: {byte_enc_missing}, "
            f"unexpected: {unexpected}"
        )
    n_new = len(missing)   # everything missing is a new randomly-init'd branch
    print(f"  Loaded byte_enc from phase1; {n_new} new-branch tensors randomly initialised")


# ---------------------------------------------------------------------------
# Per-specialist training loop
# ---------------------------------------------------------------------------

def train_specialist(
    group:           str,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    epochs:          int,
    lr:              float,
    min_lr:          float,
    device:          str,
    archive_dir:     Path,
    phase1_ckpt:     Optional[Path],
    encoder_init:    str,           # "random" | "from-phase1" | "frozen"
    resume:          bool  = False,
    weight_decay:    float = 0.01,
    accum_steps:     int   = 1,
    grad_checkpoint: bool  = False,
    label_smoothing: float = 0.1,
    warmup_pct:      float = 0.05,
    compile_model:   bool  = False,
    cutmix_alpha:    float = 0.0,
    extra_config:    dict  = {},
) -> SpecialistClassifier:

    archive_dir.mkdir(parents=True, exist_ok=True)
    log_path    = archive_dir / "training_log.json"
    num_classes = len(GROUPS[group])

    encoder = _make_encoder(group, grad_checkpoint=grad_checkpoint)

    if encoder_init in ("from-phase1", "frozen"):
        if phase1_ckpt is None:
            raise ValueError(f"--phase1-checkpoint required for --encoder-init {encoder_init}")
        _load_encoder_from_phase1(encoder, phase1_ckpt, device)
        print(f"  [{group}] encoder loaded from {phase1_ckpt.name}")

    model = SpecialistClassifier(encoder, num_classes).to(device)

    if encoder_init == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print(f"  [{group}] encoder frozen — training head only")

    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    # encoder_lr_scale=0.1: pretrained encoder updates at lr/10 so it doesn't
    # drift too fast while the randomly-initialised head catches up.
    # Has no effect when encoder is frozen (those params have requires_grad=False).
    encoder_lr_scale = 0.1 if encoder_init != "frozen" else 1.0
    optimizer = make_optimizer(
        model,
        lr               = lr,
        weight_decay     = weight_decay,
        encoder_lr_scale = encoder_lr_scale,
    )

    opt_steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_steps   = opt_steps_per_epoch * epochs
    warmup_steps  = max(1, round(total_steps * warmup_pct))
    cosine_steps  = max(1, total_steps - warmup_steps)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-4,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cosine_steps,
        eta_min = min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup_sched, cosine_sched],
        milestones = [warmup_steps],
    )

    # --- Resume ---
    start_epoch = 0
    if resume:
        ckpt = _latest_checkpoint(archive_dir)
        if ckpt is not None:
            start_epoch = _epoch_from_path(ckpt)
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model_sd = state["model"]
            if any(k.startswith("_orig_mod.") for k in model_sd):
                model_sd = {k.removeprefix("_orig_mod."): v for k, v in model_sd.items()}
            model.load_state_dict(model_sd)
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            print(f"  [{group}] resumed from {ckpt.name}  (epoch {start_epoch})")
        else:
            print(f"  [{group}] no checkpoint found — starting fresh")

    if compile_model:
        model = torch.compile(model)

    # --- Init or load JSON log ---
    if log_path.exists() and resume:
        log = json.loads(log_path.read_text())
    else:
        log = {
            "session_id":  datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at":  datetime.now().isoformat(),
            "group":       group,
            "num_classes": num_classes,
            "config": {
                "epochs":          epochs,
                "lr":              lr,
                "min_lr":          min_lr,
                "weight_decay":    weight_decay,
                "betas":           [0.9, 0.999],
                "label_smoothing": label_smoothing,
                "cutmix_alpha":    cutmix_alpha,
                "warmup_pct":      warmup_pct,
                "optimizer":       "AdamW",
                "scheduler":       "LinearWarmup+CosineAnnealing",
                "encoder_init":    encoder_init,
                "encoder_lr_scale": encoder_lr_scale,
                "amp":             scaler is not None,
                "compile":         compile_model,
                "grad_checkpoint": grad_checkpoint,
                "accum_steps":     accum_steps,
                "effective_batch": train_loader.batch_size * accum_steps,
                "device":          device,
                "batch_size":      train_loader.batch_size,
                **extra_config,
            },
            "status": "in_progress",
            "best": {"epoch": None, "val_acc": None, "val_loss": None},
            "epochs": [],
        }
        _atomic_write_json(log_path, log)

    best_val_acc = log["best"]["val_acc"] or 0.0
    num_batches  = len(train_loader)
    epoch_pad    = len(str(epochs))
    spec_start   = time.time()

    for epoch in range(start_epoch, epochs):
        ep_num   = epoch + 1
        ep_start = time.time()

        desc = f"  [{group}] {ep_num:>{epoch_pad}}/{epochs} [train]"
        with tqdm(total=num_batches, desc=desc, unit="batch",
                  bar_format="{l_bar}{bar:28}{r_bar}") as pbar:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, pbar,
                scaler=scaler, scheduler=scheduler, accum_steps=accum_steps,
                label_smoothing=label_smoothing, cutmix_alpha=cutmix_alpha,
            )

        val_loss, val_acc = eval_one_epoch(model, val_loader, device)
        elapsed = time.time() - ep_start
        print(
            f"  [{group}] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"  ({_fmt_time(elapsed)})"
        )

        ckpt_path = archive_dir / f"epoch_{ep_num:04d}.pt"
        ckpt_data = {
            "epoch":      ep_num,
            "group":      group,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }
        if scaler is not None:
            ckpt_data["scaler"] = scaler.state_dict()
        torch.save(ckpt_data, ckpt_path)

        epoch_record = {
            "epoch":      ep_num,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         round(optimizer.param_groups[-1]["lr"], 8),
            "elapsed_s":  round(elapsed, 2),
            "timestamp":  datetime.now().isoformat(),
            "checkpoint": ckpt_path.name,
        }
        log["epochs"].append(epoch_record)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            log["best"] = {
                "epoch":    ep_num,
                "val_acc":  round(val_acc,  6),
                "val_loss": round(val_loss, 6),
            }
            best_path = archive_dir / "best.pt"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(ckpt_path.name)

        _atomic_write_json(log_path, log)

    log["status"]      = "complete"
    log["finished_at"] = datetime.now().isoformat()
    log["total_time"]  = _fmt_time(time.time() - spec_start)
    _atomic_write_json(log_path, log)

    print(
        f"  [{group}] done in {log['total_time']}"
        f"  best val_acc={log['best']['val_acc']:.4f}  (epoch {log['best']['epoch']})"
    )
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Phase 2 specialists with per-epoch checkpointing"
    )
    parser.add_argument("--phase1-checkpoint", type=Path, default=None,
                        help="Path to phase1 best.pt (required unless --encoder-init random)")
    parser.add_argument("--binary-dir",    type=Path,  default=BINARY_DIR)
    parser.add_argument("--group",         type=str,   default=None,
                        choices=GROUP_NAMES,
                        help="Train only this group (default: all groups)")
    parser.add_argument("--encoder-init",  type=str,   default="from-phase1",
                        choices=["random", "from-phase1", "frozen"],
                        help="How to initialise the encoder for each specialist")
    parser.add_argument("--max-per-class", type=int,   default=None)
    parser.add_argument("--fraction",      type=float, default=None)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--min-lr",        type=float, default=1e-5,
                        help="Cosine annealing lower bound (default: 1e-5)")
    parser.add_argument("--warmup-pct",    type=float, default=0.05)
    parser.add_argument("--weight-decay",  type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--noise-prob",    type=float, default=0.02,
                        help="Fraction of bytes to randomly corrupt per training sample (default: 0.02)")
    parser.add_argument("--cutmix-alpha",  type=float, default=0.4,
                        help="Beta distribution alpha for byte CutMix; 0 to disable (default: 0.4)")
    parser.add_argument("--grad-accum",    type=int,   default=1)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--compile",       action="store_true")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume each specialist from its latest checkpoint")
    parser.add_argument("--archive-dir",   type=Path,
                        default=Path(__file__).parent / "phase2_archive",
                        help="Archive root; each group gets a sub-folder (default: ./phase2_archive)")
    parser.add_argument("--gpu-mem-fraction", type=float, default=None)
    parser.add_argument("--cpu-fraction",     type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.gpu_mem_fraction is not None and device.startswith("cuda"):
        if not 0.0 < args.gpu_mem_fraction <= 1.0:
            raise ValueError(f"--gpu-mem-fraction must be in (0, 1], got {args.gpu_mem_fraction}")
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        print(f"GPU memory cap: {args.gpu_mem_fraction * 100:.0f}%")

    if args.cpu_fraction is not None:
        import os
        if not 0.0 < args.cpu_fraction <= 1.0:
            raise ValueError(f"--cpu-fraction must be in (0, 1], got {args.cpu_fraction}")
        n_threads = max(1, int(os.cpu_count() * args.cpu_fraction))
        torch.set_num_threads(n_threads)
        print(f"CPU threads: {n_threads}/{os.cpu_count()} ({args.cpu_fraction * 100:.0f}%)")

    groups_to_train = [args.group] if args.group else GROUP_NAMES
    print(f"Groups to train: {groups_to_train}")
    print(f"Encoder init:    {args.encoder_init}")
    print(f"Archive root:    {args.archive_dir}\n")

    on_cuda = device.startswith("cuda")
    dl_kwargs = dict(
        batch_size         = args.batch_size,
        num_workers        = 4,
        pin_memory         = on_cuda,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    total_start = time.time()

    for group in groups_to_train:
        print(f"\n{'='*60}")
        print(f"  Group: {group}  ({len(GROUPS[group])} classes: {', '.join(GROUPS[group])})")
        print(f"{'='*60}")

        print(f"  Loading train split for [{group}] ...")
        train_frags, train_labels = load_group_data(
            "train", group, args.max_per_class, args.fraction, args.binary_dir
        )
        if len(train_labels) == 0:
            print(f"  [{group}] no training samples — skipping")
            continue
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print(f"  Loading val split for [{group}] ...")
        val_frags, val_labels = load_group_data(
            "val", group, args.max_per_class, args.fraction, args.binary_dir
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        train_ds = FragmentDataset(train_frags, train_labels, mode=f"specialist:{group}",
                                   augment=True, noise_prob=args.noise_prob)
        val_ds   = FragmentDataset(val_frags,   val_labels,   mode=f"specialist:{group}",
                                   augment=False)

        train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
        val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

        group_archive = args.archive_dir / group

        train_specialist(
            group            = group,
            train_loader     = train_loader,
            val_loader       = val_loader,
            epochs           = args.epochs,
            lr               = args.lr,
            min_lr           = args.min_lr,
            device           = device,
            archive_dir      = group_archive,
            phase1_ckpt      = args.phase1_checkpoint,
            encoder_init     = args.encoder_init,
            resume           = args.resume,
            weight_decay     = args.weight_decay,
            accum_steps      = args.grad_accum,
            grad_checkpoint  = args.grad_checkpoint,
            label_smoothing  = args.label_smoothing,
            warmup_pct       = args.warmup_pct,
            compile_model    = args.compile,
            cutmix_alpha     = args.cutmix_alpha,
            extra_config     = {
                "group":            group,
                "max_per_class":    args.max_per_class,
                "fraction":         args.fraction,
                "binary_dir":       str(args.binary_dir),
                "gpu_mem_fraction": args.gpu_mem_fraction,
                "cpu_fraction":     args.cpu_fraction,
            },
        )

    total_elapsed = time.time() - total_start
    print(f"\nAll specialists done in {_fmt_time(total_elapsed)}")
    print(f"Archive root: {args.archive_dir}")
