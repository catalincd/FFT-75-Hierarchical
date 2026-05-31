"""
Phase 1 Coarse Classifier Training — standalone, crash-safe.

Trains only the coarse 11-group classifier (CoarseEncoder + CoarseClassifier).
CoarseEncoder = ByteEncoder (CNN) + BigramBranch (byte co-occurrence) +
EntropyBranch (block-wise Shannon entropy).
After every epoch it:
  - saves a model checkpoint  -> phase1_archive/epoch_{N:04d}.pt
  - atomically updates a JSON -> phase1_archive/training_log.json

This means a crash at any point leaves the archive with all completed epochs intact.

Usage:
    python train_phase1.py [options]

    --binary-dir PATH     path to binary split directory (default: BINARY_DIR)
    --max-per-class N     hard cap on samples per class
    --fraction F          proportional cut, e.g. 0.1 = 10% of each class
    --epochs N            number of training epochs (default: 50)
    --batch-size N        batch size (default: 512)
    --lr LR               learning rate (default: 1e-3)
    --lazy                read fragments on demand instead of loading into RAM
    --resume              resume from the latest checkpoint in phase1_archive/
    --archive-dir PATH    override archive directory (default: ./phase1_archive)
"""

import json
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from load_binary import load_split, label_indices_to_strings, BINARY_DIR
from hierarchical_cascade import (
    CoarseEncoder,
    CoarseClassifier,
    FragmentDataset,
    GROUP_TO_IDX,
    GROUP_NAMES,
    NUM_GROUPS,
    make_optimizer,
)
from coarse_losses import (
    ConfusionWeightedCE,
    build_label_smoothing_vector,
    cost_matrix_from_log,
    cost_matrix_from_array,
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
    """Write JSON atomically so a crash never leaves a half-written file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_data(
    split: str,
    max_per_class: Optional[int],
    fraction: Optional[float],
    binary_dir: Path,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    fragments, label_indices, all_types = load_split(split, binary_dir=binary_dir)
    labels = label_indices_to_strings(label_indices, all_types)

    if max_per_class is None and fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        counts = np.bincount(label_indices)
        min_count = int(counts[counts > 0].min())
        max_per_class = max(1, round(min_count * fraction))

    if max_per_class is not None:
        rng = np.random.default_rng(seed)
        keep: list[int] = []
        for cls_idx in np.unique(label_indices):
            idx = np.where(label_indices == cls_idx)[0]
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            keep.extend(idx.tolist())
        keep_arr = np.array(sorted(keep))
        fragments = fragments[keep_arr]
        labels    = [labels[i] for i in keep_arr]

    return fragments, labels


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: CoarseClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pbar: tqdm,
    loss_fn,                       # callable(logits, target) -> scalar; e.g. ConfusionWeightedCE
    scaler=None,                  # torch.amp.GradScaler when AMP is enabled, else None
    scheduler=None,               # OneCycleLR; stepped once per optimizer update, not per batch
    grad_clip: float = 1.0,
    accum_steps: int = 1,         # gradient accumulation: effective batch = batch_size * accum_steps
    cutmix_alpha: float = 0.0,    # Beta(alpha, alpha) byte CutMix; 0 disables
    ema_model=None,               # AveragedModel updated after each optimizer step
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
        # batch into x, mix labels proportionally. Splicing (not interpolating)
        # is the only valid mix for integer byte sequences; it teaches the model
        # that fragments straddling two formats deserve uncertain predictions.
        # Applied to 50% of batches so the rest still see clean signal.
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
            # Divide loss so accumulated gradients equal the mean over the full
            # effective batch (not the sum). Scale by actual remaining steps at
            # the last incomplete accumulation window.
            if lam_eff < 1.0:
                loss = (
                    lam_eff       * loss_fn(logits, y) +
                    (1 - lam_eff) * loss_fn(logits, y_mix)
                ) / accum_steps
            else:
                loss = loss_fn(logits, y) / accum_steps

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
            if ema_model is not None:
                ema_model.update_parameters(model)

        batch_loss     = loss.item() * accum_steps          # unscale for display
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


@torch.inference_mode()  # stricter than no_grad: disables autograd engine entirely, ~5% faster
def eval_one_epoch(
    model: CoarseClassifier,
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


@torch.inference_mode()
def eval_confusion_matrix(
    model:       CoarseClassifier,
    loader:      DataLoader,
    device:      str,
    num_classes: int,
) -> np.ndarray:
    """
    One val pass → (num_classes, num_classes) int64 confusion matrix.
    conf_mat[true_label, predicted_label] = count.
    Called only when a new best val_acc is found, so the matrix always
    corresponds to the best checkpoint without needing to reload weights.
    """
    model.eval()
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        preds = model(x).argmax(dim=-1)
        for t, p in zip(y.cpu().tolist(), preds.cpu().tolist()):
            conf_mat[t, p] += 1
    return conf_mat


def format_confusion_matrix(conf_mat: np.ndarray, class_names: list[str]) -> str:
    """
    Format a confusion matrix as a compact, readable table.

    Layout  (row = true class, col = predicted class):
      • Diagonal  [XX%]  = per-class recall
      • Off-diag   XX%   = fraction of that true class predicted as that column
      • Blank            = < 1% (suppressed to reduce noise)
      • Bottom row       = per-class precision
    """
    n       = len(class_names)
    lbl_w   = max(len(c) for c in class_names)
    cell_w  = max(lbl_w, 6)          # min 6 to fit "[100%]"

    row_totals = conf_mat.sum(axis=1).clip(1)
    col_totals = conf_mat.sum(axis=0).clip(1)
    pct        = conf_mat / row_totals[:, None] * 100   # row-normalised recall

    def cell(i: int, j: int) -> str:
        v = pct[i, j]
        if i == j:
            return f"[{v:3.0f}%]".center(cell_w)
        if v >= 1.0:
            return f"{v:3.0f}% ".rjust(cell_w)
        return " " * cell_w

    sep     = "-" * (lbl_w + 2 + n * (cell_w + 1))
    header  = " " * (lbl_w + 2) + " ".join(c.center(cell_w) for c in class_names)
    lines   = [header, sep]
    for i, name in enumerate(class_names):
        lines.append(f"{name:>{lbl_w}}  " + " ".join(cell(i, j) for j in range(n)))
    lines.append(sep)

    # Precision row
    prec = np.diag(conf_mat) / col_totals * 100
    lines.append(
        f"{'prec':>{lbl_w}}  " + " ".join(f"{p:3.0f}% ".rjust(cell_w) for p in prec)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint resume helpers
# ---------------------------------------------------------------------------

def _latest_checkpoint(archive_dir: Path) -> Optional[Path]:
    """Return the highest-numbered epoch .pt file, or None."""
    ckpts = sorted(archive_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _epoch_from_path(p: Path) -> int:
    return int(p.stem.split("_")[1])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_phase1(
    train_loader:        DataLoader,
    val_loader:          DataLoader,
    epochs:              int,
    lr:                  float,
    device:              str,
    archive_dir:         Path,
    resume:              bool  = False,
    weight_decay:        float = 0.01,
    accum_steps:         int   = 1,
    grad_checkpoint:     bool  = False,
    label_smoothing:     float = 0.1,
    container_smoothing: float = 0.2,         # heavier eps on disk_image/database
    warmup_pct:          float = 0.05,
    min_lr:              float = 0.0,
    compile_model:       bool  = False,
    class_weights        = None,              # (NUM_GROUPS,) CE weight tensor
    confusion_lambda:    float = 0.5,         # off-diagonal penalty weight
    confusion_source:    Optional[Path] = None,
    cutmix_alpha:        float = 0.0,
    ema_decay:           float = 0.999,
    extra_config:        dict  = {},
) -> CoarseClassifier:

    archive_dir.mkdir(parents=True, exist_ok=True)
    log_path = archive_dir / "training_log.json"

    encoder   = CoarseEncoder(grad_checkpoint=grad_checkpoint)
    model     = CoarseClassifier(encoder).to(device)

    # EMA of the weights, updated after every optimizer step. The averaged copy
    # is evaluated alongside the raw model each epoch; whichever scores higher on
    # val becomes best.pt. use_buffers=True so BatchNorm running stats average too.
    # Built before torch.compile so the deep-copy never touches a compiled graph.
    ema_model = AveragedModel(
        model,
        multi_avg_fn = get_ema_multi_avg_fn(ema_decay),
        use_buffers  = True,
    )

    if class_weights is not None:
        class_weights = class_weights.to(device)

    # --- Loss: per-class smoothing + off-diagonal confusion penalty ---
    container_overrides = {
        GROUP_TO_IDX["disk_image"]: container_smoothing,
        GROUP_TO_IDX["database"]:   container_smoothing,
    }
    smoothing_vec = build_label_smoothing_vector(
        num_classes       = NUM_GROUPS,
        default_smoothing = label_smoothing,
        class_overrides   = container_overrides,
    )
    if confusion_source is not None and Path(confusion_source).exists():
        cost = cost_matrix_from_log(confusion_source, NUM_GROUPS)
        print(f"Loaded confusion cost matrix from {confusion_source}")
    else:
        # Uniform off-diagonal cost when no prior log exists — confusion penalty
        # is mild but still nudges away from generic-misclass behaviour.
        cost = cost_matrix_from_array(np.ones((NUM_GROUPS, NUM_GROUPS), np.float32))
        print("No confusion source given — using uniform off-diagonal cost.")

    loss_fn = ConfusionWeightedCE(
        cost_matrix      = cost,
        confusion_lambda = confusion_lambda,
        class_weights    = class_weights,
        label_smoothing  = smoothing_vec,
    ).to(device)
    print(f"  loss: ConfusionWeightedCE(lambda={confusion_lambda}, "
          f"label_smoothing={label_smoothing}, container_smoothing={container_smoothing})")

    # --- Resume (load before compile so state_dict keys match) ---
    start_epoch = 0
    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    # AdamW: decoupled weight decay matches the paper exactly (β₁=0.9, β₂=0.999).
    # Adam (without W) conflates weight decay with L2 regularisation, which interacts
    # poorly with adaptive gradient scaling. AdamW fixes this and is the right choice.
    # make_optimizer excludes embeddings, BN, LN, and biases from weight decay — the
    # byte embedding table in particular must not be decayed (collapses toward zero).
    optimizer = make_optimizer(
        model,
        lr           = lr,
        weight_decay = weight_decay,
    )

    # Linear warmup then cosine annealing, both stepped once per optimizer update.
    # With gradient accumulation the number of optimizer steps per epoch is
    # len(loader) // accum_steps, so the LR curve is correct regardless of accum_steps.
    # warmup_pct=0.05 ≈ 2.5 epochs at default 50 epochs, matching the paper's 2-epoch warmup.
    opt_steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total_steps   = opt_steps_per_epoch * epochs
    warmup_steps  = max(1, round(total_steps * warmup_pct))
    cosine_steps  = max(1, total_steps - warmup_steps)

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-4,        # ramp from lr*1e-4 → lr over warmup_steps
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

    if resume:
        ckpt = _latest_checkpoint(archive_dir)
        if ckpt is not None:
            start_epoch = _epoch_from_path(ckpt)
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model_sd = state["model"]
            # torch.compile wraps the model and prefixes all keys with "_orig_mod.";
            # strip it so the state_dict loads into the uncompiled model correctly.
            if any(k.startswith("_orig_mod.") for k in model_sd):
                model_sd = {k.removeprefix("_orig_mod."): v for k, v in model_sd.items()}
            model.load_state_dict(model_sd)
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if scaler is not None and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            if "ema" in state:
                ema_model.load_state_dict(state["ema"])
            print(f"Resumed from {ckpt.name}  (epoch {start_epoch})")
        else:
            print("No checkpoint found in archive — starting fresh.")

    # torch.compile must happen after checkpoint load — the resume block above strips
    # the "_orig_mod." prefix that compile adds, so loading into the uncompiled model works.
    # Gives 2-3x throughput via kernel fusion.
    if compile_model:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # --- Load or initialise the JSON log ---
    if log_path.exists() and resume:
        log = json.loads(log_path.read_text())
    else:
        log = {
            "session_id":   datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at":   datetime.now().isoformat(),
            "config": {
                "epochs":              epochs,
                "lr":                  lr,
                "weight_decay":        weight_decay,
                "betas":               [0.9, 0.999],
                "label_smoothing":     label_smoothing,
                "container_smoothing": container_smoothing,
                "confusion_lambda":    confusion_lambda,
                "confusion_source":    str(confusion_source) if confusion_source else None,
                "warmup_pct":          warmup_pct,
                "min_lr":              min_lr,
                "optimizer":           "AdamW",
                "scheduler":           "LinearWarmup+CosineAnnealing",
                "loss":                "ConfusionWeightedCE",
                "amp":                 scaler is not None,
                "compile":             compile_model,
                "grad_checkpoint":     grad_checkpoint,
                "accum_steps":         accum_steps,
                "effective_batch":     train_loader.batch_size * accum_steps,
                "device":              device,
                "batch_size":          train_loader.batch_size,
                "encoder":             "CoarseEncoder",
                "cutmix_alpha":        cutmix_alpha,
                "ema_decay":           ema_decay,
                **extra_config,
            },
            "status":      "in_progress",
            "best": {
                "epoch":    None,
                "val_acc":  None,
                "val_loss": None,
            },
            "epochs": [],
        }
        _atomic_write_json(log_path, log)

    best_val_acc  = log["best"]["val_acc"] or 0.0
    num_batches   = len(train_loader)
    epoch_pad     = len(str(epochs))
    total_start   = time.time()

    for epoch in range(start_epoch, epochs):
        ep_num = epoch + 1
        ep_start = time.time()

        # --- Train ---
        desc = f"Epoch {ep_num:>{epoch_pad}}/{epochs} [train]"
        with tqdm(total=num_batches, desc=desc, unit="batch",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device, pbar,
                loss_fn=loss_fn,
                scaler=scaler, scheduler=scheduler, accum_steps=accum_steps,
                cutmix_alpha=cutmix_alpha, ema_model=ema_model,
            )

        # --- Validate raw model and EMA model; the better one becomes best.pt ---
        val_loss,     val_acc     = eval_one_epoch(model,     val_loader, device)
        ema_val_loss, ema_val_acc = eval_one_epoch(ema_model, val_loader, device)

        if ema_val_acc >= val_acc:
            cand_model, cand_acc, cand_loss, cand_src = ema_model, ema_val_acc, ema_val_loss, "ema"
        else:
            cand_model, cand_acc, cand_loss, cand_src = model, val_acc, val_loss, "raw"

        elapsed = time.time() - ep_start
        print(
            f"  val_acc={val_acc:.4f}  ema_val_acc={ema_val_acc:.4f}"
            f"  (best={cand_src} {cand_acc:.4f}, {_fmt_time(elapsed)})"
        )

        # --- Save model checkpoint (raw + EMA weights) ---
        ckpt_path = archive_dir / f"epoch_{ep_num:04d}.pt"
        ckpt_data = {
            "epoch":        ep_num,
            "model":        model.state_dict(),
            "ema":          ema_model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "train_loss":   train_loss,
            "train_acc":    train_acc,
            "val_loss":     val_loss,
            "val_acc":      val_acc,
            "ema_val_loss": ema_val_loss,
            "ema_val_acc":  ema_val_acc,
            "best_source":  cand_src,   # which key ("model"/"ema") holds the better weights
        }
        if scaler is not None:
            ckpt_data["scaler"] = scaler.state_dict()
        torch.save(ckpt_data, ckpt_path)

        # --- Update JSON log ---
        epoch_record = {
            "epoch":        ep_num,
            "train_loss":   round(train_loss, 6),
            "train_acc":    round(train_acc,  6),
            "val_loss":     round(val_loss,   6),
            "val_acc":      round(val_acc,    6),
            "ema_val_loss": round(ema_val_loss, 6),
            "ema_val_acc":  round(ema_val_acc,  6),
            "lr":           round(optimizer.param_groups[-1]["lr"], 8),
            "elapsed_s":    round(elapsed, 2),
            "timestamp":    datetime.now().isoformat(),
            "checkpoint":   ckpt_path.name,
        }
        log["epochs"].append(epoch_record)

        if cand_acc > best_val_acc:
            best_val_acc = cand_acc
            log["best"] = {
                "epoch":    ep_num,
                "val_acc":  round(cand_acc,  6),
                "val_loss": round(cand_loss, 6),
                "source":   cand_src,
            }
            # best.pt points at the epoch checkpoint; best_source in the
            # checkpoint says whether the "model" or "ema" key is the winner.
            best_path = archive_dir / "best.pt"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(ckpt_path.name)

            # Confusion matrix — computed on the winning model so it always
            # matches best.pt without needing to reload weights after the loop.
            conf_mat = eval_confusion_matrix(cand_model, val_loader, device, NUM_GROUPS)
            log["confusion_matrix"]         = conf_mat.tolist()
            log["confusion_matrix_classes"] = GROUP_NAMES
            print(f"\n  val confusion matrix [{cand_src}] (row=true, col=pred):")
            for line in format_confusion_matrix(conf_mat, GROUP_NAMES).split("\n"):
                print(f"    {line}")
            print()

        _atomic_write_json(log_path, log)

    # --- Finalise log ---
    log["status"]      = "complete"
    log["finished_at"] = datetime.now().isoformat()
    log["total_time"]  = _fmt_time(time.time() - total_start)
    _atomic_write_json(log_path, log)

    print(f"\nTraining complete in {log['total_time']}")
    print(f"Best val_acc: {log['best']['val_acc']:.4f}  (epoch {log['best']['epoch']})")
    print(f"Archive: {archive_dir}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Phase 1 coarse classifier with per-epoch checkpointing"
    )
    parser.add_argument("--binary-dir",    type=Path,  default=BINARY_DIR)
    parser.add_argument("--max-per-class", type=int,   default=None)
    parser.add_argument("--fraction",      type=float, default=None)
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=128)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight-decay",  type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Cross-entropy label smoothing (default: 0.1; 0 to disable)")
    parser.add_argument("--warmup-pct",    type=float, default=0.05,
                        help="Fraction of steps used for LR warmup (default: 0.05 ≈ 2.5 epochs at 50)")
    parser.add_argument("--min-lr",        type=float, default=0.0,
                        help="Cosine annealing lower bound eta_min (default: 0.0)")
    parser.add_argument("--compile",       action="store_true",
                        help="torch.compile the model for 2-3x CUDA throughput (PyTorch >= 2.0)")
    parser.add_argument("--lazy",          action="store_true",
                        help="Read fragments on demand (saves RAM)")
    parser.add_argument("--grad-accum",    type=int,   default=1,
                        help="Gradient accumulation steps (effective_batch = batch * accum)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Recompute encoder activations in backward to save ~60%% VRAM")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from latest checkpoint in archive")
    parser.add_argument("--archive-dir",   type=Path,
                        default=Path(__file__).parent / "phase1_archive",
                        help="Where to save checkpoints and log (default: ./phase1_archive)")
    parser.add_argument("--gpu-mem-fraction", type=float, default=None,
                        help="Max fraction of GPU memory to use, e.g. 0.9 for 90%% (default: unlimited)")
    parser.add_argument("--cpu-fraction", type=float, default=None,
                        help="Fraction of CPU cores for PyTorch to use, e.g. 0.5 for 50%% (default: all)")
    parser.add_argument("--noise-prob",   type=float, default=0.02,
                        help="Legacy byte-noise fraction (used only when --gbflip-sigma=0)")
    parser.add_argument("--no-augment",   action="store_true",
                        help="Disable all augmentation on the train split")
    parser.add_argument("--gbflip-sigma", type=float, default=0.01,
                        help="GBFlip per-fragment flip rate |N(0, sigma)|; 0 falls back to byte noise (default 0.01)")
    parser.add_argument("--gbflip-max-rate", type=float, default=0.05,
                        help="Hard cap on the GBFlip flip rate (default 0.05)")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0,
                        help="Beta-distribution alpha for byte CutMix; 0 disables (default: 0.0)")
    parser.add_argument("--ema-decay",    type=float, default=0.999,
                        help="EMA decay for the weight-averaged model evaluated alongside the raw one (default: 0.999)")
    parser.add_argument("--disk-image-weight", type=float, default=0.6,
                        help="CE weight for the disk_image group (default 0.6; 1.0 = no reweighting)")
    parser.add_argument("--container-smoothing", type=float, default=0.2,
                        help="Per-class label smoothing for disk_image/database (default 0.2)")
    parser.add_argument("--confusion-lambda",  type=float, default=0.5,
                        help="Off-diagonal confusion penalty weight (0 disables; default 0.5)")
    parser.add_argument("--confusion-source",  type=Path, default=None,
                        help="Prior training_log.json to seed the confusion-cost matrix")
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
    print(f"Archive: {args.archive_dir}")

    if args.lazy:
        from load_binary_lazy import LazyFragmentDataset, load_split_lazy

        print(f"Loading train split (lazy) from {args.binary_dir} ...")
        train_frag_path, train_indices, train_labels, sector = load_split_lazy(
            split="train",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print("Loading val split (lazy) ...")
        val_frag_path, val_indices, val_labels, _ = load_split_lazy(
            split="val",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        train_ds = LazyFragmentDataset(
            train_frag_path, sector, train_indices, train_labels,
            mode            = "coarse",
            augment         = not args.no_augment,
            noise_prob      = args.noise_prob,
            gbflip_sigma    = args.gbflip_sigma,
            gbflip_max_rate = args.gbflip_max_rate,
        )
        val_ds   = LazyFragmentDataset(val_frag_path, sector, val_indices, val_labels,
                                       mode="coarse", augment=False)

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

        train_ds = FragmentDataset(
            train_frags, train_labels,
            mode            = "coarse",
            augment         = not args.no_augment,
            noise_prob      = args.noise_prob,
            gbflip_sigma    = args.gbflip_sigma,
            gbflip_max_rate = args.gbflip_max_rate,
        )
        val_ds   = FragmentDataset(val_frags, val_labels, mode="coarse", augment=False)

    on_cuda = device.startswith("cuda")
    # persistent_workers: keeps worker processes alive between epochs, avoiding the
    # per-epoch fork/join overhead. prefetch_factor=2 keeps 2 batches queued per worker.
    dl_kwargs = dict(
        batch_size        = args.batch_size,
        num_workers       = 4,
        pin_memory        = on_cuda,
        persistent_workers= True,
        prefetch_factor   = 2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    # disk_image is a container format: a fragment from inside an ISO/IMG/VMDK can
    # be byte-identical to the text / exe / archive content stored within it, so at
    # the coarse level disk_image behaves as an uncertainty "black hole" that
    # swallows other classes (epoch-1 matrix: 16% precision, attracting 20-62% of
    # six other groups). Down-weighting it in the loss makes Phase 1 claim
    # disk_image only for confidently-structural fragments (filesystem metadata,
    # volume descriptors) and lets ambiguous content fragments fall through to
    # their true class. All other groups keep weight 1.0.
    class_weights = torch.ones(NUM_GROUPS, dtype=torch.float32)
    class_weights[GROUP_TO_IDX["disk_image"]] = args.disk_image_weight
    print(f"  cross-entropy class weights: disk_image={args.disk_image_weight}, others=1.0")

    train_phase1(
        train_loader        = train_loader,
        val_loader          = val_loader,
        epochs              = args.epochs,
        lr                  = args.lr,
        weight_decay        = args.weight_decay,
        label_smoothing     = args.label_smoothing,
        container_smoothing = args.container_smoothing,
        confusion_lambda    = args.confusion_lambda,
        confusion_source    = args.confusion_source,
        warmup_pct          = args.warmup_pct,
        min_lr              = args.min_lr,
        compile_model       = args.compile,
        accum_steps         = args.grad_accum,
        grad_checkpoint     = args.grad_checkpoint,
        class_weights       = class_weights,
        cutmix_alpha        = args.cutmix_alpha,
        ema_decay           = args.ema_decay,
        device              = device,
        archive_dir         = args.archive_dir,
        resume              = args.resume,
        extra_config = {
            "max_per_class":      args.max_per_class,
            "fraction":           args.fraction,
            "lazy":               args.lazy,
            "binary_dir":         str(args.binary_dir),
            "gpu_mem_fraction":   args.gpu_mem_fraction,
            "cpu_fraction":       args.cpu_fraction,
            "augment":            not args.no_augment,
            "noise_prob":         args.noise_prob,
            "gbflip_sigma":       args.gbflip_sigma,
            "gbflip_max_rate":    args.gbflip_max_rate,
            "disk_image_weight":  args.disk_image_weight,
        },
    )
