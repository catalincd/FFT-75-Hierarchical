"""
Hierarchical Cascade Classifier for FFT-75 File Fragment Classification

Stage 1: Coarse classifier -> predicts one of 11 use-specific groups (Scenario #2)
Stage 2: Per-group specialist -> predicts fine-grained file type within that group

This mirrors the FFT-75 scenario structure:
  Scenario #1 = 75-class flat (hard)
  Scenario #2 = 11-class groups (easy, ~90%+ accuracy)
  Cascade = use Sc#2 to gate Sc#1 specialists
"""

import json
import time
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm

from load_binary import load_split, label_indices_to_strings, BINARY_DIR

# ---------------------------------------------------------------------------
# FFT-75 Group Structure (Scenario #2 -> Scenario #1 mapping)
# Each key is a group name; values are the fine-grained class labels within it.
# Adjust to match the actual FFT-75 label set you're working with.
# ---------------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
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

# Build lookup tables
GROUP_NAMES  = list(GROUPS.keys())                        # 11 groups
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUP_NAMES)}
TYPE_TO_GROUP = {t: g for g, types in GROUPS.items() for t in types}
ALL_TYPES    = [t for types in GROUPS.values() for t in types]  # 75 types
TYPE_TO_IDX  = {t: i for i, t in enumerate(ALL_TYPES)}

# Per-group: local index of each fine-grained type within its group
GROUP_LOCAL_IDX: dict[str, dict[str, int]] = {
    g: {t: i for i, t in enumerate(types)}
    for g, types in GROUPS.items()
}

NUM_GROUPS   = len(GROUP_NAMES)                 # 11
SECTOR_SIZE  = 512                              # bytes per fragment

# ---------------------------------------------------------------------------
# Building blocks (ByteResNet / JSANet inspired, 2024)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al. 2018).
    Used by JSANet (2024) as channel self-attention for file fragment classification.
    Recalibrates channel responses by modelling inter-channel dependencies.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)  ->  rescale channels by learned weights
        return x * self.se(x).unsqueeze(-1)


class ResConvBlock(nn.Module):
    """
    Post-activation residual 1D conv block with SE channel attention.

    Architecture (ByteResNet-style):
        Conv -> BN -> GELU -> Conv -> BN -> SE -> + shortcut -> GELU

    A projection shortcut (1x1 conv) is added when in_ch != out_ch so the
    skip path always matches the main path dimensionally.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock(out_ch)
        self.act   = nn.GELU()
        self.proj  = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1),
                          nn.BatchNorm1d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)))
        return self.act(x + residual)


# ---------------------------------------------------------------------------
# Attention pooling — replaces GAP; learns which positions are discriminative
# ---------------------------------------------------------------------------

class AttentionPool1d(nn.Module):
    """Learns which spatial positions are discriminative before collapsing."""
    def __init__(self, d: int):
        super().__init__()
        self.attn = nn.Linear(d, 1, bias=False)  # scalar score per position

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d, L)
        w = torch.softmax(self.attn(x.permute(0, 2, 1)), dim=1)  # (B, L, 1)
        return (x.permute(0, 2, 1) * w).sum(dim=1)               # (B, d)


# ---------------------------------------------------------------------------
# Shared Byte Encoder — ByteResNet backbone (2024 SOTA on FFT-75)
# Input: (B, L) raw bytes  ->  Output: (B, out_dim) feature vector
# Works for any sector length L (512 or 4096 bytes) via AttentionPool1d.
# ---------------------------------------------------------------------------

class ByteEncoder(nn.Module):
    """
    Residual CNN byte encoder inspired by ByteResNet (2024) and JSANet (2024).

    Improvements over the original FiFTy backbone:
      - Residual connections prevent gradient vanishing in deeper stacks
      - SE channel attention recalibrates feature maps after each residual block
      - GELU activations + BatchNorm for faster, more stable convergence
      - 5-stage progressive downsampling; AttentionPool1d handles 512 or 4096 bytes
      - Larger embedding dim (16 vs 8) encodes richer byte co-occurrence patterns
    """
    def __init__(self, embed_dim: int = 16, num_filters: int = 128,
                 grad_checkpoint: bool = False):
        super().__init__()
        F = num_filters
        self.embed          = nn.Embedding(256, embed_dim)
        self.grad_checkpoint = grad_checkpoint

        # Stem: wide kernel captures local multi-byte patterns before residual stages
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, F, kernel_size=7, padding=3),
            nn.BatchNorm1d(F),
            nn.GELU(),
        )

        # Stage 1: single block, large kernel for long-range byte context
        self.stage1 = ResConvBlock(F, F, kernel_size=7)
        self.pool1  = nn.MaxPool1d(4)                       # L -> L/4

        # Stage 2: two blocks at 2x width
        self.stage2 = nn.Sequential(
            ResConvBlock(F,   F*2, kernel_size=5),
            ResConvBlock(F*2, F*2, kernel_size=5),
        )
        self.pool2  = nn.MaxPool1d(4)                       # L/4 -> L/16

        # Stage 3: two blocks at 4x width, fine-grained pattern discrimination
        self.stage3 = nn.Sequential(
            ResConvBlock(F*2, F*4, kernel_size=3),
            ResConvBlock(F*4, F*4, kernel_size=3),
        )
        self.pool3  = nn.MaxPool1d(4)                       # L/16 -> L/64

        # Stage 4: two blocks at 8x width, high-level semantic features
        self.stage4 = nn.Sequential(
            ResConvBlock(F*4, F*8, kernel_size=3),
            ResConvBlock(F*8, F*8, kernel_size=3),
        )

        self.pool    = AttentionPool1d(F * 8)               # weighted spatial collapse
        self.out_dim = F * 8                                # 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) int64 byte values
        x = self.embed(x).permute(0, 2, 1)                 # (B, embed_dim, L)
        x = self.stem(x)
        if self.grad_checkpoint and self.training:
            # Recompute stage activations during backward instead of storing them.
            # Cuts activation memory by ~60% at the cost of one extra forward pass.
            x = self.pool1(grad_ckpt(self.stage1, x, use_reentrant=False))
            x = self.pool2(grad_ckpt(self.stage2, x, use_reentrant=False))
            x = self.pool3(grad_ckpt(self.stage3, x, use_reentrant=False))
            x = grad_ckpt(self.stage4, x, use_reentrant=False)
        else:
            x = self.pool1(self.stage1(x))
            x = self.pool2(self.stage2(x))
            x = self.pool3(self.stage3(x))
            x = self.stage4(x)
        return self.pool(x)                                 # (B, out_dim)


# ---------------------------------------------------------------------------
# Bigram branch — global byte co-occurrence statistics
# ---------------------------------------------------------------------------

def build_bigram(x: torch.Tensor) -> torch.Tensor:
    """
    Loop-free 256×256 bigram matrix from raw byte sequence.

    For L bytes there are L-1 consecutive pairs. Each pair (a, b) increments
    mat[a*256+b]. The result is normalised by (L-1) so inputs of different
    lengths produce comparable distributions.

    Returns: (B, 1, 256, 256) float32, ready for Conv2d.
    """
    B, L = x.shape
    idx = x[:, :-1].long() * 256 + x[:, 1:].long()          # (B, L-1)
    mat = torch.zeros(B, 256 * 256, device=x.device, dtype=torch.float32)
    mat.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    return (mat / (L - 1)).reshape(B, 1, 256, 256)


class BigramBranch(nn.Module):
    """
    256×256 byte co-occurrence matrix → compact feature vector.

    Motivation: the CNN backbone must build up receptive field through pooling
    layers to capture long-range byte relationships.  The bigram matrix is a
    global statistic by construction — every byte pair contributes regardless
    of position — so it completely bypasses the receptive-field problem and is
    particularly valuable for 4096-byte sectors where distant pairs are common.

    Architecture: treat the 256×256 matrix as a 1-channel "image", two conv
    stages reduce it to a 64-d global vector, then project to out_dim.
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.MaxPool2d(4),                              # → (32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                      # → (64, 1, 1)
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) int64 byte values
        return self.net(build_bigram(x))                    # (B, out_dim)


# ---------------------------------------------------------------------------
# Fused encoder — CNN branch + bigram branch, same interface as ByteEncoder
# ---------------------------------------------------------------------------

class FusedEncoder(nn.Module):
    """
    Concatenates ByteEncoder (sequential CNN) and BigramBranch (global stats).

    Exposes the same `.out_dim` / `forward(x)` interface as ByteEncoder so
    CoarseClassifier, SpecialistClassifier, and HierarchicalCascade require
    no changes beyond instantiating FusedEncoder instead of ByteEncoder.

    Default dims: ByteEncoder(F=128) → 1024, BigramBranch → 512, total 1536.
    """
    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.byte_enc   = ByteEncoder(embed_dim, num_filters, grad_checkpoint)
        self.bigram_enc = BigramBranch(out_dim=bigram_dim)
        self.out_dim    = self.byte_enc.out_dim + bigram_dim  # 1536

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) int64 byte values
        cnn_feat    = self.byte_enc(x)                       # (B, 1024)
        bigram_feat = self.bigram_enc(x)                     # (B, 512)
        return torch.cat([cnn_feat, bigram_feat], dim=-1)    # (B, 1536)


# ---------------------------------------------------------------------------
# Specialist encoder — Archive (zip / gz / bz2 / 7z / tar / rar)
# ---------------------------------------------------------------------------

def _byte_hist(x: torch.Tensor) -> torch.Tensor:
    """
    Normalised byte-frequency histogram.  x: (B, L) int64  →  (B, 256) float32.

    Unlike the bigram matrix, which encodes *pair* statistics, the unigram
    histogram captures the marginal distribution of byte values.  This is a
    compression-algorithm fingerprint: BWT (bz2) produces runs of similar bytes
    so its histogram is peaky; DEFLATE (zip/gz) and LZMA (7z) are flatter but
    with different shapes; raw TAR blocks are mostly 0x00-padded.
    """
    B, L = x.shape
    hist = torch.zeros(B, 256, device=x.device, dtype=torch.float32)
    hist.scatter_add_(1, x.long(), torch.ones(B, L, device=x.device))
    return hist / L                                          # (B, 256)


class ArchiveEncoder(nn.Module):
    """
    Encoder specialised for compressed-format discrimination (zip/gz/bz2/7z/tar/rar).

    The core problem: middle-of-file fragments from compressed files are
    pseudo-random bytes — the *only* reliable discriminating features are:

      1. Magic-header bytes (first few bytes of the *file*).  When the fragment
         starts near offset 0, these are deterministic:
           zip  →  PK\\x03\\x04
           gz   →  \\x1f\\x8b
           bz2  →  BZh
           7z   →  7z\\xbc\\xaf\\x27\\x1c
           rar  →  Rar!\\x1a\\x07
           tar  →  "ustar" at offset 257 (512-byte header block)
         Even when the magic bytes are not in the window, the start of a
         fragment may still fall near format-specific structural regions.
         → HeaderBranch: embed first HEADER_BYTES bytes → MLP

      2. Compression-algorithm statistical fingerprint: the byte-frequency
         histogram differs measurably by algorithm even in the middle of the
         compressed stream.  BWT (bz2) produces value clustering; DEFLATE
         (zip/gz) is more uniform; LZMA (7z) has characteristic valleys.
         → HistBranch: 256-d normalised histogram → MLP

      3. Global sequential patterns and pair co-occurrences (existing).
         → ByteEncoder + BigramBranch (same as FusedEncoder)

    Keeps `byte_enc` and `bigram_enc` as top-level attributes so
    `_load_encoder_from_phase1` works without modification.

    out_dim: 1024 (CNN) + 512 (bigram) + 64 (header) + 128 (hist) = 1728
    """
    HEADER_BYTES = 64   # first N bytes of the fragment; covers all magic signatures

    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        header_dim:      int  = 64,
        hist_dim:        int  = 128,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.byte_enc   = ByteEncoder(embed_dim, num_filters, grad_checkpoint)
        self.bigram_enc = BigramBranch(out_dim=bigram_dim)

        # Header branch — embed each of the first HEADER_BYTES bytes to 8-d,
        # flatten to (HEADER_BYTES * 8), project down to header_dim.
        self.header_embed = nn.Embedding(256, 8)
        self.header_mlp   = nn.Sequential(
            nn.Linear(self.HEADER_BYTES * 8, 256),
            nn.GELU(),
            nn.Linear(256, header_dim),
        )

        # Histogram branch — 256-d unigram frequency → hist_dim
        self.hist_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, hist_dim),
        )

        self.out_dim = self.byte_enc.out_dim + bigram_dim + header_dim + hist_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat    = self.byte_enc(x)                                # (B, 1024)
        bigram_feat = self.bigram_enc(x)                              # (B, 512)

        header      = x[:, :self.HEADER_BYTES].long()                # (B, 64)
        header_feat = self.header_mlp(
            self.header_embed(header).flatten(1)                      # (B, 64*8)
        )                                                             # (B, header_dim)

        hist_feat   = self.hist_mlp(_byte_hist(x))                   # (B, hist_dim)

        return torch.cat([cnn_feat, bigram_feat, header_feat, hist_feat], dim=-1)


# ---------------------------------------------------------------------------
# Specialist encoder — Text (txt / csv / xml / json / html / log)
# ---------------------------------------------------------------------------

# Structural ASCII characters that discriminate text formats.
# Each character's *frequency* within a 512-byte window is a strong signal:
#   JSON  → dense  { } [ ] : "
#   XML   → dense  < > / =
#   HTML  → dense  < > / = (+ & entities)
#   CSV   → dense  ,  and regular \n cadence
#   log   → dense  :  [ ] and high digit ratio
#   txt   → low density of all structural chars; mostly alphanumeric + space
_TEXT_STRUCT_CHARS: list[int] = [
    # JSON / dict
    ord('{'), ord('}'), ord('['), ord(']'), ord(':'),
    # XML / HTML
    ord('<'), ord('>'), ord('/'), ord('='), ord('&'),
    # Quoting / strings
    ord('"'), ord("'"),
    # CSV / tabular
    ord(','), ord(';'), ord('|'),
    # Whitespace patterns
    ord('\n'), ord('\r'), ord('\t'),
    # Log / miscellaneous structural
    ord('#'), ord('@'), ord('\\'), ord('('), ord(')'), ord('!'),
]   # 23 characters


class TextEncoder(nn.Module):
    """
    Encoder specialised for text-format discrimination (txt/csv/xml/json/html/log).

    The core problem: all six classes are ASCII/UTF-8 text, so the byte
    distribution is globally similar.  Discriminating features are:

      1. Structural-character frequency: a 23-dim vector of per-character
         occurrence rates computed directly from the raw bytes.  This is
         what a human uses to identify format at a glance — curly braces mean
         JSON, angle brackets mean XML/HTML, comma density means CSV — but the
         CNN must build it up slowly via receptive-field expansion.  Providing
         it explicitly as an additional branch short-circuits that problem.
         → StructBranch: freq(23 chars) → MLP → 64-d

      2. Global sequential + co-occurrence patterns (existing).
         → ByteEncoder + BigramBranch (same as FusedEncoder)

    Keeps `byte_enc` and `bigram_enc` as top-level attributes so
    `_load_encoder_from_phase1` works without modification.

    out_dim: 1024 (CNN) + 512 (bigram) + 64 (struct) = 1600
    """

    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        struct_dim:      int  = 64,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.byte_enc   = ByteEncoder(embed_dim, num_filters, grad_checkpoint)
        self.bigram_enc = BigramBranch(out_dim=bigram_dim)

        n = len(_TEXT_STRUCT_CHARS)
        # Register as buffer so the tensor moves to the correct device automatically
        self.register_buffer(
            "_struct_chars",
            torch.tensor(_TEXT_STRUCT_CHARS, dtype=torch.long),
        )

        # Structural character branch — (B, n) frequency vector → struct_dim
        self.struct_mlp = nn.Sequential(
            nn.Linear(n, 128),
            nn.GELU(),
            nn.Linear(128, struct_dim),
        )

        self.out_dim = self.byte_enc.out_dim + bigram_dim + struct_dim

    def _struct_freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-character occurrence rate.
        x: (B, L) int64  →  (B, n_chars) float32  (values in [0, 1])
        """
        # (B, L, 1) == (1, 1, n_chars)  →  (B, L, n_chars)  →  mean over L
        return (x.unsqueeze(-1) == self._struct_chars).float().mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat    = self.byte_enc(x)                       # (B, 1024)
        bigram_feat = self.bigram_enc(x)                     # (B, 512)
        struct_feat = self.struct_mlp(self._struct_freq(x))  # (B, struct_dim)
        return torch.cat([cnn_feat, bigram_feat, struct_feat], dim=-1)


# ---------------------------------------------------------------------------
# Stage 1: Coarse Group Classifier
# ---------------------------------------------------------------------------

class CoarseClassifier(nn.Module):
    """
    Predicts one of NUM_GROUPS coarse groups.

    Over the baseline 3-layer BN-MLP:
      - LayerNorm: per-sample, no batch-stat sensitivity; cascade routing often
        processes small per-group sub-batches after Stage 1 argmax
      - GeGLU (Noam et al., 2020): the gate learns to suppress encoder channels
        irrelevant to group separation, outperforming plain GELU on projection
        layers; costs the same FLOPs as the original first Linear
      - Single residual hidden layer: Scenario #2 (11 classes) is well-separated
        by the encoder; the third BN-MLP layer added depth without benefit
      - Skip projection: gradient highway so the encoder is never a bottleneck
    """
    def __init__(self, encoder: ByteEncoder):
        super().__init__()
        self.encoder = encoder
        d      = encoder.out_dim          # 1536 with FusedEncoder (1024 CNN + 512 bigram)
        hidden = d // 2                   # 256

        self.norm       = nn.LayerNorm(d)
        # Single matrix → split into value + gate (same param count as Linear(d,d))
        self.geglu_proj = nn.Linear(d, hidden * 2, bias=False)
        self.drop       = nn.Dropout(0.3)
        # Projection shortcut: aligns d → hidden so residual dimensions match
        self.skip       = nn.Linear(d, hidden, bias=False)
        self.out_norm   = nn.LayerNorm(hidden)
        self.head       = nn.Linear(hidden, NUM_GROUPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.norm(self.encoder(x))                    # (B, d)
        v, g = self.geglu_proj(feat).chunk(2, dim=-1)        # each (B, hidden)
        h    = self.drop(v * F.gelu(g)) + self.skip(feat)   # GeGLU + residual
        return self.head(self.out_norm(h))                   # (B, NUM_GROUPS)

# ---------------------------------------------------------------------------
# Stage 2: Per-Group Fine-Grained Specialist
# One specialist per group; each only predicts within its own class set.
# ---------------------------------------------------------------------------

class SpecialistClassifier(nn.Module):
    """
    Predicts the fine-grained type within a single group.

    Head matches CoarseClassifier quality: LayerNorm → Linear → GELU →
    Dropout → residual skip → LayerNorm → head.  The original Sequential
    (Linear → ReLU → Dropout → Linear) had no input normalisation and no
    skip path, which made it harder to train when the encoder is pretrained
    and the head is randomly initialised.
    """
    def __init__(self, encoder: ByteEncoder, num_classes: int):
        super().__init__()
        self.encoder  = encoder
        d             = encoder.out_dim
        hidden        = d // 4              # 384 for FusedEncoder (1536 // 4)
        self.norm     = nn.LayerNorm(d)
        self.fc1      = nn.Linear(d, hidden, bias=False)
        self.drop     = nn.Dropout(0.3)
        self.skip     = nn.Linear(d, hidden, bias=False)
        self.out_norm = nn.LayerNorm(hidden)
        self.head     = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.norm(self.encoder(x))                   # (B, d)
        h    = self.drop(F.gelu(self.fc1(feat))) + self.skip(feat)  # (B, hidden)
        return self.head(self.out_norm(h))                  # (B, num_classes)


# ---------------------------------------------------------------------------
# Optimizer factory — correct AdamW parameter grouping
# ---------------------------------------------------------------------------

def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    encoder_lr_scale: float = 1.0,
) -> torch.optim.AdamW:
    """
    AdamW with two correctness fixes applied:

    1.  No weight decay on embeddings, BatchNorm, LayerNorm, or biases.
        Decaying the byte embedding table collapses vectors toward zero and
        degrades the learned byte co-occurrence representation.  BN/LN
        scale+shift parameters are unit-scale priors — decaying them fights
        the normalisation layer's own purpose.

    2.  Optional lower LR for the encoder (encoder_lr_scale < 1.0).
        When the encoder is warm-started from phase1 and the head is randomly
        initialised, using the same LR for both causes the encoder to drift
        too fast (catastrophic forgetting) while the head barely converges.
        Typical value: 0.1 (encoder updates at lr/10, head at lr).
    """
    # Collect parameter ids that belong to no-decay module types
    no_decay_types = (nn.Embedding, nn.BatchNorm1d, nn.LayerNorm)
    no_decay_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, no_decay_types):
            for p in mod.parameters(recurse=False):
                no_decay_ids.add(id(p))

    has_encoder = hasattr(model, "encoder")
    encoder_lr  = lr * encoder_lr_scale

    buckets: dict[str, list] = {
        "enc_decay":    [],
        "enc_nodecay":  [],
        "hd_decay":     [],
        "hd_nodecay":   [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_encoder  = has_encoder and name.startswith("encoder.")
        is_no_decay = id(param) in no_decay_ids or name.endswith(".bias")
        key = ("enc" if is_encoder else "hd") + ("_nodecay" if is_no_decay else "_decay")
        buckets[key].append(param)

    param_groups = [
        {"params": buckets["enc_decay"],   "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": buckets["enc_nodecay"], "lr": encoder_lr, "weight_decay": 0.0},
        {"params": buckets["hd_decay"],    "lr": lr,         "weight_decay": weight_decay},
        {"params": buckets["hd_nodecay"],  "lr": lr,         "weight_decay": 0.0},
    ]
    param_groups = [g for g in param_groups if g["params"]]  # drop empty buckets

    return torch.optim.AdamW(param_groups, betas=betas)


# ---------------------------------------------------------------------------
# Hierarchical Cascade: orchestrates Stage 1 + Stage 2
# ---------------------------------------------------------------------------

class HierarchicalCascade(nn.Module):
    """
    Combines a shared encoder, one coarse classifier, and one specialist
    per group into a single module.

    Specialists can optionally share the encoder with the coarse model
    (shared_encoder=True) or use independent encoders (shared_encoder=False).
    Sharing saves memory and promotes general representations; separate encoders
    allow each specialist to learn format-specific low-level features.
    """
    def __init__(self, shared_encoder: bool = True):
        super().__init__()

        if shared_encoder:
            encoder = FusedEncoder()
            self.coarse = CoarseClassifier(encoder)
            self.specialists = nn.ModuleDict({
                group: SpecialistClassifier(encoder, len(types))
                for group, types in GROUPS.items()
            })
        else:
            self.coarse = CoarseClassifier(FusedEncoder())
            self.specialists = nn.ModuleDict({
                group: SpecialistClassifier(FusedEncoder(), len(types))
                for group, types in GROUPS.items()
            })

    def forward_train_coarse(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1 logits for training the coarse classifier."""
        return self.coarse(x)

    def forward_train_specialist(
        self, x: torch.Tensor, group: str
    ) -> torch.Tensor:
        """Stage 2 logits for training a single specialist."""
        return self.specialists[group](x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_confidence: bool = False,
    ) -> tuple[list[str], Optional[torch.Tensor]]:
        """
        Full cascade inference.

        Returns:
            predictions: list of predicted fine-grained type strings, length B
            confidence:  (B, 2) tensor of (group_conf, type_conf) if requested
        """
        # Stage 1: predict group
        group_logits = self.coarse(x)                       # (B, 11)
        group_probs  = F.softmax(group_logits, dim=-1)
        group_preds  = group_logits.argmax(dim=-1)          # (B,)

        predictions: list[str] = []
        group_confs  = group_probs.max(dim=-1).values       # (B,)
        type_confs   = torch.zeros(len(x), device=x.device)

        # Route each sample to its predicted specialist
        # Process group-by-group to avoid a Python loop over every sample.
        for group_idx, group_name in enumerate(GROUP_NAMES):
            mask = (group_preds == group_idx).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue

            x_sub = x[mask]                                 # (k, 512)
            specialist = self.specialists[group_name]
            type_logits = specialist(x_sub)                 # (k, num_types)
            type_probs  = F.softmax(type_logits, dim=-1)
            local_preds = type_logits.argmax(dim=-1)        # (k,)

            type_confs[mask] = type_probs.max(dim=-1).values
            local_types = GROUPS[group_name]
            for i, sample_idx in enumerate(mask.tolist()):
                predictions.insert(sample_idx, local_types[local_preds[i].item()])

        if return_confidence:
            conf = torch.stack([group_confs, type_confs], dim=-1)  # (B, 2)
            return predictions, conf

        return predictions, None

# ---------------------------------------------------------------------------
# Data loading with optional subsampling
# ---------------------------------------------------------------------------

def load_data(
    split: str = "train",
    max_per_class: Optional[int] = None,
    fraction: Optional[float] = None,
    binary_dir: Path = BINARY_DIR,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Load a split from binary files and optionally cut it down.

    Subsampling is done per-class to preserve label balance.
    Use max_per_class to set a hard cap, or fraction for a proportional cut.
    If both are given, max_per_class takes precedence.

    Returns:
        fragments:  (N, sector_size) uint8 array
        labels:     list[str] of fine-grained type names, length N
    """
    fragments, label_indices, all_types = load_split(split, binary_dir=binary_dir)
    labels = label_indices_to_strings(label_indices, all_types)

    if max_per_class is None and fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        # Convert fraction to a per-class cap based on the smallest class
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
# Dataset
# ---------------------------------------------------------------------------

class FragmentDataset(Dataset):
    """
    Dataset wrapper for FFT-75 style data with optional byte-level augmentation.

    Expected: fragments as (N, 512) uint8 numpy array,
              fine-grained labels as list of type strings length N.

    Augmentation (training only, set augment=True):
        Byte noise — randomly replaces `noise_prob` fraction of bytes with
        uniformly random values (0-255) each time a sample is fetched.
        Because the DataLoader re-fetches every epoch, the same fragment
        appears in a slightly different corrupted form every epoch, giving
        the model effectively infinite variations of each training example.
        This prevents memorisation of exact byte patterns without requiring
        any additional real data.
    """
    def __init__(
        self,
        fragments:  np.ndarray,
        labels:     list[str],
        mode:       str   = "coarse",   # "coarse" | "specialist:<group>"
        augment:    bool  = False,      # enable byte-noise augmentation
        noise_prob: float = 0.02,       # fraction of bytes to corrupt per sample
    ):
        assert len(fragments) == len(labels)
        assert mode == "coarse" or mode.startswith("specialist:")

        self.mode       = mode
        self.augment    = augment
        self.noise_prob = noise_prob
        target_group    = mode.split(":")[1] if ":" in mode else None

        if target_group is not None:
            keep = [
                i for i, lbl in enumerate(labels)
                if TYPE_TO_GROUP.get(lbl) == target_group
            ]
            self.fragments = fragments[keep]
            self.labels    = [labels[i] for i in keep]
            self.label_map = GROUP_LOCAL_IDX[target_group]
        else:
            self.fragments = fragments
            self.labels    = labels
            self.label_map = GROUP_TO_IDX

    def __len__(self) -> int:
        return len(self.fragments)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.fragments[idx].astype(np.int64))

        if self.augment and self.noise_prob > 0:
            # Salt-and-pepper byte corruption: random bytes override ~noise_prob
            # of positions.  Byte values stay in [0, 255].
            mask  = torch.rand(x.shape) < self.noise_prob
            noise = torch.randint_like(x, 0, 256)
            x     = torch.where(mask, noise, x)

        if self.mode == "coarse":
            group = TYPE_TO_GROUP[self.labels[idx]]
            y = self.label_map[group]
        else:
            y = self.label_map[self.labels[idx]]

        return x, y

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr:          float = 1e-3
    epochs:      int   = 50
    batch_size:  int   = 512
    device:      str   = "cuda" if torch.cuda.is_available() else "cpu"
    shared_encoder: bool = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pbar: tqdm,
) -> float:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        total_loss += batch_loss * len(x)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += len(x)
        running_loss += batch_loss
        pbar.set_postfix(
            loss=f"{running_loss / step:.4f}",
            acc=f"{total_correct / total_samples:.3f}",
            refresh=False,
        )
        pbar.update(1)
    return total_loss / len(loader.dataset), total_correct / total_samples


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _write_checkpoint(path: Path, data: dict) -> None:
    """Atomically overwrite the checkpoint JSON."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def train_cascade(
    fragments: Optional[np.ndarray],
    labels: list[str],
    cfg: TrainConfig = TrainConfig(),
    dataset_factory=None,   # callable(mode: str) -> Dataset; overrides FragmentDataset
    checkpoint_path: Optional[Path] = None,
) -> HierarchicalCascade:
    """
    Two-phase training:
      Phase 1 — train the coarse classifier on all data
      Phase 2 — train each specialist on its group's data only

    dataset_factory, if provided, is called as dataset_factory(mode) and must
    return a Dataset.  Use this for lazy / on-disk loading (see load_binary_lazy).
    When None, the default FragmentDataset is used with the fragments array.
    """
    def _make_ds(mode: str):
        if dataset_factory is not None:
            return dataset_factory(mode)
        return FragmentDataset(fragments, labels, mode=mode)

    total_start = time.time()
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    cascade = HierarchicalCascade(shared_encoder=cfg.shared_encoder).to(cfg.device)

    progress: dict = {
        "session_id":   session_id,
        "started_at":   datetime.now().isoformat(),
        "config":       {"epochs": cfg.epochs, "batch_size": cfg.batch_size,
                         "lr": cfg.lr, "shared_encoder": cfg.shared_encoder,
                         "device": cfg.device},
        "status":       "in_progress",
        "phase1_coarse":     {"epochs": []},
        "phase2_specialists": {g: {"epochs": []} for g in GROUP_NAMES},
    }
    if checkpoint_path is not None:
        _write_checkpoint(checkpoint_path, progress)

    # --- Phase 1: coarse classifier ---
    print("=== Phase 1: training coarse classifier ===")
    coarse_ds     = _make_ds("coarse")
    coarse_loader = DataLoader(coarse_ds, batch_size=cfg.batch_size, shuffle=True)
    num_batches   = len(coarse_loader)

    coarse_params = (
        list(cascade.coarse.encoder.parameters()) +
        list(cascade.coarse.head.parameters())
    )
    opt = torch.optim.Adam(coarse_params, lr=cfg.lr)

    phase1_start = time.time()
    for epoch in range(cfg.epochs):
        desc = f"Epoch {epoch+1:>{len(str(cfg.epochs))}}/{cfg.epochs}"
        with tqdm(total=num_batches, desc=desc, unit="batch",
                  bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            loss, acc = train_one_epoch(cascade.coarse, coarse_loader, opt, cfg.device, pbar)
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
        if checkpoint_path is not None:
            progress["phase1_coarse"]["epochs"].append({
                "epoch": epoch + 1,
                "loss":  round(loss, 6),
                "acc":   round(acc,  6),
                "timestamp": datetime.now().isoformat(),
            })
            _write_checkpoint(checkpoint_path, progress)
    print(f"Phase 1 complete in {_fmt_time(time.time() - phase1_start)}\n")

    # --- Phase 2: train each specialist ---
    print("=== Phase 2: training specialists ===")
    phase2_start = time.time()
    for group_name in GROUP_NAMES:
        spec_ds = _make_ds(f"specialist:{group_name}")
        if len(spec_ds) == 0:
            print(f"  [{group_name}] no samples — skipping")
            continue

        spec_loader = DataLoader(spec_ds, batch_size=cfg.batch_size, shuffle=True)
        specialist  = cascade.specialists[group_name]
        num_classes = len(GROUPS[group_name])
        num_batches = len(spec_loader)
        print(f"  [{group_name}]  {len(spec_ds)} samples, {num_classes} classes")

        if cfg.shared_encoder:
            spec_params = list(specialist.head.parameters())
        else:
            spec_params = list(specialist.parameters())

        opt = torch.optim.Adam(spec_params, lr=cfg.lr)
        spec_start = time.time()

        for epoch in range(cfg.epochs):
            desc = f"  Epoch {epoch+1:>{len(str(cfg.epochs))}}/{cfg.epochs}"
            with tqdm(total=num_batches, desc=desc, unit="batch",
                      bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
                loss, acc = train_one_epoch(specialist, spec_loader, opt, cfg.device, pbar)
                pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
            if checkpoint_path is not None:
                progress["phase2_specialists"][group_name]["epochs"].append({
                    "epoch": epoch + 1,
                    "loss":  round(loss, 6),
                    "acc":   round(acc,  6),
                    "timestamp": datetime.now().isoformat(),
                })
                _write_checkpoint(checkpoint_path, progress)

        print(f"  [{group_name}] done in {_fmt_time(time.time() - spec_start)} | loss: {loss:.4f} | acc: {acc:.3f}\n")

    if checkpoint_path is not None:
        progress["status"] = "complete"
        progress["finished_at"] = datetime.now().isoformat()
        _write_checkpoint(checkpoint_path, progress)

    print(f"Phase 2 complete in {_fmt_time(time.time() - phase2_start)}")
    print(f"\nTotal training time: {_fmt_time(time.time() - total_start)}")
    return cascade

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    cascade: HierarchicalCascade,
    fragments: np.ndarray,
    labels: list[str],
    batch_size: int = 512,
) -> dict:
    """
    Returns per-stage accuracy:
      - coarse_acc: how often Stage 1 predicted the correct group
      - fine_acc:   end-to-end fine-grained accuracy
      - oracle_fine_acc: fine accuracy when Stage 1 is assumed perfect
                         (upper bound — isolates specialist quality)
    """
    device = next(cascade.coarse.parameters()).device
    cascade.eval()

    coarse_correct = fine_correct = oracle_correct = total = 0

    for start in range(0, len(fragments), batch_size):
        batch_frags  = fragments[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        x = torch.from_numpy(batch_frags.astype(np.int64)).to(device)

        with torch.no_grad():
            group_logits = cascade.coarse(x)
            group_preds  = group_logits.argmax(dim=-1)

        for i, (lbl, g_pred_idx) in enumerate(zip(batch_labels, group_preds.tolist())):
            true_group = TYPE_TO_GROUP.get(lbl)
            pred_group = GROUP_NAMES[g_pred_idx]
            coarse_correct += int(pred_group == true_group)
            total += 1

            # Oracle: use true group to route (upper bound for specialist)
            with torch.no_grad():
                oracle_logits = cascade.specialists[true_group](x[i:i+1])
                oracle_pred   = GROUPS[true_group][oracle_logits.argmax(-1).item()]
            oracle_correct += int(oracle_pred == lbl)

            # Real cascade: use predicted group to route
            with torch.no_grad():
                spec_logits = cascade.specialists[pred_group](x[i:i+1])
                fine_pred   = GROUPS[pred_group][spec_logits.argmax(-1).item()]
            fine_correct += int(fine_pred == lbl)

    return {
        "coarse_acc":      coarse_correct / total,
        "fine_acc":        fine_correct   / total,
        "oracle_fine_acc": oracle_correct / total,
        "total_samples":   total,
    }


def evaluate_lazy(
    cascade: HierarchicalCascade,
    frag_path: Path,
    file_indices: np.ndarray,
    labels: list[str],
    sector_size: int,
    batch_size: int = 512,
) -> dict:
    """
    Same metrics as evaluate() but reads fragments on demand via memmap.
    Keeps only one batch of fragments in RAM at a time.
    """
    total_n = frag_path.stat().st_size // sector_size
    mm = np.memmap(frag_path, dtype=np.uint8, mode="r", shape=(total_n, sector_size))

    device = next(cascade.coarse.parameters()).device
    cascade.eval()

    coarse_correct = fine_correct = oracle_correct = total = 0
    n = len(file_indices)

    for start in range(0, n, batch_size):
        end          = min(start + batch_size, n)
        batch_fi     = file_indices[start:end]
        batch_labels = labels[start:end]

        batch_frags = mm[batch_fi]                          # pages in only these rows
        x = torch.from_numpy(batch_frags.astype(np.int64)).to(device)

        with torch.no_grad():
            group_logits = cascade.coarse(x)
            group_preds  = group_logits.argmax(dim=-1)

        for i, (lbl, g_pred_idx) in enumerate(zip(batch_labels, group_preds.tolist())):
            true_group = TYPE_TO_GROUP.get(lbl)
            pred_group = GROUP_NAMES[g_pred_idx]
            coarse_correct += int(pred_group == true_group)
            total += 1

            with torch.no_grad():
                oracle_logits = cascade.specialists[true_group](x[i:i+1])
                oracle_pred   = GROUPS[true_group][oracle_logits.argmax(-1).item()]
            oracle_correct += int(oracle_pred == lbl)

            with torch.no_grad():
                spec_logits = cascade.specialists[pred_group](x[i:i+1])
                fine_pred   = GROUPS[pred_group][spec_logits.argmax(-1).item()]
            fine_correct += int(fine_pred == lbl)

    return {
        "coarse_acc":      coarse_correct / total,
        "fine_acc":        fine_correct   / total,
        "oracle_fine_acc": oracle_correct / total,
        "total_samples":   total,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hierarchical cascade on FFT-75 binary data")
    parser.add_argument("--binary-dir",    type=Path, default=BINARY_DIR,
                        help="Path to binary split directory (default: %(default)s)")
    parser.add_argument("--max-per-class", type=int,   default=None,
                        help="Hard cap on samples per class (preserves balance)")
    parser.add_argument("--fraction",      type=float, default=None,
                        help="Proportional cut, e.g. 0.1 = 10%% of each class")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--no-shared-encoder", action="store_true",
                        help="Give each specialist its own encoder (slower, higher ceiling)")
    parser.add_argument("--lazy", action="store_true",
                        help="Read fragments on demand from disk instead of loading into RAM")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to a JSON file that will be created/updated with per-epoch progress")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        shared_encoder=not args.no_shared_encoder,
    )

    if args.lazy:
        from load_binary_lazy import LazyFragmentDataset, load_split_lazy

        print(f"Loading train split (lazy) from {args.binary_dir} ...")
        train_frag_path, train_file_indices, train_labels, sector = load_split_lazy(
            split="train",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print("Loading val split (lazy) ...")
        val_frag_path, val_file_indices, val_labels, _ = load_split_lazy(
            split="val",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        def _lazy_factory(mode: str):
            return LazyFragmentDataset(
                train_frag_path, sector, train_file_indices, train_labels, mode
            )

        cascade = train_cascade(None, train_labels, cfg, dataset_factory=_lazy_factory,
                                checkpoint_path=args.checkpoint)

        print("\n=== Evaluation on val split ===")
        metrics = evaluate_lazy(cascade, val_frag_path, val_file_indices, val_labels, sector)

    else:
        print(f"Loading train split from {args.binary_dir} ...")
        train_frags, train_labels = load_data(
            split="train",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(train_labels)} fragments, {len(set(train_labels))} classes")

        print("Loading val split ...")
        val_frags, val_labels = load_data(
            split="val",
            max_per_class=args.max_per_class,
            fraction=args.fraction,
            binary_dir=args.binary_dir,
        )
        print(f"  {len(val_labels)} fragments, {len(set(val_labels))} classes")

        cascade = train_cascade(train_frags, train_labels, cfg,
                                checkpoint_path=args.checkpoint)

        print("\n=== Evaluation on val split ===")
        metrics = evaluate(cascade, val_frags, val_labels)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # --- Save model to archive ---
    archive_dir = Path(__file__).parent.parent / "archive"
    archive_dir.mkdir(exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_samples  = len(train_labels)
    fine_acc   = metrics["fine_acc"]
    save_path  = archive_dir / f"cascade_{timestamp}_n{n_samples}_e{args.epochs}_acc{fine_acc:.4f}.pt"
    torch.save(cascade.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
