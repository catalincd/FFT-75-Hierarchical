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
import random
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
NUM_TYPES    = len(ALL_TYPES)                   # 58
SECTOR_SIZE  = 512                              # bytes per fragment

# Global index for each (group, local_idx) pair: GROUP_GLOBAL_INDICES[g] -> (num_local,) tensor
# Used by predict_soft to scatter per-group probabilities back to the global 75-class space.
GROUP_GLOBAL_INDICES: dict[str, list[int]] = {
    g: [TYPE_TO_IDX[t] for t in types] for g, types in GROUPS.items()
}

# ---------------------------------------------------------------------------
# Magic-byte signatures used as self-supervised "is_header" pseudo-labels
# Only the most reliable, unambiguous signatures are listed.  When a fragment's
# first bytes match one of its true class' magic patterns, it is labelled a
# header (1); otherwise non-header (0).  This auxiliary signal teaches the
# encoder to attend to magic-byte regions when present and to fall back on
# distribution-only features when they are absent.
# ---------------------------------------------------------------------------

_HEADER_MAGICS: dict[str, list[bytes]] = {
    "jpg":    [b"\xFF\xD8\xFF"],
    "png":    [b"\x89PNG\r\n\x1a\n"],
    "gif":    [b"GIF87a", b"GIF89a"],
    "bmp":    [b"BM"],
    "tiff":   [b"II*\x00", b"MM\x00*"],
    "webp":   [b"RIFF"],   # also need WEBP at offset 8; we do that check separately
    "pdf":    [b"%PDF-"],
    "doc":    [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"],
    "docx":   [b"PK\x03\x04"],   # OOXML is a zip — partial signal
    "xls":    [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"],
    "xlsx":   [b"PK\x03\x04"],
    "ppt":    [b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"],
    "pptx":   [b"PK\x03\x04"],
    "zip":    [b"PK\x03\x04", b"PK\x05\x06"],
    "gz":     [b"\x1F\x8B"],
    "bz2":    [b"BZh"],
    "7z":     [b"7z\xBC\xAF\x27\x1C"],
    "rar":    [b"Rar!\x1A\x07\x00", b"Rar!\x1A\x07\x01\x00"],
    "elf":    [b"\x7FELF"],
    "exe":    [b"MZ"],
    "dll":    [b"MZ"],
    "class":  [b"\xCA\xFE\xBA\xBE"],
    "mp3":    [b"ID3", b"\xFF\xFB", b"\xFF\xFA", b"\xFF\xF3", b"\xFF\xF2"],
    "flac":   [b"fLaC"],
    "ogg":    [b"OggS"],
    "wav":    [b"RIFF"],
    "m4a":    [b"\x00\x00\x00\x20ftyp", b"\x00\x00\x00\x18ftyp"],
    "aac":    [b"\xFF\xF1", b"\xFF\xF9"],
    "mp4":    [b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00\x20ftyp", b"\x00\x00\x00\x1Cftyp"],
    "mov":    [b"\x00\x00\x00\x14ftyp", b"\x00\x00\x00\x20ftypqt"],
    "mkv":    [b"\x1A\x45\xDF\xA3"],
    "avi":    [b"RIFF"],
    "wmv":    [b"\x30\x26\xB2\x75\x8E\x66\xCF\x11"],
    "flv":    [b"FLV\x01"],
    "sqlite": [b"SQLite format 3\x00"],
    "ttf":    [b"\x00\x01\x00\x00", b"true", b"OTTO"],
    "ps":     [b"%!PS"],
    "eps":    [b"%!PS-Adobe", b"\xC5\xD0\xD3\xC6"],
    "psd":    [b"8BPS"],
    "swf":    [b"FWS", b"CWS", b"ZWS"],
    "iso":    [],   # CD001 is at offset 0x8001 — outside our 512-byte window
    "img":    [],
    "vmdk":   [b"KDMV", b"# Disk DescriptorFile"],
    "db":     [b"SQLite format 3\x00"],   # often sqlite-backed
    "mdb":    [b"\x00\x01\x00\x00Standard Jet DB"],
    "cr2":    [b"II*\x00\x10\x00\x00\x00CR"],
    "nef":    [b"MM\x00*"],
    "arw":    [b"II*\x00"],
    "dng":    [b"II*\x00"],
    "orf":    [b"IIRO", b"MMOR"],
    # text formats have no fixed magic; left empty so they always get is_header=0
}

# Build a per-class-index list of magic-byte tensors for fast batched checking on GPU.
# For each true class index, store list of (magic_bytes, length) pairs.
def _build_magic_table() -> list[list[bytes]]:
    return [_HEADER_MAGICS.get(t, []) for t in ALL_TYPES]

_MAGIC_TABLE: list[list[bytes]] = _build_magic_table()

# Maximum magic length across all classes — used to bound how many bytes to inspect.
_MAX_MAGIC_LEN = max((len(m) for ms in _MAGIC_TABLE for m in ms), default=8)


def detect_header_pseudo_labels(
    x: torch.Tensor,
    fine_label_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a binary header/non-header pseudo-label per sample.

    A fragment is labelled a header (1) iff its first bytes match one of its
    true class' known magic-byte signatures, otherwise non-header (0).
    Performed on CPU because byte-exact matching is awkward in tensor ops and
    the cost (~few hundred µs per batch) is negligible vs the encoder forward.

    Args:
        x:                    (B, L) byte values, any device
        fine_label_indices:   (B,)   int64 fine-grained class indices (0..NUM_TYPES-1)

    Returns:
        is_header:            (B,)   int64 ∈ {0, 1} on the same device as x
    """
    B = x.shape[0]
    head = x[:, :_MAX_MAGIC_LEN].detach().to(torch.uint8).cpu().numpy().tobytes()
    # head is concatenated rows, each of length _MAX_MAGIC_LEN
    out = np.zeros(B, dtype=np.int64)
    fine_idx_cpu = fine_label_indices.detach().cpu().numpy()
    for i in range(B):
        magics = _MAGIC_TABLE[int(fine_idx_cpu[i])]
        if not magics:
            continue
        row = head[i * _MAX_MAGIC_LEN : (i + 1) * _MAX_MAGIC_LEN]
        for m in magics:
            if row[: len(m)] == m:
                out[i] = 1
                break
    return torch.from_numpy(out).to(x.device)

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
# Byte2Image intrabyte branch (ByteNet, IEEE TMM 2025 — arXiv:2410.20855)
# ---------------------------------------------------------------------------

def _bytes_to_bit_patches(x: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
    """
    (B, 512) int64 → (B, n_patches, patch_size²) float32.

    Expands each byte into its 8 constituent bits (MSB first), forming a
    512×8 = 4096-bit sequence.  This is reshaped into a 64×64 binary "image"
    and then divided into non-overlapping patch_size×patch_size tiles.

    The 2D layout is meaningful: each row corresponds to one byte's bit pattern,
    so row-aligned attention captures intra-byte structure while column-aligned
    attention captures alignment across consecutive bytes.
    """
    B = x.shape[0]
    # Use only the first 512 bytes regardless of sector size; the bit-image
    # layout is always 512*8 = 4096 bits → 64×64.
    x      = x[:, :512]
    shifts = torch.arange(7, -1, -1, device=x.device)       # [7,6,...,0]
    bits   = ((x.long().unsqueeze(-1) >> shifts) & 1).float()  # (B, 512, 8)
    bits   = bits.reshape(B, 64, 64)                         # (B, 64, 64)
    p      = patch_size
    n      = 64 // p
    # unfold both spatial dims to extract non-overlapping p×p patches
    patches = bits.unfold(1, p, p).unfold(2, p, p)           # (B, n, n, p, p)
    return patches.reshape(B, n * n, p * p)                  # (B, n², p²)


class Byte2ImageBranch(nn.Module):
    """
    ByteNet intrabyte branch: 512 bytes → 64×64 bit image → tiny ViT → out_dim.

    The CNN backbone treats bytes as discrete tokens via Embedding(256, d).
    It captures which byte *values* appear but is blind to the bit-level
    structure within each byte.  This branch exposes that structure:

      • DEFLATE (zip/gz): Huffman-coded bitstreams have characteristic entropy
        patterns detectable at the bit column level.
      • BWT (bz2): block-sorting produces local bit-run clustering.
      • LZMA (7z): range-coded streams have distinct bit transition statistics.
      • JSON/XML/CSV: ASCII printable bytes all share bit 7 = 0, bit 6 = 1,
        and differ in the lower 6 bits — patterns a byte-value CNN must learn
        indirectly but a bit-image ViT sees immediately.

    Architecture (IMG_SIZE=64, PATCH_SIZE=8):
        4096 bits → 64×64 image → 64 non-overlapping 8×8 patches
        → PatchEmbed (64 → embed_dim) + CLS token + learned pos embedding
        → N × TransformerEncoderLayer (pre-norm, GELU, mlp_ratio=2)
        → CLS token → LayerNorm → Linear → out_dim
    """
    IMG_SIZE   = 64    # sqrt(512 * 8) = 64
    PATCH_SIZE = 8     # 8×8 patches → (64/8)² = 64 tokens

    def __init__(
        self,
        out_dim:   int = 256,
        embed_dim: int = 128,
        n_layers:  int = 4,
        n_heads:   int = 4,
    ):
        super().__init__()
        self.out_dim  = out_dim
        n_patches     = (self.IMG_SIZE // self.PATCH_SIZE) ** 2   # 64

        self.patch_embed = nn.Linear(self.PATCH_SIZE ** 2, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = n_heads,
            dim_feedforward = embed_dim * 2,   # mlp_ratio=2 keeps params light
            dropout         = 0.1,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,            # pre-norm for stable deep training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(embed_dim)
        self.proj        = nn.Linear(embed_dim, out_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = _bytes_to_bit_patches(x, self.PATCH_SIZE)          # (B, 64, 64)
        tokens  = self.patch_embed(patches)                           # (B, 64, embed_dim)
        cls     = self.cls_token.expand(x.shape[0], -1, -1)          # (B, 1, embed_dim)
        tokens  = torch.cat([cls, tokens], dim=1) + self.pos_embed   # (B, 65, embed_dim)
        out     = self.transformer(tokens)
        return self.proj(self.norm(out[:, 0]))                        # (B, out_dim)


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
    Set use_b2i=True to add the Byte2ImageBranch (+ b2i_dim, default 256).
    """
    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        grad_checkpoint: bool = False,
        use_b2i:         bool = False,
        b2i_dim:         int  = 256,
    ):
        super().__init__()
        self.byte_enc   = ByteEncoder(embed_dim, num_filters, grad_checkpoint)
        self.bigram_enc = BigramBranch(out_dim=bigram_dim)
        self.b2i_enc    = Byte2ImageBranch(out_dim=b2i_dim) if use_b2i else None
        self.out_dim    = self.byte_enc.out_dim + bigram_dim + (b2i_dim if use_b2i else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) int64 byte values
        cnn_feat    = self.byte_enc(x)                       # (B, 1024)
        bigram_feat = self.bigram_enc(x)                     # (B, 512)
        parts = [cnn_feat, bigram_feat]
        if self.b2i_enc is not None:
            parts.append(self.b2i_enc(x))                   # (B, b2i_dim)
        return torch.cat(parts, dim=-1)


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

      4. Intrabyte bit-level patterns (ByteNet, IEEE TMM 2025).
         DEFLATE, BWT, and LZMA have characteristic Huffman/range-coding bit
         patterns detectable at the individual-bit level — invisible to a
         byte-value CNN but directly exploitable by a ViT on the bit image.
         → Byte2ImageBranch: 512 bytes → 64×64 bit image → tiny ViT → b2i_dim

    Keeps `byte_enc` and `bigram_enc` as top-level attributes so
    `_load_encoder_from_phase1` works without modification.

    out_dim: 1024 (CNN) + 512 (bigram) + 64 (header) + 128 (hist) + 256 (b2i) = 1984
    """
    HEADER_BYTES = 64   # first N bytes of the fragment; covers all magic signatures

    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        header_dim:      int  = 64,
        hist_dim:        int  = 128,
        b2i_dim:         int  = 256,
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

        # Bit-image branch — captures intrabyte entropy patterns per algorithm
        self.b2i_enc = Byte2ImageBranch(out_dim=b2i_dim)

        self.out_dim = self.byte_enc.out_dim + bigram_dim + header_dim + hist_dim + b2i_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat    = self.byte_enc(x)                                # (B, 1024)
        bigram_feat = self.bigram_enc(x)                              # (B, 512)

        header      = x[:, :self.HEADER_BYTES].long()                # (B, 64)
        header_feat = self.header_mlp(
            self.header_embed(header).flatten(1)                      # (B, 64*8)
        )                                                             # (B, header_dim)

        hist_feat   = self.hist_mlp(_byte_hist(x))                   # (B, hist_dim)
        b2i_feat    = self.b2i_enc(x)                                 # (B, b2i_dim)

        return torch.cat([cnn_feat, bigram_feat, header_feat, hist_feat, b2i_feat], dim=-1)


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

_TEXT_BIGRAMS: list[tuple[int, int]] = [
    (ord('"'), ord(':')),   # "key": — JSON key-value (never in CSV)
    (ord('}'), ord(',')),   # }, — JSON object in array
    (ord(']'), ord(',')),   # ], — JSON nested array element
    (ord('<'), ord('/')),   # </tag> — XML/HTML closing tag
    (ord('='), ord('"')),   # attr=" — XML/HTML attribute value
    (ord('/'), ord('>')),   # /> — XML/HTML self-closing tag
    (ord('<'), ord('!')),   # <!-- — HTML comment/DOCTYPE
    (ord(','), 10),         # ,\n — CSV row-end
    (10, ord('"')),         # \n" — CSV quoted field at line start
    (ord(':'), ord(' ')),   # ": " — log timestamps & JSON values
]   # 10 bigrams


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

      3. Intrabyte bit-level patterns (ByteNet, IEEE TMM 2025).
         ASCII printable bytes all have bit-7 = 0 and bit-6 = 1; the lower
         6 bits encode the character.  Text formats that heavily use specific
         character ranges (e.g. JSON digits/punctuation vs XML alpha tags)
         produce distinct patterns in the lower-bit columns of the bit image.
         → Byte2ImageBranch: 512 bytes → 64×64 bit image → tiny ViT → b2i_dim

    Keeps `byte_enc` and `bigram_enc` as top-level attributes so
    `_load_encoder_from_phase1` works without modification.

    out_dim: 1024 (CNN) + 512 (bigram) + 64 (struct) + 256 (b2i) = 1856
    """

    def __init__(
        self,
        embed_dim:       int  = 16,
        num_filters:     int  = 128,
        bigram_dim:      int  = 512,
        struct_dim:      int  = 64,
        b2i_dim:         int  = 256,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.byte_enc   = ByteEncoder(embed_dim, num_filters, grad_checkpoint)
        self.bigram_enc = BigramBranch(out_dim=bigram_dim)

        n_uni = len(_TEXT_STRUCT_CHARS)
        n_bi  = len(_TEXT_BIGRAMS)
        # Register as buffers so tensors move to the correct device automatically
        self.register_buffer(
            "_struct_chars",
            torch.tensor(_TEXT_STRUCT_CHARS, dtype=torch.long),
        )
        bigram_a = [a for a, _ in _TEXT_BIGRAMS]
        bigram_b = [b for _, b in _TEXT_BIGRAMS]
        self.register_buffer("_bigram_a", torch.tensor(bigram_a, dtype=torch.long))
        self.register_buffer("_bigram_b", torch.tensor(bigram_b, dtype=torch.long))

        # Structural branch — (B, n_uni + n_bi) frequency vector → struct_dim
        self.struct_mlp = nn.Sequential(
            nn.Linear(n_uni + n_bi, 128),
            nn.GELU(),
            nn.Linear(128, struct_dim),
        )

        # Bit-image branch — captures ASCII bit-column structure per format
        self.b2i_enc = Byte2ImageBranch(out_dim=b2i_dim)

        self.out_dim = self.byte_enc.out_dim + bigram_dim + struct_dim + b2i_dim

    def _struct_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Unigram character occurrence rates. x: (B, L) → (B, n_uni)"""
        return (x.unsqueeze(-1) == self._struct_chars).float().mean(dim=1)

    def _bigram_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Consecutive-pair occurrence rates. x: (B, L) → (B, n_bi)"""
        match_a = (x[:, :-1].unsqueeze(-1) == self._bigram_a)
        match_b = (x[:, 1:].unsqueeze(-1)  == self._bigram_b)
        return (match_a & match_b).float().mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat    = self.byte_enc(x)                                              # (B, 1024)
        bigram_feat = self.bigram_enc(x)                                            # (B, 512)
        uni_freq    = self._struct_freq(x)                                          # (B, 23)
        bi_freq     = self._bigram_freq(x)                                          # (B, 10)
        struct_feat = self.struct_mlp(torch.cat([uni_freq, bi_freq], dim=-1))       # (B, struct_dim)
        b2i_feat    = self.b2i_enc(x)                                               # (B, b2i_dim)
        return torch.cat([cnn_feat, bigram_feat, struct_feat, b2i_feat], dim=-1)


# ---------------------------------------------------------------------------
# Stage 1: Coarse Group Classifier
# ---------------------------------------------------------------------------

class CoarseClassifier(nn.Module):
    """
    Predicts one of NUM_GROUPS coarse groups, with an optional auxiliary
    binary "is_header" position head trained jointly via multitask loss.

    Over the baseline 3-layer BN-MLP:
      - LayerNorm: per-sample, no batch-stat sensitivity; cascade routing often
        processes small per-group sub-batches after Stage 1 argmax
      - GeGLU (Noam et al., 2020): the gate learns to suppress encoder channels
        irrelevant to group separation, outperforming plain GELU on projection
        layers; costs the same FLOPs as the original first Linear
      - Single residual hidden layer: Scenario #2 (11 classes) is well-separated
        by the encoder; the third BN-MLP layer added depth without benefit
      - Skip projection: gradient highway so the encoder is never a bottleneck

    Position head (optional):
      A small 2-class head sharing the encoder feature.  Supervised by
      magic-byte pseudo-labels (see detect_header_pseudo_labels).  Forces the
      encoder to expose information about whether the fragment starts at a
      file boundary, which improves group routing for header-heavy formats
      (zip, gz, ELF, MZ, …) and makes the encoder explicitly position-aware.
    """
    def __init__(self, encoder: ByteEncoder, with_position_head: bool = False):
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

        # Optional auxiliary head: 2-class is_header.  Tiny (a single MLP) so
        # most parameters remain in the main head; the encoder feature is shared.
        self.position_head: Optional[nn.Module] = None
        if with_position_head:
            self.position_head = nn.Sequential(
                nn.Linear(d, hidden, bias=False),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, 2),
            )

    def _shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward + LayerNorm — shared between main and position heads."""
        return self.norm(self.encoder(x))

    def _main_logits(self, feat: torch.Tensor) -> torch.Tensor:
        v, g = self.geglu_proj(feat).chunk(2, dim=-1)        # each (B, hidden)
        h    = self.drop(v * F.gelu(g)) + self.skip(feat)   # GeGLU + residual
        return self.head(self.out_norm(h))                   # (B, NUM_GROUPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._main_logits(self._shared_features(x))

    def forward_with_position(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (group_logits, position_logits) where position_logits is None
        if no position head was attached.  Used during training when a
        position pseudo-label is available; inference paths can use the
        plain forward() to skip the auxiliary head.
        """
        feat = self._shared_features(x)
        main = self._main_logits(feat)
        pos  = self.position_head(feat) if self.position_head is not None else None
        return main, pos

# ---------------------------------------------------------------------------
# Stage 2: Per-Group Fine-Grained Specialist
# One specialist per group; each only predicts within its own class set.
# ---------------------------------------------------------------------------

class SpecialistClassifier(nn.Module):
    """
    Predicts the fine-grained type within a single group.  When
    `with_rejection=True`, an extra output channel is added at index
    `num_classes` representing "out-of-group".  This serves two purposes:

      1. Calibration for soft-routing inference: the specialist learns to
         output low in-group probability mass for inputs that don't belong to
         its group.  Marginalising P(g|x) · P(c|x,g) across all groups then
         gives a coherent global posterior over the 75 classes.

      2. Robustness to coarse-routing errors: when the coarse classifier
         mis-routes a sample, the wrong specialist now has the option of
         flagging "this isn't mine" rather than guessing one of its in-group
         labels.

    Head architecture is shared: LayerNorm → Linear → GELU → Dropout → residual
    skip → LayerNorm → head.  Only the final head's output dimension changes.
    """
    def __init__(
        self,
        encoder: ByteEncoder,
        num_classes: int,
        with_rejection: bool = False,
    ):
        super().__init__()
        self.encoder        = encoder
        self.num_classes    = num_classes
        self.with_rejection = with_rejection
        d             = encoder.out_dim
        hidden        = d // 4              # 384 for FusedEncoder (1536 // 4)
        self.norm     = nn.LayerNorm(d)
        self.fc1      = nn.Linear(d, hidden, bias=False)
        self.drop     = nn.Dropout(0.3)
        self.skip     = nn.Linear(d, hidden, bias=False)
        self.out_norm = nn.LayerNorm(hidden)
        out_dim       = num_classes + (1 if with_rejection else 0)
        self.head     = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.norm(self.encoder(x))                   # (B, d)
        h    = self.drop(F.gelu(self.fc1(feat))) + self.skip(feat)  # (B, hidden)
        return self.head(self.out_norm(h))                  # (B, num_classes [+1])


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
    def __init__(
        self,
        shared_encoder:     bool = True,
        coarse_use_b2i:     bool = True,
        coarse_position_head: bool = True,
        specialists_with_rejection: bool = True,
    ):
        super().__init__()

        if shared_encoder:
            encoder = FusedEncoder(use_b2i=coarse_use_b2i)
            self.coarse = CoarseClassifier(encoder, with_position_head=coarse_position_head)
            self.specialists = nn.ModuleDict({
                group: SpecialistClassifier(
                    encoder, len(types), with_rejection=specialists_with_rejection
                )
                for group, types in GROUPS.items()
            })
        else:
            self.coarse = CoarseClassifier(
                FusedEncoder(use_b2i=coarse_use_b2i),
                with_position_head=coarse_position_head,
            )
            self.specialists = nn.ModuleDict({
                group: SpecialistClassifier(
                    FusedEncoder(use_b2i=coarse_use_b2i), len(types),
                    with_rejection=specialists_with_rejection,
                )
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
        Full cascade inference (hard routing — argmax at coarse stage).

        Returns:
            predictions: list of predicted fine-grained type strings, length B
            confidence:  (B, 2) tensor of (group_conf, type_conf) if requested
        """
        # Stage 1: predict group
        group_logits = self.coarse(x)                       # (B, 11)
        group_probs  = F.softmax(group_logits, dim=-1)
        group_preds  = group_logits.argmax(dim=-1)          # (B,)

        predictions: list[Optional[str]] = [None] * len(x)
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
            type_logits = specialist(x_sub)                 # (k, num_types [+1])
            num_local   = len(GROUPS[group_name])
            # Drop the rejection class (last channel) if present — it must not
            # be predicted at hard-cascade inference time.
            in_group_logits = type_logits[:, :num_local]
            type_probs      = F.softmax(in_group_logits, dim=-1)
            local_preds     = in_group_logits.argmax(dim=-1)

            type_confs[mask] = type_probs.max(dim=-1).values
            local_types = GROUPS[group_name]
            for i, sample_idx in enumerate(mask.tolist()):
                predictions[sample_idx] = local_types[local_preds[i].item()]

        if return_confidence:
            conf = torch.stack([group_confs, type_confs], dim=-1)  # (B, 2)
            return predictions, conf

        return predictions, None

    @torch.no_grad()
    def predict_soft(
        self,
        x:     torch.Tensor,
        top_k: int = 3,
    ) -> tuple[list[str], torch.Tensor]:
        """
        Mixture-of-Experts inference: marginalise over top-K coarse groups.

        For each sample we compute
            P(c | x) = Σ_g P(g | x) · P(c | x, g)     for c ∈ group g
        restricted to the top-K coarse groups (re-normalised) for efficiency.
        When specialists carry a rejection class, the in-group probability
        mass is left as-is (not re-normalised) so out-of-group inputs that
        the specialist correctly rejects contribute negligibly.

        This recovers most of the cascade routing error: when the true group
        is the coarse classifier's #2 or #3 candidate (instead of #1), the
        joint score for the true class can still win.

        Args:
            x:     (B, L) byte values
            top_k: number of coarse candidates to score per sample (1..NUM_GROUPS)

        Returns:
            predictions:  list of NUM_TYPES-string predictions, length B
            full_probs:   (B, NUM_TYPES) marginalised posterior
        """
        assert 1 <= top_k <= NUM_GROUPS
        B = x.shape[0]
        device = x.device

        coarse_logits = self.coarse(x)                       # (B, NUM_GROUPS)
        coarse_probs  = F.softmax(coarse_logits, dim=-1)

        # Top-K group probabilities, re-normalised so they sum to 1.
        topk_probs, topk_idx = coarse_probs.topk(top_k, dim=-1)      # (B, K)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        full_probs = torch.zeros(B, NUM_TYPES, device=device)

        # For each group, run the specialist once on all samples that have it
        # in their top-K, weight by P(g|x), and scatter into the global slots.
        for g_idx, g_name in enumerate(GROUP_NAMES):
            in_topk = (topk_idx == g_idx)                            # (B, K) bool
            sample_in_topk = in_topk.any(dim=-1)                     # (B,)
            if not sample_in_topk.any():
                continue

            sel = sample_in_topk.nonzero(as_tuple=True)[0]           # (M,)
            x_sub = x[sel]
            spec_logits = self.specialists[g_name](x_sub)
            num_local   = len(GROUPS[g_name])
            spec_probs  = F.softmax(spec_logits, dim=-1)
            # In-group mass only (rejection slot, if present, is dropped here
            # but its softmax presence already dampened the in-group magnitudes).
            local_probs = spec_probs[:, :num_local]                  # (M, num_local)

            # Per-sample weight = P(g|x) if g is in this sample's top-K, else 0.
            # Sum across the K axis (row has at most one True for this g).
            weights = (topk_probs[sel] * in_topk[sel].float()).sum(dim=-1)  # (M,)
            weighted = local_probs * weights.unsqueeze(-1)           # (M, num_local)

            global_indices = torch.tensor(
                GROUP_GLOBAL_INDICES[g_name], device=device, dtype=torch.long
            )                                                        # (num_local,)
            # Add to the right global columns for the right rows
            full_probs[sel.unsqueeze(-1), global_indices.unsqueeze(0)] += weighted

        pred_idx = full_probs.argmax(dim=-1).tolist()
        predictions = [ALL_TYPES[i] for i in pred_idx]
        return predictions, full_probs

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

def _is_header_for_label(raw: bytes, label_str: str) -> int:
    """1 iff the raw bytes start with one of label_str's known magics, else 0."""
    magics = _HEADER_MAGICS.get(label_str)
    if not magics:
        return 0
    for m in magics:
        if raw[:len(m)] == m:
            return 1
    return 0


class FragmentDataset(Dataset):
    """
    Dataset wrapper for FFT-75 style data with optional byte-level augmentation.

    Expected: fragments as (N, 512) uint8 numpy array,
              fine-grained labels as list of type strings length N.

    Modes:
        "coarse"             — y is the group index (0..NUM_GROUPS-1)
        "specialist:<group>" — y is the local fine-grained class index within
                               the group; or `num_local` (the rejection slot)
                               for rejection-mixed samples.

    Augmentation (training only, set augment=True):
        Byte noise — randomly replaces `noise_prob` fraction of bytes with
        uniformly random values (0-255) each time a sample is fetched.

    Multi-task heads (optional, additive return values):
        with_position=True   — emit a per-sample binary is_header pseudo-label
                               (computed via _is_header_for_label).  Coarse
                               mode only.  Returned as the third item of the
                               tuple; ignored by the training loop when the
                               coarse classifier has no position head.
        rejection_prob > 0   — Specialist mode only.  With this probability
                               each fetched sample is replaced by an
                               out-of-group fragment labelled with the
                               rejection class index (= num_local).  Trains
                               the specialist to flag inputs that don't
                               belong to its group.
    """
    def __init__(
        self,
        fragments:      np.ndarray,
        labels:         list[str],
        mode:           str   = "coarse",
        augment:        bool  = False,
        noise_prob:     float = 0.02,
        with_position:  bool  = False,
        rejection_prob: float = 0.0,
    ):
        assert len(fragments) == len(labels)
        assert mode == "coarse" or mode.startswith("specialist:")
        if with_position and mode != "coarse":
            raise ValueError("with_position only supported in coarse mode")
        if rejection_prob > 0 and mode == "coarse":
            raise ValueError("rejection_prob only supported in specialist mode")

        self.mode           = mode
        self.augment        = augment
        self.noise_prob     = noise_prob
        self.with_position  = with_position
        self.rejection_prob = rejection_prob
        # Use Python's `random` module — it is auto-reseeded per DataLoader worker.

        target_group = mode.split(":")[1] if ":" in mode else None

        if target_group is not None:
            keep = [
                i for i, lbl in enumerate(labels)
                if TYPE_TO_GROUP.get(lbl) == target_group
            ]
            self.fragments    = fragments[keep]
            self.labels       = [labels[i] for i in keep]
            self.label_map    = GROUP_LOCAL_IDX[target_group]
            self.num_local    = len(GROUPS[target_group])
            self.target_group = target_group

            if rejection_prob > 0:
                # Out-of-group pool: every sample whose group ≠ target.
                # Stored as a view into the original fragments array (no copy).
                out_keep = [
                    i for i, lbl in enumerate(labels)
                    if TYPE_TO_GROUP.get(lbl) != target_group
                ]
                if not out_keep:
                    raise RuntimeError(
                        f"rejection_prob>0 but no out-of-group samples available "
                        f"for group {target_group}"
                    )
                self._out_fragments = fragments[out_keep]
            else:
                self._out_fragments = None
        else:
            self.fragments    = fragments
            self.labels       = labels
            self.label_map    = GROUP_TO_IDX
            self.num_local    = None
            self.target_group = None
            self._out_fragments = None

    def __len__(self) -> int:
        return len(self.fragments)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.augment and self.noise_prob > 0:
            mask  = torch.rand(x.shape) < self.noise_prob
            noise = torch.randint_like(x, 0, 256)
            x     = torch.where(mask, noise, x)
        return x

    def __getitem__(self, idx: int):
        # Rejection-class injection (specialist mode only).
        if self._out_fragments is not None and random.random() < self.rejection_prob:
            out_idx = random.randrange(len(self._out_fragments))
            x = torch.from_numpy(self._out_fragments[out_idx].astype(np.int64))
            x = self._augment(x)
            y = self.num_local                          # rejection class index
            return x, y

        x = torch.from_numpy(self.fragments[idx].astype(np.int64))
        x = self._augment(x)

        if self.mode == "coarse":
            label_str = self.labels[idx]
            y = self.label_map[TYPE_TO_GROUP[label_str]]
            if self.with_position:
                # is_header is computed on the *clean* fragment header before
                # augmentation could corrupt the magic bytes.
                raw = bytes(self.fragments[idx][:_MAX_MAGIC_LEN].tolist())
                is_header = _is_header_for_label(raw, label_str)
                return x, y, is_header
            return x, y

        # specialist mode (no rejection draw)
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

def _evaluate_batches(
    cascade: HierarchicalCascade,
    iter_batches,                       # iterable of (x_tensor, labels_list)
    soft_top_k: int = 3,
) -> dict:
    """
    Shared evaluation engine for both in-RAM and lazy variants.

    Reports:
      - coarse_acc:        Stage 1 top-1 group accuracy
      - coarse_topk_acc:   Stage 1 top-K group accuracy (K=soft_top_k); upper
                           bound on what soft routing can recover
      - fine_acc:          end-to-end accuracy with hard cascade (argmax)
      - soft_fine_acc:     end-to-end accuracy with soft (top-K MoE) routing
      - oracle_fine_acc:   fine accuracy with perfect coarse routing
    """
    device = next(cascade.coarse.parameters()).device
    cascade.eval()

    coarse_correct = coarse_topk_correct = 0
    fine_correct = soft_correct = oracle_correct = total = 0

    for x, batch_labels in iter_batches:
        x = x.to(device)
        B = x.shape[0]

        with torch.no_grad():
            group_logits = cascade.coarse(x)                  # (B, 11)
            group_preds  = group_logits.argmax(dim=-1)        # (B,)
            _, topk_idx  = group_logits.topk(soft_top_k, dim=-1)  # (B, K)

            # Soft routing predictions for the whole batch in one shot
            soft_preds, _ = cascade.predict_soft(x, top_k=soft_top_k)

        # True group indices per sample (-1 if label is missing from mapping)
        true_group_idx = torch.tensor([
            GROUP_TO_IDX.get(TYPE_TO_GROUP.get(lbl, ""), -1)
            for lbl in batch_labels
        ], device=device)

        coarse_correct      += (group_preds == true_group_idx).sum().item()
        coarse_topk_correct += (topk_idx == true_group_idx.unsqueeze(-1)).any(dim=-1).sum().item()

        # Oracle (true group → its specialist)
        for true_g_name in set(TYPE_TO_GROUP[l] for l in batch_labels):
            mask = [i for i, l in enumerate(batch_labels) if TYPE_TO_GROUP[l] == true_g_name]
            if not mask:
                continue
            mask_t = torch.tensor(mask, device=device)
            with torch.no_grad():
                oracle_logits = cascade.specialists[true_g_name](x[mask_t])
            num_local = len(GROUPS[true_g_name])
            oracle_pred_idx = oracle_logits[:, :num_local].argmax(dim=-1).tolist()
            local_types = GROUPS[true_g_name]
            oracle_correct += sum(
                int(local_types[oracle_pred_idx[k]] == batch_labels[mask[k]])
                for k in range(len(mask))
            )

        # Hard cascade — group by predicted group for batched specialist forward
        for g_idx, g_name in enumerate(GROUP_NAMES):
            sel = (group_preds == g_idx).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            with torch.no_grad():
                spec_logits = cascade.specialists[g_name](x[sel])
            num_local = len(GROUPS[g_name])
            local_pred_idx = spec_logits[:, :num_local].argmax(dim=-1).tolist()
            local_types = GROUPS[g_name]
            sel_list = sel.tolist()
            for k, sample_idx in enumerate(sel_list):
                if local_types[local_pred_idx[k]] == batch_labels[sample_idx]:
                    fine_correct += 1

        # Soft routing accuracy
        for i, lbl in enumerate(batch_labels):
            if soft_preds[i] == lbl:
                soft_correct += 1

        total += B

    return {
        "coarse_acc":       coarse_correct      / total,
        "coarse_topk_acc":  coarse_topk_correct / total,
        "fine_acc":         fine_correct        / total,
        "soft_fine_acc":    soft_correct        / total,
        "oracle_fine_acc":  oracle_correct      / total,
        "total_samples":    total,
        "soft_top_k":       soft_top_k,
    }


def evaluate(
    cascade:    HierarchicalCascade,
    fragments:  np.ndarray,
    labels:     list[str],
    batch_size: int = 512,
    soft_top_k: int = 3,
) -> dict:
    """
    In-RAM evaluation.  See _evaluate_batches for metric definitions.
    """
    def batches():
        for start in range(0, len(fragments), batch_size):
            x = torch.from_numpy(fragments[start:start + batch_size].astype(np.int64))
            yield x, labels[start:start + batch_size]
    return _evaluate_batches(cascade, batches(), soft_top_k=soft_top_k)


def evaluate_lazy(
    cascade:      HierarchicalCascade,
    frag_path:    Path,
    file_indices: np.ndarray,
    labels:       list[str],
    sector_size:  int,
    batch_size:   int = 512,
    soft_top_k:   int = 3,
) -> dict:
    """
    Same metrics as evaluate() but reads fragments on demand via memmap.
    Keeps only one batch of fragments in RAM at a time.
    """
    total_n = frag_path.stat().st_size // sector_size
    mm = np.memmap(frag_path, dtype=np.uint8, mode="r", shape=(total_n, sector_size))

    def batches():
        for start in range(0, len(file_indices), batch_size):
            end      = min(start + batch_size, len(file_indices))
            batch_fi = file_indices[start:end]
            x = torch.from_numpy(np.ascontiguousarray(mm[batch_fi]).astype(np.int64))
            yield x, labels[start:end]

    return _evaluate_batches(cascade, batches(), soft_top_k=soft_top_k)


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
