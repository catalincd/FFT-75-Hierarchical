"""
LiteCoarseClassifier — small-but-strong student for the 11-group coarse split.

Backbone
--------
  Depthwise-separable 1D CNN (3 stages, 32 -> 64 -> 128 ch) ->
  2-layer Bi-GRU over the pooled sequence ->
  additive attention pool over time -> 256-d sequence feature.

The BiGRU + attention pool sit at the position-level — they capture the
*rhythm* of byte structure (tag bursts, csv row cadence, magic-header
locality), which the structural-character branch alone cannot see. This is
the key add for the text<->archive bottleneck.

Side branches (cheap, position-free, same idea as before):
  - byte histogram          — 256-d unigram fingerprint
  - block Shannon entropy   — 16 blocks + (mean, std, min, max) summary
  - structural-char freqs   — 23 unigram + 10 bigram structural rates

Concat (seq + hist + entropy + struct) -> LayerNorm -> Linear -> head.

Parameter count at defaults: ~380k (vs. ~10M for the heavy CoarseEncoder).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_GROUPS = 11

_STRUCT_CHARS: list[int] = [
    ord('{'), ord('}'), ord('['), ord(']'), ord(':'),
    ord('<'), ord('>'), ord('/'), ord('='), ord('&'),
    ord('"'), ord("'"),
    ord(','), ord(';'), ord('|'),
    ord('\n'), ord('\r'), ord('\t'),
    ord('#'), ord('@'), ord('\\'), ord('('), ord(')'), ord('!'),
]
_STRUCT_BIGRAMS: list[tuple[int, int]] = [
    (ord('"'), ord(':')),
    (ord('}'), ord(',')),
    (ord(']'), ord(',')),
    (ord('<'), ord('/')),
    (ord('='), ord('"')),
    (ord('/'), ord('>')),
    (ord('<'), ord('!')),
    (ord(','), 10),
    (10, ord('"')),
    (ord(':'), ord(' ')),
]


# ---------------------------------------------------------------------------
# Depthwise-separable conv block — the workhorse of the lite backbone.
# ---------------------------------------------------------------------------

class DSConvBlock(nn.Module):
    """Depthwise 1D conv + pointwise 1x1; cheap residual, BN, GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw   = nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch)
        self.pw   = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.pw(self.dw(x))
        x = self.act(self.bn(x))
        return x + residual


# ---------------------------------------------------------------------------
# Additive attention pool — learned query over the BiGRU output sequence.
# ---------------------------------------------------------------------------

class AttentionPool1d(nn.Module):
    """One scalar score per time-step, softmax, weighted sum."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        w = torch.softmax(self.score(x), dim=1)            # (B, T, 1)
        return (x * w).sum(dim=1)                          # (B, d)


# ---------------------------------------------------------------------------
# Cheap statistics — all O(L) and fully vectorised.
# ---------------------------------------------------------------------------

def _byte_hist(x: torch.Tensor) -> torch.Tensor:
    B, L = x.shape
    hist = torch.zeros(B, 256, device=x.device, dtype=torch.float32)
    hist.scatter_add_(1, x.long(), torch.ones(B, L, device=x.device))
    return hist / L


def _block_entropy(x: torch.Tensor, n_blocks: int) -> torch.Tensor:
    B, L = x.shape
    blk  = L // n_blocks
    xb   = x[:, :blk * n_blocks].reshape(B, n_blocks, blk).long()
    hist = torch.zeros(B, n_blocks, 256, device=x.device, dtype=torch.float32)
    hist.scatter_add_(2, xb, torch.ones_like(xb, dtype=torch.float32))
    p   = hist / blk
    ent = -torch.special.xlogy(p, p).sum(dim=-1) / math.log(2)
    return ent / 8.0


# ---------------------------------------------------------------------------
# LiteCoarseClassifier — student model.
# ---------------------------------------------------------------------------

class LiteCoarseClassifier(nn.Module):
    """
    Three-stage DS-CNN  ->  BiGRU(2 layers)  ->  attention pool  ->  seq feat.
    Concat with histogram + entropy + structural-char branches  ->  head.

    Defaults (sector size 512):
      Stage layout       channels        seq_len after pool
      stem (k=7)         32              512
      block1 (k=5) + p4  32              128
      block2 (k=3) + p4  64              32
      block3 (k=3)       128             32
      BiGRU(2 layers, hidden 96, bidir): 32 -> 192-d per step
      AttentionPool1d:   192-d, single vector
    """

    SECTOR_SIZE = 512

    def __init__(
        self,
        embed_dim:        int = 12,
        cnn_channels:     tuple[int, int, int] = (32, 64, 128),
        gru_hidden:       int = 96,
        gru_layers:       int = 2,
        gru_dropout:      float = 0.2,
        n_entropy_blocks: int = 16,
        head_hidden:      int = 192,
        dropout:          float = 0.3,
    ):
        super().__init__()
        self.n_blocks = n_entropy_blocks

        # ---- byte embedding + tiny CNN ----
        self.embed = nn.Embedding(256, embed_dim)
        c1, c2, c3 = cnn_channels
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, c1, kernel_size=7, padding=3),
            nn.BatchNorm1d(c1),
            nn.GELU(),
        )
        self.block1 = DSConvBlock(c1, c1, kernel_size=5)
        self.pool1  = nn.MaxPool1d(4)                          # L -> L/4
        self.block2 = DSConvBlock(c1, c2, kernel_size=3)
        self.pool2  = nn.MaxPool1d(4)                          # L/4 -> L/16
        self.block3 = DSConvBlock(c2, c3, kernel_size=3)
        # No pool after block3 — BiGRU runs on the L/16 = 32-step sequence.

        # ---- BiGRU + attention pool ----
        self.gru = nn.GRU(
            input_size    = c3,
            hidden_size   = gru_hidden,
            num_layers    = gru_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = gru_dropout if gru_layers > 1 else 0.0,
        )
        seq_dim = gru_hidden * 2
        self.attn_pool = AttentionPool1d(seq_dim)

        # ---- statistical side branches ----
        self.hist_mlp = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 32),
        )
        self.entropy_mlp = nn.Sequential(
            nn.Linear(n_entropy_blocks + 4, 32), nn.GELU(),
        )
        self.register_buffer(
            "_struct_chars", torch.tensor(_STRUCT_CHARS, dtype=torch.long)
        )
        self.register_buffer(
            "_bi_a", torch.tensor([a for a, _ in _STRUCT_BIGRAMS], dtype=torch.long)
        )
        self.register_buffer(
            "_bi_b", torch.tensor([b for _, b in _STRUCT_BIGRAMS], dtype=torch.long)
        )
        n_struct = len(_STRUCT_CHARS) + len(_STRUCT_BIGRAMS)
        self.struct_mlp = nn.Sequential(
            nn.Linear(n_struct, 32), nn.GELU(),
        )

        # ---- fused head ----
        fused_dim = seq_dim + 32 + 32 + 32
        self.norm = nn.LayerNorm(fused_dim)
        self.fc1  = nn.Linear(fused_dim, head_hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(head_hidden, NUM_GROUPS)

    # ----- per-branch features -----

    def _seq_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) int64
        h = self.embed(x).permute(0, 2, 1)             # (B, embed_dim, L)
        h = self.stem(h)
        h = self.pool1(self.block1(h))                 # (B, c1, L/4)
        h = self.pool2(self.block2(h))                 # (B, c2, L/16)
        h = self.block3(h)                             # (B, c3, L/16)
        h = h.permute(0, 2, 1).contiguous()            # (B, T, c3)
        h, _ = self.gru(h)                             # (B, T, 2*gru_hidden)
        return self.attn_pool(h)                       # (B, 2*gru_hidden)

    def _entropy_feat(self, x: torch.Tensor) -> torch.Tensor:
        ent   = _block_entropy(x, self.n_blocks)
        stats = torch.stack([
            ent.mean(dim=1),
            ent.std(dim=1),
            ent.amin(dim=1),
            ent.amax(dim=1),
        ], dim=-1)
        return self.entropy_mlp(torch.cat([ent, stats], dim=-1))

    def _struct_feat(self, x: torch.Tensor) -> torch.Tensor:
        uni     = (x.unsqueeze(-1) == self._struct_chars).float().mean(dim=1)
        match_a = (x[:, :-1].unsqueeze(-1) == self._bi_a)
        match_b = (x[:, 1:].unsqueeze(-1)  == self._bi_b)
        bi      = (match_a & match_b).float().mean(dim=1)
        return self.struct_mlp(torch.cat([uni, bi], dim=-1))

    # ----- forward -----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq  = self._seq_feat(x)
        hist = self.hist_mlp(_byte_hist(x))
        ent  = self._entropy_feat(x)
        strc = self._struct_feat(x)
        feat = torch.cat([seq, hist, ent, strc], dim=-1)
        feat = self.norm(feat)
        h    = self.drop(F.gelu(self.fc1(feat)))
        return self.head(h)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
