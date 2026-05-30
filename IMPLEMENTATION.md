# FFT-75 File Fragment Classifier — CNN + BiGRU "Backbone-Plus"

**Target:** maximize accuracy on FFT-75 **Scenario #1, 4096-byte** fragments (the common hard-disk sector size), while staying honest about the ~84% label-intrinsic ceiling on that scenario.

**Audience:** this document is a build spec for an implementing coding agent. Scaffold a PyTorch project from it. Where a value is given, use it as the default; where a `# TODO(verify)` appears, confirm against the actual dataset before trusting it.

---

## 0. Design rationale (read first — it explains every choice below)

The literature converges on ~83–84% for Scenario #1 @ 4096B across CNN, recurrent, and transformer models. That convergence means the ceiling is **label-intrinsic** (container formats embed other types, so the label and the bytes genuinely disagree), not architectural. We therefore do **not** chase a single monolithic model. We build a CNN+BiGRU backbone that is strong where signal exists, and bolt on targeted components for the three problem clusters:

| Cluster | Why models fail | Our lever |
|---|---|---|
| **Text** (csv/html/xml/txt/json/…) | near-identical byte/bigram stats; signal is *structural* | BiGRU (captures tag/delimiter rhythm) + explicit structural features |
| **Archive** (zip/gz/rar/7z/ooxml/encrypted) | entropy saturates; compressed≈encrypted | statistical-feature fusion + optional high-entropy expert head |
| **disk_image / containers** (dmg/iso/pdf) | single-fragment classification is ill-posed | label-smoothing/noise-aware loss + report as a ceiling (context model is out of scope here) |

Core principle: **the BiGRU helps text; it does nothing for high-entropy noise.** So statistical features must be fused in a parallel path, not expected to emerge from the sequence model.

---

## 1. Repository layout

```
fft75-cnn-bigru/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml            # all hyperparameters + cluster/superclass maps
├── data/
│   ├── __init__.py
│   ├── dataset.py              # FFT-75 npz loader, returns (bytes, aux_feats, label)
│   ├── features.py             # auxiliary statistical feature extraction
│   └── augment.py              # GBFlip
├── models/
│   ├── __init__.py
│   ├── backbone.py             # embedding → multi-scale CNN → BiGRU → pool
│   ├── heads.py                # main head, aux-fusion, (phase2) high-entropy expert
│   └── classifier.py           # assembles backbone + heads
├── train.py                    # training loop, AdamW, ReduceLROnPlateau, early stop
├── evaluate.py                 # overall/per-class/superclass/per-cluster metrics
├── distill.py                  # phase 3: KD from a transformer teacher (optional)
└── utils/
    ├── metrics.py              # confusion matrix, cluster leakage, superclass collapse
    └── seed.py
```

---

## 2. Environment

```
python >= 3.10
torch >= 2.1
numpy
pyyaml
scikit-learn        # confusion matrix, metrics
matplotlib          # confusion-matrix plots
tqdm
```

`requirements.txt` should pin these. Assume a single CUDA GPU (the user runs on vast.ai / local RTX). Keep batch sizes configurable so it runs on 12–24 GB cards.

---

## 3. Data pipeline

### 3.1 Source
FFT-75 from IEEE DataPort. For this project download **`4k_1.tar.gz`** (Scenario #1, 4096-byte). Also support `512_1.tar.gz` via config for the 512B sanity check. Extract to `--data_dir`.

### 3.2 Expected on-disk format
FiFTy ships each scenario/size as a folder containing `train.npz`, `val.npz`, `test.npz`. Each `.npz` holds:
- `x`: `uint8` array, shape `(N, block_size)` — raw fragment bytes
- `y`: integer array, shape `(N,)` — class id in `[0, 74]`

`# TODO(verify)` the exact key names (`x`/`y` vs `arr_0`/`arr_1`) on first load and adapt `dataset.py`. Print shapes and dtype on load.

### 3.3 `dataset.py`
A `torch.utils.data.Dataset` that, per item, returns:
- `bytes_tensor`: `LongTensor (block_size,)` — byte values 0–255 (embedding indices)
- `aux_feats`: `FloatTensor (F_aux,)` — see §4, computed on the **clean** (pre-GBFlip) bytes
- `label`: `long`

Order of operations per `__getitem__`:
1. load raw `uint8` block
2. compute `aux_feats` from clean bytes (§4)
3. if training, apply GBFlip to a copy of the bytes (§5)
4. return `(bytes_tensor, aux_feats, label)`

Compute aux features from **clean** bytes so the statistical path sees ground truth; GBFlip only perturbs the sequence path. Cache aux features to disk (`.npy` memmap keyed by split) since they're deterministic and recomputation is wasteful across epochs.

---

## 4. Auxiliary statistical features (`features.py`)

Single function `extract_features(block: np.uint8[block_size]) -> np.float32[F_aux]`. These target the clusters the BiGRU can't crack. Compute, in this order (document the index of each in a comment so `aux_feats` slices are stable):

**Entropy / randomness (archive cluster):**
1. Shannon unigram entropy (bits/byte, 0–8), normalized to [0,1]
2. Bigram entropy (normalized)
3. Compression ratio: `len(zlib.compress(block, level=6)) / len(block)` — strong low-vs-high entropy separator; compressed/encrypted ≈1.0, text ≪1.0
4. Chi-square statistic of byte histogram vs uniform (randomness test), log-scaled
5. Mean run-length of repeated bytes

**Byte-distribution summary (cheap, dense):**
6. Fraction of null bytes (`0x00`)
7. Fraction of high bytes (`0x80–0xFF`)
8. Fraction printable ASCII (`0x20–0x7E`)
9. Fraction whitespace (`0x09,0x0A,0x0D,0x20`)

**Structural markers (text cluster) — densities per fragment:**
10. `<` and `>` density (html/xml)
11. `{` `}` `[` `]` density (json)
12. `,` density (csv)
13. `;` `=` density
14. `\n` (newline) density
15. `/` density (paths/urls)

Set `F_aux` = total count (≈15). **Normalize** each feature to roughly [0,1] or standardize with train-split mean/std saved to the config/checkpoint. Do **not** include the full 256-bin histogram here — it duplicates what the embedding+CNN already learns and bloats the fusion MLP; the summary stats above are the orthogonal signal.

---

## 5. GBFlip augmentation (`augment.py`)

Gaussian Bit-Flip (from XMP). Simulates real bit-errors and regularizes the high-entropy classes.

```python
def gbflip(block: np.uint8[L], sigma: float = 0.01, rng) -> np.uint8[L]:
    # per-fragment flip rate ~ |Normal(0, sigma)|, clipped to [0, max_rate]
    rate = min(abs(rng.normal(0, sigma)), 0.05)
    n_bits = int(rate * L * 8)
    if n_bits == 0: return block
    bit_positions = rng.choice(L * 8, size=n_bits, replace=False)
    out = block.copy()
    for bp in bit_positions:
        byte_i, bit_i = bp // 8, bp % 8
        out[byte_i] ^= (1 << bit_i)
    return out
```

Apply **train only**, after aux-feature extraction. Make `sigma` a config knob; start at `0.01`. Vectorize the bit-flip loop if it shows up in profiling.

---

## 6. Model architecture

### 6.1 Backbone (`backbone.py`)

Input `bytes: Long (B, L)`, `L=4096`.

```
Embedding(256 -> d_emb=24)                      -> (B, L, 24)
transpose                                        -> (B, 24, L)

Multi-scale 1D CNN (parallel branches, capture different n-gram widths):
  branch_k for k in [9, 27]:                     # ByteSCAN found k=27 effective
    Conv1d(24, 128, kernel=k, padding=same)
    BatchNorm1d(128)
    GELU
  concat branches over channels                  -> (B, 256, L)
  MaxPool1d(4)                                    -> (B, 256, L/4)
  Conv1d(256, 256, kernel=3, padding=same), BN, GELU
  MaxPool1d(4)                                    -> (B, 256, L/16)   # 4096 -> 256

transpose                                        -> (B, 256, 256)  (seq_len=256, feat=256)

BiGRU(input=256, hidden=128, num_layers=2,
      bidirectional=True, dropout=0.2)           -> (B, 256, 256)

Pooling (attention pool over time; see note):    -> (B, 256)
```

**Activation:** GELU throughout the CNN. If you later add any FFN/attention block, use **GEGLU** there (gated GELU; size hidden to ~2/3 to hold params constant) — small consistent gain.

**Pooling note:** use additive attention pooling over the BiGRU outputs (a learned query scoring each timestep, softmax, weighted sum) rather than just last-hidden-state. Attention pool localizes the discriminative region — exactly what helps text (a tag burst) and media (a header). Concatenating mean-pool ⊕ attention-pool is a fine default.

Keep the BiGRU **shallow (1–2 layers)**: ByteRCNN's gains come from modest recurrence; depth mostly buys latency.

### 6.2 Heads (`heads.py`)

**Aux-fusion + main head:**
```
aux_mlp: Linear(F_aux -> 64), GELU, LayerNorm(64)        -> (B, 64)
fused = concat(seq_feat (B,256), aux_feat (B,64))         -> (B, 320)
Dropout(0.3)
main_head: Linear(320 -> 75)                              -> logits (B, 75)
```

**(Phase 2) High-entropy expert — soft mixture, off by default:**
```
gate = sigmoid(Linear(entropy_feats_subset -> 1))         # uses features 1–5 only
expert_head: Linear(320 -> 75)                            # trained weighted toward high-entropy classes
final_logits = (1 - gate) * main_logits + gate * expert_logits
```
Gate input is the entropy/randomness features (indices 1–5 from §4), so routing is driven by "is this fragment high-entropy?" Train with the expert's gradient up-weighted on archive-cluster samples. Add a small entropy-regularizer on `gate` so it doesn't collapse to 0.5. Ship phase 1 without this; add it only if archive recall is the bottleneck after phase 1.

---

## 7. Loss (`train.py`)

**Phase 1:** cross-entropy with **per-class label smoothing**.
- Default smoothing `0.1`.
- **Elevated smoothing (`0.2`) on container/ambiguous classes** (disk_image, pdf, ooxml family). Hard targets on these classes just teach the model to overfit genuinely-ambiguous fragments; smoothing tells it "be uncertain here, that's correct." Implement as a per-class smoothing vector, not a scalar.

**(Phase 3) Knowledge distillation (`distill.py`, optional):**
```
L = alpha * KL(softmax(student/T) || softmax(teacher/T)) * T^2
  + (1 - alpha) * CE_labelsmooth(student, y)
```
Teacher = a stronger model's logits (CarveFormer/XMP if you can train or obtain one, else an ensemble of phase-1 runs). `T≈4`, `alpha≈0.5`. The point is to transfer the inter-class similarity geometry (zip↔gz, txt↔html) the BiGRU won't discover alone. Cache teacher logits to disk to avoid running the teacher every epoch.

---

## 8. Training config (`configs/default.yaml`)

```yaml
data:
  data_dir: ./data/4k_1
  block_size: 4096
  num_classes: 75
model:
  d_emb: 24
  cnn_kernels: [9, 27]
  cnn_channels: 128
  gru_hidden: 128
  gru_layers: 2
  aux_dim: 15            # F_aux, must match features.py
  dropout: 0.3
  use_expert: false      # phase 2 switch
train:
  batch_size: 256        # raise to 512/1024 if VRAM allows (CarveFormer used 1024)
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 0.05
  scheduler: reduce_on_plateau   # factor 0.5, patience 3, monitor val_acc
  max_epochs: 100
  early_stop_patience: 10
  label_smoothing: 0.1
  container_smoothing: 0.2
  gbflip_sigma: 0.01
  amp: true              # mixed precision
clusters:                # see §9; VERIFY against FFT-75 label map
  text: []
  archive: []
  disk_image: []
```

Select the best checkpoint by **validation accuracy** measured every epoch (CarveFormer/ByteSCAN convention). Log per-epoch train/val loss+acc. Set seeds (`utils/seed.py`) and report mean±std over ≥3 runs for any headline number — gains in this field are often within run-to-run noise.

---

## 9. Cluster & superclass maps (`# TODO(verify)` — critical)

FFT-75 ships class **ids** (0–74), not names. Before evaluation you MUST build the id→name→cluster map from the dataset's documentation/`readme.md`. The lists below are **representative, not authoritative** — confirm every membership against the official label table.

```yaml
# Representative groupings — VERIFY before use.
clusters:
  text:        [csv, html, xml, txt, json, md, log, css, js, ...]
  archive:     [zip, gz, rar, 7z, bz2, xz, tar, deb, rpm, apk, jar, ooxml(doc/x,xls/x,ppt/x), ...]
  disk_image:  [dmg, iso, ...]            # plus pdf as a "publish" container
superclasses_11:    # ByteNet use-case tags, for collapsed confusion matrix
  - archive, audio, bitmap, executable, published,
    office/ooxml, raw_camera, text, video, ... # VERIFY exact 11
```

Provide a helper that loads the official mapping and validates that every class id lands in exactly one superclass and at most one problem cluster; fail loudly if not.

---

## 10. Evaluation & analysis (`evaluate.py`, `utils/metrics.py`)

Report all of the following on the **test** split:

1. **Overall top-1 accuracy** (the headline; compare to ~84% literature ceiling on #1@4096).
2. **Per-class accuracy** table, sorted ascending (worst first).
3. **Full 75×75 confusion matrix** (save CSV + normalized PNG).
4. **Superclass-collapsed accuracy** (map 75→11, recompute) — separates fixable errors from intrinsic ones.
5. **Per-cluster diagnostics** for text / archive / disk_image:
   - within-cluster recall (how often a text fragment is called *some* text type)
   - cross-cluster leakage matrix (3×3: where do misroutes go)
   - this is the real scorecard for this project, more than the flat 75-way number.
6. **Ceiling-aware accuracy** (optional): top-1 excluding misroutes *into* container classes from their genuine embedded type — quantifies how much of the residual error is label-intrinsic vs model error.

Acceptance for phase 1: match literature on #1@4096 (**≥83%** overall), and demonstrate that the BiGRU + structural features lift **text within-cluster recall** above a CNN-only baseline (ablate the BiGRU and the aux path to prove each earns its place).

---

## 11. Build order (milestones)

1. **M1 — data + baseline.** `dataset.py` loading, sanity-print shapes, train a CNN-only classifier (no BiGRU, no aux) to reproduce ~FiFTy-level accuracy. Establishes the harness.
2. **M2 — BiGRU.** Add the BiGRU + attention pool. Expect gains concentrated in text/media; verify via per-cluster metrics.
3. **M3 — aux fusion.** Add `features.py` + fusion path. Expect archive/high-entropy improvement. Ablate to confirm.
4. **M4 — GBFlip + label smoothing.** Add augmentation and per-class smoothing. Re-measure; lock the phase-1 config.
5. **M5 (optional) — high-entropy expert.** Flip `use_expert: true`, train, keep only if archive recall improves.
6. **M6 (optional) — distillation.** `distill.py` from a transformer/ensemble teacher.

Ablations to always report: −BiGRU, −aux, −GBFlip, −container_smoothing. Each milestone must show its component pays for itself on the per-cluster metrics, not just the flat number.

---

## 12. Explicit non-goals / honest limits

- **disk_image is a ceiling, not a target.** A single random data sector of a disk image *is* whatever file it contains; no single-fragment model fixes this. The label-smoothing treatment is damage control, not a solution. Resolving it requires inter-sector context (out of scope here) — note it in the README as future work.
- **ooxml↔zip and compressed↔encrypted are partially intrinsic.** Don't over-tune toward them; superclass evaluation is the fair lens.
- Report numbers as mean±std; do not claim a sub-1% improvement as real without repeated runs.
