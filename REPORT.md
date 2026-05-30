# File Fragment Classification & Reconstruction — State of the Art & Next Steps

## Project Context

This codebase implements a **hierarchical cascade classifier** for the FFT-75 dataset: given a single 512-byte disk sector, predict which of 75 file types it belongs to. The pipeline has two stages:

1. **Coarse** — ResNet-1D + bigram branch (1536-d) → 11 semantic groups (~87% accuracy)
2. **Specialist** — per-group heads, some with domain-specific encoders (ArchiveEncoder, TextEncoder) → fine-grained type within group

End-to-end accuracy: **79.4%**. Primary bottlenecks: `archive` and `text` groups (~66% each), where byte-level signatures overlap heavily for non-header fragments.

The codebase performs **classification only** — no file reconstruction logic exists yet.

---

## Current State of the Art (2023–2025)

### Representation

| Paper | Key Idea | Benchmark |
|---|---|---|
| **FiFTy** (2020, Vulinović et al.) | Baseline large-scale CNN, FFT-75 dataset | 66% top-1 (75 classes) |
| **JSANet** (KBS 2024) | Joint Self-Attention at byte, channel, and **sector** level; inter-sector context | +5% on FFT-75, +16% on VFF-16 |
| **ByteNet** (IEEE TMM 2025, [arxiv](https://arxiv.org/abs/2410.20855)) | Dual-branch: raw bytes (FC) + **Byte2Image** (bit-level n-grams → 2D grayscale → ViT) | +12.2% over prior SOTA |
| **Transformer-based carving** (ECCWS 2025) | Pure transformer on byte sequences for forensic carving | — |
| **CNN+LSTM** (DSP 2023) | Sequential modeling over bytes within a sector | modest gains |
| **Bytes Are All You Need** (arXiv 2023) | Direct transformer operating on raw file bytes, modality-independent | general classification |

### Key Themes in Recent Work

1. **Intrabyte information** — ByteNet mines bit-level patterns within each byte (not just byte values), treating the fragment as a 2D bit image. This captures entropy and structure invisible to byte-value CNNs.
2. **Inter-sector context** — JSANet's Sector Self-Attention (SSA) uses neighboring sectors to refine predictions. Adjacent sectors in a file share structural properties; isolated classification throws this away.
3. **Dual-branch fusion** — Combining a global statistics branch (bigrams, histograms) with a local sequential branch is now standard practice. This project already does this; ByteNet extends it to image-domain ViT.
4. **Explainability** — Growing interest (SIFT, 2024) in which bytes/positions drive decisions, especially for forensic use cases.

---

## Reconstructing Files from Fragment Classifications

The step from *classifying fragments* to *reconstructing files* requires:

1. **Fragment ordering** — inferring the sequence of sectors that belong to one file
2. **Boundary detection** — detecting file start (header) and end (footer/EOF)
3. **Gap filling / error correction** — handling interleaved fragments from different files

### Existing Approaches

- **Header-footer carving** (Scalpel, Foremost): rule-based, works only for formats with known magic bytes. Fails on middle fragments.
- **Graph-based reassembly**: build a compatibility graph between fragment pairs (e.g., by byte-level continuity scores or learned similarity), then solve a Hamiltonian path / TSP variant. Used for image reassembly (JPEG block reconstruction).
- **Sequential sector classification + CRF/HMM**: classify each sector independently, then use a sequence model (Viterbi, CRF) to smooth and enforce type-consistency constraints across a run of sectors.
- **LSTM/Transformer sequence models over sector sequences**: JSANet's SSA is a step in this direction — attending to neighbor sectors to improve single-sector classification. Extending this to full-sequence labeling is natural.

---

## What Could Be Explored Next

### 1. Inter-Sector Sequential Modeling (High Impact, Moderate Effort)

The FFT-75 dataset stores fragments independently, but real forensic images contain **runs of contiguous sectors from the same file**. Training a sequence model (Transformer encoder or LSTM) over windows of N consecutive sectors — outputting a label per sector — would:
- Exploit the strong type-consistency priors within a file
- Improve middle-fragment accuracy (the hardest case for isolated classification)
- Enable boundary detection as a side-effect (type transitions = file boundaries)

This project's current architecture treats each sector independently. Adding a sliding-window context layer over the specialist output would be a natural first step.

### 2. Byte2Image / Intrabyte Features (Likely +3–8% on archive/text)

ByteNet's `Byte2Image` converts the 512-byte fragment into a **2D grayscale image by expanding each byte into its 8 bits** and stacking n-gram windows row-wise. A ViT then mines spatial patterns in bit-space. This is orthogonal to the current bigram branch and likely to help the `archive` and `text` groups where compressed vs. uncompressed data differs at the bit level. Implementable as a third branch alongside `ByteEncoder` + `BigramBranch`.

### 3. Header / Footer Anchor Detection

Train an auxiliary head — or repurpose the attention weights — to predict whether a fragment is a **file header, middle, or footer**. This information:
- Directly enables ordered reassembly (sort by anchor type within a type-run)
- Makes the coarse classifier much more accurate for header-heavy formats (ZIP, PDF, ELF)
- Could be supervised with position metadata during training (fragment byte-offset ÷ file size)

The current dataset likely has this metadata available (fragment positions within source files).

### 4. Contrastive / Self-Supervised Pre-Training on Raw Bytes

Instead of training from scratch, pre-train the `ByteEncoder` with a masked-byte objective (mask 15% of bytes, predict them) or a contrastive objective (fragments from the same file are similar). This:
- Builds richer byte-level representations before classification fine-tuning
- Could dramatically improve the hardest cases (middle-of-file fragments with no header signal)
- Is analogous to BERT pre-training; the domain corpus is just raw binary disk images

### 5. Full-File Reconstruction Pipeline

The end-game goal — reconstructing a file from its scattered fragments — could be implemented as:

```
Classify all sectors → type runs → boundary detection → 
fragment similarity scoring (learned pairwise compatibility) → 
graph-based reassembly (max-weight Hamiltonian path) → 
reconstructed file bytes
```

A learned compatibility model (does fragment A plausibly precede fragment B?) trained on consecutive sector pairs from the training corpus could power the graph step. This has been demonstrated for JPEG blocks; extending to 75 types is novel.

---

## Summary Recommendations (Prioritized)

| Priority | Idea | Expected Gain | Effort |
|---|---|---|---|
| 1 | **Inter-sector context window** (SSA / sliding Transformer) | +5–10% end-to-end | Medium |
| 2 | **Byte2Image intrabyte branch** (bit-level 2D + ViT branch) | +3–8% on archive/text | Medium |
| 3 | **Header/footer position head** (fragment offset supervision) | +3–5% + enables carving | Low–Medium |
| 4 | **Masked-byte self-supervised pre-training** | +2–5%, especially middle fragments | High |
| 5 | **Full reconstruction pipeline** (pairwise compatibility + graph reassembly) | Novel contribution | High |

---

## Sources

- [ByteNet: Rethinking Multimedia File Fragment Classification Through Visual Perspectives](https://arxiv.org/abs/2410.20855) (IEEE TMM 2025)
- [Intra- and inter-sector contextual information fusion with joint self-attention (JSANet)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124002004) (KBS 2024)
- [Transformer-Based File Fragment Type Classification for File Carving](https://www.researchgate.net/publication/393049441_Transformer-Based_File_Fragment_Type_Classification_for_File_Carving_in_Digital_Forensics) (ECCWS 2025)
- [File Fragment Type Identification Based on CNN and LSTM](https://dl.acm.org/doi/10.1145/3585542.3585545) (DSP 2023)
- [Bytes Are All You Need: Transformers Operating Directly On File Bytes](https://arxiv.org/abs/2306.00238) (arXiv 2023)
- [FiFTy: Large-Scale File Fragment Type Identification Using CNNs](https://www.researchgate.net/publication/342370999_FiFTy_Large-Scale_File_Fragment_Type_Identification_Using_Convolutional_Neural_Networks)
- [Multi-Resolution Isometric Sampling for Fragmented Image Reassembly](https://link.springer.com/article/10.1007/s12204-025-2861-1) (Springer 2025)
- [Digital Forensics and File Fragment Classification — Nature Research Intelligence](https://www.nature.com/research-intelligence/nri-topic-summaries/digital-forensics-and-file-fragment-classification-micro-427284)
