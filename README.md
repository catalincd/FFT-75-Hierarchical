# FFT-75 Hierarchical Cascade Classifier

A two-stage hierarchical model for file fragment type classification over the FFT-75 dataset.

- **Stage 1 (Coarse):** ByteResNet + GeGLU head predicts one of 11 file-type groups
- **Stage 2 (Specialist):** Per-group fine-grained classifiers predict among 75 file types
- **Results:** 87% coarse accuracy · 79.4% end-to-end accuracy across 58 leaf classes

---

## Results

End-to-end accuracy accounts for coarse routing errors propagating into specialist predictions.

| Group | Classes | Specialist Acc | End-to-End Acc |
|-------|:-------:|:--------------:|:--------------:|
| video | 6 | 99.94% | 86.95% |
| image_raw | 5 | 99.88% | 86.90% |
| image_raster | 6 | 99.81% | 86.83% |
| other | 5 | 99.09% | 86.21% |
| disk_image | 3 | 99.31% | 86.40% |
| executable | 5 | 98.65% | 85.83% |
| database | 3 | 95.12% | 82.75% |
| audio | 6 | 94.89% | 82.56% |
| document | 7 | 94.11% | 81.87% |
| archive | 6 | 66.38% | 57.75% |
| text | 6 | 66.39% | 57.76% |
| **Total** | **58** | — | **79.4%** |

The `archive` and `text` groups are the primary bottleneck — their byte-level signatures overlap significantly at the fragment level.

---

## Repository Layout

```
FFT-75-Hierarchical/
├── src/
│   ├── hierarchical_cascade.py   # Model definitions + full cascade training
│   ├── train_phase1.py           # Standalone Phase 1 training (crash-safe checkpointing)
│   ├── load_binary.py            # Fast binary data loader
│   ├── load_binary_lazy.py       # On-demand fragment loading (for large datasets)
│   └── convert_npz_to_binary.py  # Convert NPZ -> binary (run once per dataset)
├── scripts/
│   ├── setup_vastai.sh           # One-shot setup on a fresh vast.ai instance
│   ├── train_phase1.sh           # Launch Phase 1 training
│   ├── upload_model.sh           # Push checkpoints to HuggingFace Hub
│   ├── download_data.sh          # Pull dataset from HuggingFace Hub
│   └── download_model.sh         # Pull checkpoints from HuggingFace Hub (local use)
├── latex/                        # Thesis source (compile with pdflatex)
├── data/                         # Downloaded dataset lives here (gitignored)
├── checkpoints/                  # Training checkpoints (gitignored)
└── requirements.txt
```

---

## Data

The dataset (~30 GB binary) is hosted on HuggingFace Hub and is not included in this repo.

You must first upload the binary split files to your own HF dataset repo:

```
HF dataset repo: YOUR_HF_USERNAME/fft75-dataset
Expected files:  train_fragments.bin  train_labels.bin  train_meta.json
                 val_fragments.bin    val_labels.bin    val_meta.json
                 test_fragments.bin   test_labels.bin   test_meta.json
```

If you have the original NPZ files, convert them first:

```bash
python src/convert_npz_to_binary.py --source-dir /path/to/npz --out-dir data/4k_1/binary
```

---

## Training on vast.ai

### 1. Rent an instance

Pick a GPU instance with at least 16 GB VRAM and ~50 GB disk. A100 or RTX 4090 recommended. Use a PyTorch image (e.g. `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel`).

### 2. Clone and set up

```bash
git clone https://github.com/YOUR_GH_USERNAME/FFT-75-Hierarchical.git
cd FFT-75-Hierarchical
bash scripts/setup_vastai.sh --hf-user YOUR_HF_USERNAME
```

This installs dependencies and downloads the dataset (~30 GB, takes a few minutes).

To also pull the latest checkpoint for resuming:

```bash
bash scripts/setup_vastai.sh --hf-user YOUR_HF_USERNAME --resume-checkpoint
```

### 3. Train

```bash
# Full run (recommended): lazy loading, 50 epochs, compile for max throughput
bash scripts/train_phase1.sh --compile

# Resume after a crash or instance restart
bash scripts/train_phase1.sh --resume --compile

# Quick test run with a small subset
bash scripts/train_phase1.sh --max-per-class 1000 --epochs 5
```

Checkpoints are saved to `checkpoints/phase1_archive/` after every epoch. A crash at any point leaves all completed epochs intact.

### 4. Upload checkpoints

After training (or at any point during):

```bash
bash scripts/upload_model.sh YOUR_HF_USERNAME
```

---

## Using Models Locally

```bash
# Download checkpoints from HuggingFace
bash scripts/download_model.sh YOUR_HF_USERNAME

# Load and run inference
python - <<'EOF'
import torch
from pathlib import Path
from src.hierarchical_cascade import ByteEncoder, CoarseClassifier, GROUP_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = ByteEncoder()
model   = CoarseClassifier(encoder).to(device)

ckpt = torch.load("checkpoints/phase1_archive/best.pt", map_location=device, weights_only=True)
model_sd = ckpt["model"]
if any(k.startswith("_orig_mod.") for k in model_sd):
    model_sd = {k.removeprefix("_orig_mod."): v for k, v in model_sd.items()}
model.load_state_dict(model_sd)
model.eval()

# Classify a raw 512-byte file sector
sector = open("/path/to/file", "rb").read(512)
x = torch.frombuffer(bytearray(sector), dtype=torch.uint8).to(torch.int64).unsqueeze(0).to(device)
with torch.inference_mode():
    pred = model(x).argmax(-1).item()
print("Predicted group:", GROUP_NAMES[pred])
EOF
```

---

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--binary-dir` | `data/4k_1/binary` | Path to binary split directory |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 512 | Per-GPU batch size |
| `--lr` | 1e-4 | Peak learning rate (OneCycleLR) |
| `--lazy` | off | Read fragments on demand (saves RAM) |
| `--compile` | off | `torch.compile` for 2-3x CUDA throughput |
| `--grad-checkpoint` | off | Recompute activations to save ~60% VRAM |
| `--max-per-class` | none | Hard cap on samples per class |
| `--resume` | off | Resume from latest checkpoint in archive |
| `--archive-dir` | `checkpoints/phase1_archive` | Checkpoint output directory |

---

## Model Architecture

```
Input: (B, 512) raw byte values

ByteEncoder (ByteResNet backbone):
  Embedding(256, 16)
  Stem: Conv1d(7) + BN + GELU
  Stage 1: ResConvBlock(F, F, k=7) + MaxPool(4)
  Stage 2: ResConvBlock(F, 2F, k=5) x2 + MaxPool(4)
  Stage 3: ResConvBlock(2F, 4F, k=3) x2
  AdaptiveAvgPool1d(1)
  -> (B, 512) feature vector

CoarseClassifier head:
  LayerNorm -> GeGLU(512->256) + Skip -> LayerNorm -> Linear(256, 11)
  -> (B, 11) group logits

SpecialistClassifier head (per group):
  Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, num_classes)
  -> (B, num_classes) fine-grained logits
```

Each residual block uses SE (Squeeze-and-Excitation) channel attention after the second conv.
