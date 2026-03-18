<div align="center">

# XAI-Guided Neural Network Pruning

### Skin Lesion Classification · HAM10000 · ResNet18

*Pruning a neural network by what it finds relevant — not just by weight magnitude.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey?style=flat-square)](LICENSE.md)
[![Paper](https://img.shields.io/badge/Paper-ICW%20Report%20(DE)-8A2BE2?style=flat-square&logo=adobeacrobatreader&logoColor=white)](docs/ICW_report.pdf)
[![ProtoPNet Repo](https://img.shields.io/badge/Companion%20Repo-ProtoPNet__ICW1-24292e?style=flat-square&logo=github)](https://github.com/ilonae/ProtoPNet_ICW1)

**Independent Coursework (ICW 1) · M.Sc. Applied Computer Science · HTW Berlin**
*Ilona Eisenbraun · [ilonaeisenbraun@gmail.com](mailto:ilonaeisenbraun@gmail.com)*

## What is this?

Standard network pruning removes filters with the smallest *weights* — but small weights don't necessarily mean unimportant ones. This project investigates whether using **explainability signals** (specifically what the network actually *uses* to make a decision) produces better pruning criteria.

Two XAI-driven approaches are implemented and compared on a real-world medical imaging task:

| Approach                    | Explainability Method            | What gets pruned                                              |
| --------------------------- | -------------------------------- | ------------------------------------------------------------- |
| **LRP pruning**       | Layer-wise Relevance Propagation | Convolutional filters with lowest relevance scores            |
| **Prototype pruning** | ProtoPNet concept activation     | Visual prototypes that activate on non-discriminative regions |

Both are benchmarked against naive **weight magnitude pruning** on ResNet18 trained for 7-class skin lesion classification.

> The original coursework report (written in German) is included in [`docs/ICW_report.pdf`](docs/ICW_report.pdf).
> The ProtoPNet implementation lives in the companion repo: [ilonae/ProtoPNet_ICW1](https://github.com/ilonae/ProtoPNet_ICW1).

---

## Results

**Baseline:** ResNet18, 10 epochs → **90.21% test accuracy**

### Accuracy across pruning methods

| Pruning Criterion    | After pruning | After fine-tuning | Δ from baseline |
| -------------------- | :-----------: | :---------------: | :--------------: |
| LRP activations      |    72.16%    | **84.04%** |     −6.17%     |
| Weight magnitudes    |    72.16%    |      84.04%      |     −6.17%     |
| ProtoPNet prototypes |    81.32%    | **90.11%** |     −0.10%     |

### Computational cost reduction (LRP, 20% pruning rate, 30 iterative steps)

```
FLOPs:  1.8T  ──────────────────────►  1.1T   (−39%)
```

### Key takeaway

> LRP-guided pruning recovers accuracy **more stably** than weight magnitude across all 30 pruning steps.
> ProtoPNet prototype pruning achieves near-lossless compression — but operates on a fundamentally different, inherently interpretable architecture.
> Weight magnitude pruning reaches the same final accuracy as LRP here, but shows **higher variance** in recovery across steps (see training curves in the paper).

---

## How LRP pruning works

```
  Input image
      │
      ▼
 ┌─────────────┐    Forward pass     ┌──────────────────────────┐
 │  ResNet18   │ ──────────────────► │  Classification output   │
 └─────────────┘                     └──────────────┬───────────┘
                                                     │
                                          LRP backward pass
                                                     │
                                                     ▼
                                     ┌──────────────────────────┐
                                     │  Relevance score R per   │
                                     │  filter across all layers│
                                     └──────────────┬───────────┘
                                                     │
                                         Rank filters by R
                                                     │
                                                     ▼
                                     ┌──────────────────────────┐
                                     │  Remove bottom-k filters │
                                     │  Fine-tune 1 epoch       │
                                     │  Repeat for 30 steps     │
                                     └──────────────────────────┘
```

The relevance of neuron *i* in layer *l* is propagated as:

$$
R_i^{(l)} = \sum_j \frac{a_i \, w_{ij}^+}{\sum_{i'} a_{i'} \, w_{i'j}^+} \; R_j^{(l+1)}
$$

The algorithm is initialised with $R_c^{(l)} = 1$ at the output, making it robust to prediction uncertainty.

---

## Dataset

**HAM10000** — 10,015 dermatoscopic images labelled by medical professionals across 7 classes:

| Code      | Class                         | Notes                                 |
| --------- | ----------------------------- | ------------------------------------- |
| `nv`    | Melanocytic nevi              | Largest class — oversampling applied |
| `mel`   | Melanoma                      | Clinically critical                   |
| `bkl`   | Benign keratosis-like lesions |                                       |
| `bcc`   | Basal cell carcinoma          |                                       |
| `akiec` | Actinic keratoses             |                                       |
| `vasc`  | Vascular lesions              | Smallest class                        |
| `df`    | Dermatofibroma                |                                       |

Class imbalance is addressed via **per-class augmentation resampling** in the data loader.

**Download options:**

- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — original source, DOI: 10.7910/DVN/DBW86T
- [ISIC Archive](https://isic-archive.com) — ISIC 2018 Task 3

**Expected directory structure:**

```
data/ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part1/
│   └── *.jpg
└── HAM10000_images_part2/
    └── *.jpg
```

---

## Modernization — what changed from the original coursework

The original code targeted Python 2, PyTorch 0.4, and pandas 1.x. The `update/modernize-dependencies` branch brings it fully up to date:

| # | File(s)                                               | What was broken                                                                                           | Fix                                                                      |
| - | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1 | `modules/data.py`                                   | `df.append()` removed in pandas 2.0                                                                     | Replaced with `pd.concat()`                                            |
| 2 | `modules/network.py`, `modules/prune_*.py`        | `pretrained=True` deprecated in torchvision 0.13                                                        | Replaced with `weights='DEFAULT'`                                      |
| 3 | `modules/lrp.py`                                    | `NameError(...)` instantiated but never raised — unsupported modules silently did nothing              | Replaced with `raise NotImplementedError()`                            |
| 4 | `modules/prune_resnet.py`, `modules/prune_vgg.py` | `torch.autograd.Variable(...)` is a no-op since PyTorch 0.4                                             | Removed entirely                                                         |
| 5 | `main.py`                                           | All architectures instantiated before dict lookup — VGG downloaded weights even when `--arch resnet18` | Replaced eager dict with lazy conditional                                |
| 6 | `main.py`                                           | `argparse type=bool` converts string `"False"` → `True`, so `--resume False` was always truthy   | Replaced with `BooleanOptionalAction` (`--resume` / `--no-resume`) |
| 7 | `modules/network.py`                                | `AvgPool2d(7)` assumes 7×7 spatial maps (224×224 input) — crashes on CIFAR10                         | Replaced with `AdaptiveAvgPool2d((1, 1))`                              |
| 8 | All modules                                           | `from __future__ import` statements throughout (Python 2 compat)                                        | Removed (dead code)                                                      |

`requirements.txt` was also added — the original repo had none.

---

## Setup

```bash
git clone https://github.com/ilonae/LRP_pruning_ICW1.git
cd LRP_pruning_ICW1

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **macOS SSL note:** Python from python.org ships without system certificates.
> If you get `CERTIFICATE_VERIFY_FAILED` when downloading model weights, run:
>
> ```bash
> open /Applications/Python\ 3.*/Install\ Certificates.command
> ```

---

## Usage

### 1 — Train from scratch

```bash
python main.py \
  --arch resnet18 \
  --data-type ham1000 \
  --classnum 7 \
  --train \
  --no-resume \
  --no-prune \
  --epochs 10
```

Checkpoint saved to `./checkpoint/resnet18_ham1000_orig_ckpt.pth`.

### 2 — Prune a trained model

```bash
python main.py \
  --arch resnet18 \
  --data-type ham1000 \
  --classnum 7 \
  --no-train \
  --resume \
  --prune \
  --method-type lrp \
  --total-pr 0.6 \
  --pr-step 0.15
```

### 3 — Quick end-to-end test with CIFAR10 (no data download needed)

```bash
# Train
python main.py \
  --arch resnet18 \
  --data-type cifar10 \
  --classnum 10 \
  --train \
  --no-resume \
  --no-prune \
  --epochs 5

# Prune
python main.py \
  --arch resnet18 \
  --data-type cifar10 \
  --classnum 10 \
  --no-train \
  --resume \
  --prune \
  --method-type lrp
```

### CLI reference

| Argument                       | Default      | Description                                            |
| ------------------------------ | ------------ | ------------------------------------------------------ |
| `--arch`                     | `resnet18` | `resnet18` · `resnet50` · `vgg16` · `vgg19` |
| `--data-type`                | `ham1000`  | `ham1000` · `cifar10` · `imagenet`             |
| `--classnum`                 | `7`        | Number of output classes                               |
| `--method-type`              | `weight`   | `lrp` · `weight` · `grad` · `taylor`        |
| `--total-pr`                 | `0.6`      | Total fraction of filters to prune                     |
| `--pr-step`                  | `0.15`     | Fraction pruned per iteration                          |
| `--epochs`                   | `10`       | Training epochs                                        |
| `--train` / `--no-train`   | off          | Enable training phase                                  |
| `--resume` / `--no-resume` | on           | Load checkpoint before running                         |
| `--prune` / `--no-prune`   | on           | Enable pruning phase                                   |
| `--no-cuda`                  | off          | Force CPU execution                                    |

---

## Project structure

```
LRP_pruning_ICW1/
│
├── main.py                    # Entry point — training & pruning
├── visualize.py               # Accuracy / FLOPs comparison plots
├── requirements.txt
│
├── modules/
│   ├── data.py                # HAM10000, CIFAR10, ImageNet data loaders
│   ├── network.py             # Model definitions (ResNet, VGG, AlexNet)
│   ├── lrp.py                 # LRP rules: simple, α-β, ε, first-layer
│   ├── filterprune.py         # Filter ranking and removal
│   ├── prune_resnet.py        # ResNet training / pruning loop
│   ├── prune_vgg.py           # VGG / AlexNet training / pruning loop
│   ├── prune_layer.py         # Layer-level pruning ops
│   ├── resnet_kuangliu.py     # LRP-wrapped ResNet (forward/backward hooks)
│   ├── flop.py                # FLOPs counter
│   └── flops_counter*.py      # FLOPs utilities (masked / unmasked)
│
├── utils/
│   └── lrp_general6.py        # General LRP wrapper
│
└── docs/
    └── ICW_report.pdf         # Full coursework paper (German)
```

---

## Background & references

This work adapts and extends:

> **Yeom et al. (2021)** — *Pruning by explaining: A novel criterion for deep neural network pruning.*
> Pattern Recognition · Elsevier · [doi:10.1016/j.patcog.2021.107899](https://doi.org/10.1016/j.patcog.2021.107899)

> **Chen et al. (2019)** — *This looks like that: deep learning for interpretable image recognition.*
> NeurIPS 2019

> **Tschandl et al. (2018)** — *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.*
> Scientific Data · [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

```bibtex
@article{yeom2021pruning,
  title     = {Pruning by explaining: A novel criterion for deep neural network pruning},
  author    = {Yeom, Seul-Ki and Seegerer, Philipp and Lapuschkin, Sebastian and
               Binder, Alexander and Wiedemann, Simon and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
  journal   = {Pattern Recognition},
  pages     = {107899},
  year      = {2021},
  publisher = {Elsevier}
}
```

---

## License

[CC BY-NC-SA 4.0](LICENSE.md) — non-commercial use with attribution.
Original framework © Fraunhofer HHI · TU Berlin · SUTD Singapore (2019–2020).
