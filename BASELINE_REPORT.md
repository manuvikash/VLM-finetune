# Baseline Results & Technical Specification

This document records the **control-group baseline** for the project: a standard image classifier without hierarchy-aware training. It is the reference point for later experiments (e.g., hierarchy-aware models).

---

## Baseline Results

The baseline is a **flat** multi-class classifier: **ResNet-18** initialized with **ImageNet** pretrained weights, with the final fully connected layer replaced to predict **50** leaf classes. Training uses **cross-entropy** and ignores any taxonomy in the JSON `"path"` field.

**Dataset (this run):**

| Split | Samples |
|-------|--------:|
| Train | 1,280 |
| Test  |   320 |

**Training loss by epoch** (mean cross-entropy over the training set):

| Epoch | Train loss |
|------:|-----------:|
| 1 | 3.3586 |
| 2 | 1.8694 |
| 3 | 0.9854 |
| 4 | 0.5072 |

**Test performance:**

| Metric | Value |
|--------|------:|
| **Top-1 accuracy** | **55.31%** (177 / 320 correct) |

This is a reasonable starting point for a small fine-grained dataset on CPU with only four epochs; accuracy is expected to change if you use a GPU, train longer, or tune augmentation and learning rate.

---

## Technical Specification

### Experimental purpose

Establish a **simple, reproducible baseline** for image classification: pretrained CNN + standard supervised fine-tuning. No hierarchical loss, no label noise experiments—those belong to follow-up work.

### Hardware

| Item | Detail |
|------|--------|
| **Device (logged run)** | CPU (`Device: cpu` in `train_baseline.py`) |
| **GPU** | Not used in the logged run—add your GPU model here if you rerun on CUDA. |

*Replace the row above with your machine’s CPU/GPU and RAM if your instructor requires a full hardware table.*

### Software

| Component | Version / notes |
|-----------|-----------------|
| **Python** | 3.x (compatible with PyTorch 2.2+) |
| **PyTorch** | ≥ 2.2.0 |
| **torchvision** | ≥ 0.17.0 |
| **Pillow** | ≥ 10.0.0 |
| **OS** | Windows (paths and `num_workers=0` default suit local runs) |

Full pinned dependencies are listed in `requirements.txt` in the repository.

### Data pipeline

| Setting | Value |
|---------|--------|
| **Manifests** | `data/train.json`, `data/test.json` |
| **Label encoding** | Alphabetically sorted unique training labels → integers `0 … 49` |
| **Input resolution** | 224 × 224 |
| **Normalization** | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| **Augmentation** | None (resize + tensor + normalize only) |

### Model

| Item | Value |
|------|--------|
| **Architecture** | ResNet-18 |
| **Pretraining** | `ResNet18_Weights.IMAGENET1K_V1` (ImageNet-1K) |
| **Head** | Final `fc` layer: `512 → num_classes` (50 classes) |
| **Loss** | `CrossEntropyLoss` (single flat softmax over classes) |

### Hyperparameters (default `train_baseline.py`)

| Hyperparameter | Value |
|----------------|------:|
| Epochs | 4 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| DataLoader workers | 0 (Windows-friendly default) |

### Evaluation

| Metric | Definition |
|--------|------------|
| **Top-1 accuracy** | Fraction of test images whose predicted class (argmax over logits) equals the ground-truth leaf label. |

### Reproducibility

From the project root (with `data/train.json`, `data/test.json`, and image paths valid):

```bash
python train_baseline.py
```

Optional: `--epochs`, `--batch-size`, `--lr`, `--data-root` as documented in `train_baseline.py`.

---

## Role of this baseline

This setup is the **control group**: a conventional CNN fine-tuning recipe. Comparing future models (e.g., hierarchy-aware training) against these **same data splits, metrics, and reporting format** keeps experiments fair and easy to interpret.
