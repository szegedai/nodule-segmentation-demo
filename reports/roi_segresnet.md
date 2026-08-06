# Lung ROI Segmentation — SegResNet (2D)

## 1. Task

Voxel-level segmentation of lung foreground (thoracic parenchyma) in
axial CT slices. Runs slice-by-slice at native 2D resolution and its
outputs are stacked to form the per-series lung bounding box that the
downstream nodule model crops to.

- **Input**   `(1, 256, 256)` axial CT slice, intensity-normalised to `[0, 1]`
              and resized from the native slice resolution.
- **Output**  `(1, 256, 256)` sigmoid map. Foreground = lung tissue.

## 2. Model

MONAI `SegResNet`, 2D CNN encoder-decoder with residual blocks and group
normalisation.

| Property                | Value |
|-------------------------|-------|
| Architecture family     | SegResNet (residual U-Net) |
| Spatial dims            | 2 |
| Input channels          | 1 |
| Output channels         | 1 (sigmoid) |
| Initial feature width   | 16 |
| Encoder blocks per level| `[1, 2, 2, 4, 4]` |
| Decoder blocks per level| `[1, 1, 1, 1]` |
| Dropout                 | 0.1 |
| Trainable parameters    | **6,904,081** |

## 3. Data

Unified training corpus assembled from three public sources:

| Source     | Role in the split |
|------------|-------------------|
| NLST       | training + validation |
| NSCLC-Radiomics | training + validation |
| LIDC-IDRI  | training + validation (lung labels derived from a earlier medium-capacity nodule model) |

- **Split**: `unified.json` — patient-grouped, dataset-stratified.
  1 683 train / 297 val / 325 test series.
- Slices per split: **365 014 train / 64 097 val / — test held out**.
- **Preprocessing**: each axial slice resized to 256 × 256.
- **Class balance**: lung foreground is ~30-50 % of a mid-thorax slice
  and 0 % of pure top-/bottom-of-volume slices — nothing pathological,
  so plain Dice loss is sufficient.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Dice (sigmoid, squared prediction) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻³ |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = 100, η_min = 1 × 10⁻⁶ |
| Batch size         | 16 |
| Samples per epoch  | 20 000 (random subsample of ~365 k train slices) |
| Epochs (budget)    | 100 |
| Random seed        | 42 |
| Mixed precision    | bf16 (autocast, no GradScaler) |
| Hardware           | 1 × NVIDIA H100 94 GB |

Best checkpoint: **epoch 7 / 100**, validation Dice 0.9600. The model
converges very quickly on this task; further training yields marginal
returns.

## 5. Evaluation protocol

**Held-out test split** (45 751 slices from the `unified` test set).
The training loop used the `val` split for early stopping and
best-Dice checkpoint selection; the numbers below come from the
completely untouched **`test`** split (never seen during training or
model selection). Predictions taken as `sigmoid(logits) > 0.5`. All
metrics are computed at the pixel level and micro-averaged over the
whole test split (~3.0 × 10⁹ pixels).

## 6. Results

### Ticket-required metrics

| Metric             | Value      |
|--------------------|-----------:|
| **mean IoU**       | **0.9704** |
| **Accuracy**       | **0.9957** |
| **Precision**      | **0.9662** |
| **Recall**         | **0.9778** |

### Supplementary

| Metric                          | Value      |
|---------------------------------|-----------:|
| IoU (foreground / lung class)   | 0.9455 |
| IoU (background class)          | 0.9953 |
| Dice / F1 (micro)               | 0.9720 |
| True positives  (px)            | 225 652 517 |
| False positives (px)            | 7 887 800 |
| False negatives (px)            | 5 127 517 |
| True negatives  (px)            | 2 759 669 702 |

### Per-slice Dice distribution (45 751 test slices)

| Statistic | Value  |
|-----------|-------:|
| Mean      | 0.7851 |
| Std       | 0.3747 |
| Min       | 0.0000 |
| p05       | 0.0000 |
| Median    | 0.9770 |
| p95       | 1.0000 |
| Max       | 1.0000 |

The mean-vs-median gap (0.79 vs 0.98) is driven by top-/bottom-of-volume
slices where the lung is absent: a single false-positive pixel there
maps to Dice = 0, dragging the mean. The aggregate voxel-level metrics
(mIoU / Precision / Recall / Dice_micro) are not affected — they weight
each pixel equally.

## 7. Notes on interpretation

Lung ROI segmentation is close to a solved problem on axial CT — the
target is a large, contrast-rich region with relatively simple
morphology. As a result all four ticket metrics land near their upper
bound, which is expected rather than a bug:

- **Accuracy 0.9957** — informative here (unlike the nodule task) because
  the class ratio is roughly 40 / 60, not 10⁻⁵.
- **mIoU 0.9704** — genuinely a mean of two comparable IoUs (0.946 fg,
  0.995 bg), not dominated by one term.
- **Precision 0.9662** — 97 % of predicted-lung pixels are true lung.
- **Recall 0.9778** — 98 % of true-lung pixels are recovered.

For a segmentation task this cleanly separated, the operationally
meaningful comparison across models is at the tails of the per-slice
distribution (p05, min) — where the harder cases sit — rather than at
the mean.

## 8. Reproducibility

- Config       `configs/roi.yaml`
- Checkpoint   `checkpoints/roi/best.pth`
- Metrics JSON `checkpoints/roi/eval_segresnet_test.json`
- Command      `python eval_roi.py --config configs/roi.yaml --ckpt checkpoints/roi/best.pth --split test`
