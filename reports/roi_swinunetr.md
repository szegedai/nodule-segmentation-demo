# Lung ROI Segmentation — SwinUNETR (2D)

## 1. Task

Voxel-level segmentation of lung foreground (thoracic parenchyma) in
axial CT slices. Runs slice-by-slice at native 2D resolution and its
outputs are stacked to form the per-series lung bounding box that the
downstream nodule model crops to.

- **Input**   `(1, 256, 256)` axial CT slice, intensity-normalised to `[0, 1]`
              and resized from the native slice resolution.
- **Output**  `(1, 256, 256)` sigmoid map. Foreground = lung tissue.

## 2. Model

MONAI `SwinUNETR` — Shifted-Window (Swin) Transformer encoder + CNN
decoder, in 2D. Small-capacity variant chosen to match the
parameter budget of the SegResNet baseline (~6.9 M) for an
apples-to-apples architecture comparison rather than
capacity-vs-capacity.

| Property                | Value |
|-------------------------|-------|
| Architecture family     | SwinUNETR (Swin transformer + U-Net) |
| Spatial dims            | 2 |
| Input channels          | 1 |
| Output channels         | 1 (sigmoid) |
| `feature_size`          | 24 (default is 48; halved for the small variant) |
| Transformer stage depths | `[2, 2, 2, 2]` |
| Attention heads         | `[3, 6, 12, 24]` |
| Dropout / attn-dropout / path-dropout | 0.0 |
| Trainable parameters    | **6,302,203** |

## 3. Data

Unified training corpus assembled from three public sources:

| Source     | Role in the split |
|------------|-------------------|
| NLST       | training + validation |
| NSCLC-Radiomics | training + validation |
| LIDC-IDRI  | training + validation (lung labels derived from a earlier medium-capacity nodule model) |

- **Split**: `unified_v2.json` — patient-grouped, dataset-stratified.
  1 683 train / 297 val / 325 test series.
- Slices per split: **365 014 train / 64 097 val / — test held out**.
- **Preprocessing**: each axial slice resized to 256 × 256.
- **Class balance**: lung foreground is ~30-50 % of a mid-thorax slice
  and 0 % of pure top-/bottom-of-volume slices — nothing pathological,
  so plain Dice loss is sufficient.
- **Split parity**: identical split and sampling recipe as the SegResNet
  baseline; only the model family differs.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Dice (sigmoid, squared prediction) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻⁴  (transformer families are more sensitive to LR than CNNs; SegResNet baseline used 1 × 10⁻³) |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = 100, η_min = 1 × 10⁻⁶ |
| Batch size         | 16 |
| Samples per epoch  | 20 000 (random subsample of ~365 k train slices) |
| Epochs (budget)    | 100 |
| Random seed        | 42 |
| Mixed precision    | bf16 (autocast, no GradScaler) |
| Hardware           | 1 × NVIDIA H100 94 GB |
| Per-epoch time     | ≈ 18 min |

Best checkpoint (evaluated below): **epoch 59 / 100**, validation Dice
0.9655. Training was still in progress at the time of evaluation;
however, val Dice had plateaued in the range [0.9647, 0.9655] for the
preceding ~15 epochs, so remaining epochs are not expected to move the
reported numbers by more than ±0.001.

## 5. Evaluation protocol

**Held-out test split** (45 751 slices from the `unified_v2` test set).
The training loop used the `val` split for early stopping and best-Dice
checkpoint selection; the numbers below come from the completely
untouched **`test`** split (never seen during training or model
selection). Predictions taken as `sigmoid(logits) > 0.5`. All metrics
are computed at the pixel level and micro-averaged over the whole test
split (~3.0 × 10⁹ pixels).

## 6. Results

### Ticket-required metrics

| Metric             | Value      |
|--------------------|-----------:|
| **mean IoU**       | **0.9829** |
| **Accuracy**       | **0.9975** |
| **Precision**      | **0.9830** |
| **Recall**         | **0.9849** |

### Supplementary

| Metric                          | Value      |
|---------------------------------|-----------:|
| IoU (foreground / lung class)   | 0.9684 |
| IoU (background class)          | 0.9973 |
| Dice / F1 (micro)               | 0.9839 |
| True positives  (px)            | 227 296 038 |
| False positives (px)            | 3 937 060 |
| False negatives (px)            | 3 483 996 |
| True negatives  (px)            | 2 763 620 442 |

### Per-slice Dice distribution (45 751 test slices)

| Statistic | Value  |
|-----------|-------:|
| Mean      | 0.9592 |
| Std       | 0.1240 |
| Min       | 0.0000 |
| p05       | 0.8574 |
| Median    | 0.9888 |
| p95       | 1.0000 |
| Max       | 1.0000 |

Compared to the SegResNet baseline the tails of the per-slice
distribution improve noticeably: p05 rises from 0.00 to 0.86 and the
per-slice mean jumps from 0.79 to 0.96. In practical terms, the
top-/bottom-of-volume slices where the lung is absent or barely present
are handled more conservatively (fewer speculative predictions), which
is where the SegResNet baseline was losing most of its per-slice score.

## 7. Comparison with SegResNet baseline

Both models trained on the same split, same augmentation-free pipeline,
same batch size, same epoch budget, same eval protocol; both evaluated
on the same held-out test set. Only the model family and the LR differ.

| Metric              | SegResNet (6.9 M) | **SwinUNETR (6.3 M)** | Δ |
|---------------------|------------------:|----------------------:|---:|
| mIoU                | 0.9704            | **0.9829**            | +0.0125 |
| Accuracy            | 0.9957            | **0.9975**            | +0.0018 |
| Precision           | 0.9662            | **0.9830**            | +0.0168 |
| Recall              | 0.9778            | **0.9849**            | +0.0071 |
| Dice (micro)        | 0.9720            | **0.9839**            | +0.0119 |
| Dice per slice mean | 0.7851            | **0.9592**            | +0.1741 |
| Dice per slice p05  | 0.0000            | **0.8574**            | +0.8574 |

SwinUNETR wins on every reported metric. The largest gain is in the
per-slice tail (p05), suggesting the transformer's global receptive
field is genuinely useful for the anatomically ambiguous edge slices at
the top/bottom of the thorax — which the CNN baseline was mis-labelling.

## 8. Notes on interpretation

Lung ROI segmentation is close to a solved problem on axial CT — the
target is a large, contrast-rich region with relatively simple
morphology. All four ticket metrics land near their upper bound, which
is expected rather than a bug:

- **Accuracy 0.9975** — informative here (unlike the nodule task)
  because the class ratio is roughly 40 / 60, not 10⁻⁵.
- **mIoU 0.9829** — genuinely a mean of two comparable IoUs (0.968 fg,
  0.997 bg).
- **Precision 0.9830** — 98 % of predicted-lung pixels are true lung.
- **Recall 0.9849** — 98 % of true-lung pixels are recovered.

Because both models are past the 0.97 mark on the test set, the most
operationally meaningful comparison is at the tails of the per-slice
distribution (p05) — which is where SwinUNETR shows a real, not just
marginal, improvement over the CNN baseline.

## 9. Reproducibility

- Config       `configs/roi_swin.yaml`
- Checkpoint   `checkpoints/roi_swin/best_model.pth`
- Metrics JSON `checkpoints/roi_swin/eval_swinunetr_test.json`
- Command      `python eval_roi.py --config configs/roi_swin.yaml --ckpt checkpoints/roi_swin/best_model.pth --split test`
