# Lung ROI Segmentation — SegResNet (2D) — sliding-window (256²), native resolution

## 1. Task

Per-slice binary lung segmentation of axial CT slices at the native
1 mm grid. Same task as the earlier resize-recipe ROI models — the
difference is the recipe: **256×256 patch training + 2D sliding-window
inference at native resolution** instead of resizing each slice to
256×256 (which distorts the lung outline on non-square slices).

- **Input**   `(1, H, W)` axial slice, intensity-normalised to `[0, 1]`,
              native voxel grid (270–500 px per side in this corpus).
- **Training input** 256×256 patches via `RandCropByPosNegLabeld`
              (`pos:neg = 1:1`, 4 patches/slice); slices smaller than the
              patch are zero-padded to 256×256.
- **Inference** MONAI `sliding_window_inference`, ROI `(256, 256)`,
              `overlap = 0.5`, Gaussian blending, sigmoid > 0.5.
- **Output**  `(1, H, W)` binary lung mask.

## 2. Model

SegResNet (residual U-Net), `spatial_dims: 2`, `init_filters: 16`,
encoder blocks `[1, 2, 2, 4, 4]`, decoder blocks `[1, 1, 1, 1]`,
dropout 0.1, 1-channel sigmoid head.
Trainable parameters: **6,904,401**.

## 3. Data

`unified` split, slices with lung annotation (NSCLC-Radiomics + NLST):
**365,014 train / 64,097 val / 45,751 test** slices. Patient-grouped,
dataset-stratified split, seed 42.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Dice (sigmoid, squared-pred) |
| Optimizer          | Adam, LR 1 × 10⁻³, weight decay 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = 100, η_min = 1 × 10⁻⁶ |
| Batch size         | 4 slices × 4 patches = 16 patches / step |
| Samples per epoch  | 20 000 slices (random subsample) |
| Epochs             | 100 |
| Mixed precision    | bf16 |
| Best val Dice      | 0.9691 |

## 5. Results — held-out test split (45,751 slices)

| Metric             | Value |
|--------------------|------:|
| **Dice / F1 (micro)** | **0.9823** |
| Precision          | 0.9832 |
| Recall             | 0.9814 |
| mIoU               | 0.9814 |
| IoU (lung)         | 0.9652 |

### Per-slice Dice distribution

| Statistic | Value |
|-----------|------:|
| Mean      | 0.9511 |
| Std       | 0.1560 |
| Median    | 0.9891 |
| p05       | 0.8352 |
| p95       | 1.0000 |

Unlike the resize-recipe SegResNet baseline (per-slice p05 = 0.0), both
SW-trained ROI models keep a usable Dice on the near-empty apex/base
slices — the native-resolution patch recipe removes that failure mode.

## 6. Reproducibility

- Config       `configs/roi_sw.yaml`
- Checkpoint   `checkpoints/roi_sw/best_model.pth`
- Metrics JSON `results_eval/eval_roi_sw_test.json`
- Command      `python eval_roi.py --config configs/roi_sw.yaml --ckpt checkpoints/roi_sw/best_model.pth --split test`
