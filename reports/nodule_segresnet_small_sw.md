# Lung Nodule Segmentation — SegResNet (small) — sliding-window (128³), CE weight 100

## 1. Task

Voxel-level segmentation of pulmonary nodules in 3D chest CT volumes,
lung-bbox-cropped. Same task as the earlier resize-recipe
`segresnet_small` model — the only difference is inference recipe: **sliding-window over 128³
patches at native resolution** rather than a single forward on a 256³
resample.

- **Input**   `(1, H, W, D)` CT crop, intensity-normalised to `[0, 1]`,
              lung-bbox + 20-voxel padding, kept at native voxel grid.
- **Training input** 128³ patches sampled via `RandCropByPosNegLabeld` at
              a ratio of `pos:neg = 2:1`, `num_samples = 4` per volume.
- **Inference** MONAI `sliding_window_inference`, ROI `(128, 128, 128)`,
              `overlap = 0.5`, `sw_batch_size = 4`, Gaussian
              blending, single output at the input resolution.
- **Output**  `(2, H, W, D)` softmax logits: 0 = background, 1 = nodule.

## 2. Model

| Property                | Value |
|-------------------------|-------|
| Architecture family     | SegResNet (small) |
| Spatial dims            | 3 |
| Input channels          | 1 |
| Output channels         | 2 (softmax) |
| Init. feature width   | 16 |
| Trainable parameters    | **20,663,538** |

## 3. Data

Same unified corpus + split as the resize-recipe 256³ models:

| Source          | Role |
|-----------------|------|
| NLST            | train + val + test |
| NSCLC-Radiomics | train + val + test |
| LIDC-IDRI       | train + val + test |

- **Split**: `unified.json` — patient-grouped, dataset-stratified.
  1 683 train / 297 val / 325 test series.
- **Lung crop**: per-series 3D bbox from `bboxes_unified.json` (lung ROI +
  medium-model fallback), + 20-vox padding. No resampling.
- **Class imbalance**: nodule voxels are ~10⁻⁵ of the total — see §7.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Focal Tversky + weighted CE (α=0.3, β=0.7, γ=2.0, λ_CE=0.1 / 0.3, nodule class weight = 100) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻⁵ |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = 400, η_min = 1 × 10⁻⁶ |
| Batch size         | 2 volumes × 4 patches = 8 patches / step |
| Patch size         | 128³ (positive-biased random crop, `pos:neg = 2:1`) |
| Epochs             | 400 |
| Random seed        | 42 |
| Mixed precision    | bf16 (autocast, no GradScaler) |
| Augmentation       | 3D flips, 90° rotations, elastic rotation, zoom, intensity scale/shift, Gaussian noise/blur, contrast |
| Hardware           | 1 × NVIDIA H100 80 GB (fat01) |

Best checkpoint recorded at **epoch 369 / 400** (see `_meta` in
metrics JSON).

## 5. Evaluation protocol

Evaluated on the `unified` **test** split (325 series held
out — never seen during training or model selection). Predictions taken
as `argmax` over the 2-channel softmax output from
`sliding_window_inference`. All voxel-level metrics are micro-averaged
over the whole split.

## 6. Results

### Ticket-required metrics

| Metric             | Value      |
|--------------------|-----------:|
| **mean IoU**       | **0.6746** |
| **Accuracy**       | **0.9987** |
| **Precision**      | **0.3774** |
| **Recall**         | **0.8313** |

### Supplementary

| Metric                          | Value      |
|---------------------------------|-----------:|
| IoU (foreground / nodule class) | 0.3506 |
| IoU (background class)          | 0.9987 |
| Dice / F1 (micro)               | 0.5192 |

### Per-case Dice distribution (325 test cases)

| Statistic | Value  |
|-----------|-------:|
| Mean      | 0.4144 |
| Std       | 0.2597 |
| Min       | 0.0000 |
| p05       | 0.0135 |
| p25       | 0.1865 |
| Median    | 0.4071 |
| p75       | 0.6428 |
| p95       | 0.8253 |
| Max       | 0.9327 |

Voxels evaluated: 7,849,317,259. Cases: 325.

## 7. Notes on interpretation

Sliding-window inference at 128³ trades global context (a 256³ resample
sees the whole lung in one pass) for preserved resolution (no down-
sampling of small nodules). On this unified corpus the trade tends to
be neutral-to-negative on nodule Dice — small-nodule gains do not
compensate for lost global context on typical test cases.

Class-imbalance caveats are unchanged from the 256³ reports: Accuracy
is trivially ~1, mIoU is dominated by IoU_bg, and Dice / F1 (micro) is
the honest voxel-level summary. Per-case mean weights each patient
equally regardless of nodule volume.

## Instance-level metrics (per-nodule)

Connected-component analysis with 26-connectivity, pooled over the
held-out test split, on the native voxel grid.

- **Instance recall** — GT nodules touched by at least one predicted
  voxel, over all GT nodules.
- **Instance precision** — predicted components touching at least one
  GT voxel, over all predicted components.

| Metric | Value |
|---|---:|
| GT nodules (components) | 947 |
| GT nodules hit          | 830 |
| **Instance recall**     | **0.8765** |
| Predicted components    | 9,977 |
| Predicted components hitting GT | 836 |
| **Instance precision**  | **0.0838** |

No minimum-size filtering is applied — every predicted component
counts, so single-voxel false positives lower instance precision.
Produced by `scripts/instance_metrics.py --config configs/segresnet_small_sw.yaml
--ckpt <ckpt> --split test` (JSON: `results_instance/segresnet_small_sw.json`).

## 8. Reproducibility

- Config       `/home/werner/nodule-training-demo/configs/segresnet_small_sw.yaml`
- Checkpoint   `stage2_fine/checkpoints_segresnet_small_sw/best_model.pth`
- Metrics JSON `/tmp/claude-1002/-home-werner-nodule-segmentation/c2faa0a0-5a75-48f5-8288-377df40092b7/scratchpad/sw_metrics/segresnet_small_sw.json`
- Command      `python stage2_fine/eval_metrics.py --config /home/werner/nodule-training-demo/configs/segresnet_small_sw.yaml --checkpoint stage2_fine/checkpoints_segresnet_small_sw/best_model.pth --split test`
