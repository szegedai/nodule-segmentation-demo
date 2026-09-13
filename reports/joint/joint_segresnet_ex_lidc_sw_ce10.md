# Joint 3-class Lung Segmentation — SegResNet, ex-LIDC (NLST + NSCLC only) — sliding-window (128³), reduced CE weight

## 1. Task

End-to-end 3-class semantic segmentation of a full chest CT into
{background, lung, nodule}. Same task as the paper's
`joint_segresnet_ex_lidc` — the differences are a reduced nodule CE
weight (10 instead of 100) and the inference recipe: **sliding-window over 128³ patches at native
resolution** rather than a single forward on a 256³ resample.

- **Input**   `(1, H, W, D)` CT at native voxel grid.
- **Training input** 128³ patches sampled via `RandCropByLabelClassesd`
              with class-balanced ratios `[bg, lung, nodule] = [0.2, 0.3, 0.5]`,
              `num_samples = 4` per volume.
- **Inference** MONAI `sliding_window_inference`, ROI `(128, 128, 128)`,
              `overlap = 0.5`, `sw_batch_size = 4`, Gaussian
              blending, output at input resolution.
- **Output**  `(3, H, W, D)` softmax logits: 0 = bg, 1 = lung, 2 = nodule.

## 2. Model

| Property                | Value |
|-------------------------|-------|
| Architecture family     | SegResNet |
| Spatial dims            | 3 |
| Input channels          | 1 |
| Output channels         | 3 (softmax) |
| Init. feature width   | 16 |
| Trainable parameters    | **20,663,555** |

## 3. Data

| Source          | Role |
|-----------------|------|
| NLST            | train + val + test |
| NSCLC-Radiomics | train + val + test |
| LIDC-IDRI       | excluded (no GT lung labels available) |

- **Split**: `unified_v2_ex_lidc.json` — patient-grouped, dataset-stratified.
- **Preprocessing**: none beyond intensity clipping / normalization; the
  full CT enters the sliding-window inference at its native voxel grid.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Multiclass Focal Tversky + weighted CE (α=0.3, β=0.7, γ=2.0, λ_CE=0.1, class weights [1, 1, 10]) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻⁵ |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = 400, η_min = 1 × 10⁻⁶ |
| Batch size         | 2 volumes × 4 patches = 8 patches / step |
| Patch size         | 128³ (class-biased random crop, ratios `[bg, lung, nodule] = [0.2, 0.3, 0.5]`) |
| Epochs             | 400 |
| Random seed        | 42 |
| Mixed precision    | bf16 |
| Augmentation       | 3D flips, 90° rotations, elastic rotation, zoom, intensity scale/shift, Gaussian noise/blur, contrast |
| Hardware           | 1 × NVIDIA H100 80 GB (fat01) |

Best checkpoint recorded at **epoch 399 / 400**.

## 5. Evaluation protocol

Evaluated on the `unified_v2_ex_lidc` **test** split (129 series). All
voxel-level metrics micro-averaged over the whole split. Per-class
metrics computed against argmax predictions.

## 6. Results

### Ticket-required metrics (3-class average)

| Metric                     | Value      |
|----------------------------|-----------:|
| **mean IoU (3-cls)**       | **0.7627** |
| **mean IoU (lung + nodule)** | **0.6458** |
| **Accuracy**               | **0.996490** |
| **Macro Precision (lung + nodule)** | **0.6660** |
| **Macro Recall (lung + nodule)**    | **0.9235** |

### Per-class breakdown

| Class      | Dice (micro) | Recall | Precision | IoU    |
|------------|-------------:|-------:|----------:|-------:|
| background | 0.9982 | 0.9979 | 0.9985 | 0.9964 |
| lung       | 0.9797 | 0.9758 | 0.9836 | 0.9602 |
| nodule     | 0.4978 | 0.8711 | 0.3485 | 0.3314 |

### Per-case Dice distribution — nodule class (129 test cases)

| Statistic | Value  |
|-----------|-------:|
| Mean      | 0.4530 |
| Std       | 0.2694 |
| Min       | 0.0000 |
| p25       | 0.2219 |
| Median    | 0.4881 |
| p75       | 0.7019 |
| Max       | 0.9225 |

Voxels evaluated: 9,789,871,397. Cases: 129.

## 7. Notes on interpretation

Lung Dice at ≈ 0.98 is essentially saturated across all architectures
and inference recipes — the lung is a large, well-defined foreground
class. Nodule Dice is the real signal.

Sliding-window inference at 128³ trades global lung context for
preserved local resolution. On this corpus the trade tends to be
neutral-to-negative on nodule Dice compared to the 256³ resize
baseline — patches see only ~15% of a typical lung volume, which
limits anatomy-conditional reasoning.

## Instance-level metrics (per-nodule)

Connected-component analysis with 26-connectivity, pooled over the
held-out test split, on the native voxel grid.

- **Instance recall** — GT nodules touched by at least one predicted
  voxel, over all GT nodules.
- **Instance precision** — predicted components touching at least one
  GT voxel, over all predicted components.

| Metric | Value |
|---|---:|
| GT nodules (components) | 311 |
| GT nodules hit          | 216 |
| **Instance recall**     | **0.6945** |
| Predicted components    | 5,304 |
| Predicted components hitting GT | 3,482 |
| **Instance precision**  | **0.6565** |

No minimum-size filtering is applied — every predicted component
counts, so single-voxel false positives lower instance precision.
Produced by `scripts/instance_metrics.py --config configs/joint_segresnet_ex_lidc_sw_ce10.yaml
--ckpt <ckpt> --split test` (JSON: `results_instance/joint_segresnet_ex_lidc_sw_ce10.json`).

## 8. Reproducibility

- Config       `/home/werner/nodule-segmentation/configs/joint_segresnet_ex_lidc_sw_ce10.yaml`
- Checkpoint   `stage3_joint/checkpoints_segresnet_ex_lidc_sw_ce10/best_model.pth`
- Metrics JSON `/tmp/claude-1002/-home-werner-nodule-segmentation/c2faa0a0-5a75-48f5-8288-377df40092b7/scratchpad/ce10_test/joint_segresnet_ex_lidc.json`
- Command      `python stage3_joint/eval_metrics.py --config /home/werner/nodule-segmentation/configs/joint_segresnet_ex_lidc_sw_ce10.yaml --checkpoint stage3_joint/checkpoints_segresnet_ex_lidc_sw_ce10/best_model.pth --split test`
