# Nodule segmentation — reproduction demo

A simplified, non-cluster copy of the production training pipeline
covering all three tasks the paper reports:

- **ROI** — 2D per-slice lung foreground segmentation (feeds the two-stage pipeline)
- **Nodule** — 3D nodule segmentation on lung-bbox-cropped input
- **Joint (end-to-end)** — 3D full-volume 3-class {bg, lung, nodule} model, no bbox stage

Each bundled config reproduces one of the trained models reported in the
paper — same model, same loss, same split, same bboxes, same seed. All
nine trained checkpoints are hosted on HuggingFace under
[`Kakimaki00`](https://huggingface.co/Kakimaki00).

## Quickstart

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/unified          # dir with ct_3d/, nodule_sem_seg_3d/
python train.py --config configs/v7.yaml   # reproduce the paper's best model
```

That's it. Every hyperparameter, the persisted train/val split, and the
precomputed lung bboxes are all bundled — the only external input is the
unified CT corpus itself.

To resume a run after interruption:

```bash
python train.py --config configs/v7.yaml --resume
```

## Inference on a new CT

Two pipelines supported by `inference.py`:

**Two-stage** (ROI → bbox → nodule) — matches how v6/v7/v9 were trained:

```bash
pip install huggingface_hub                     # if not already installed
huggingface-cli login                           # first time only

huggingface-cli download Kakimaki00/roi-swinunetr-2d          --local-dir ./ckpts/roi
huggingface-cli download Kakimaki00/nodule-segresnet-3d-wide  --local-dir ./ckpts/nodule

python inference.py \
    --ct         path/to/case.npz \
    --roi-dir    ./ckpts/roi \
    --nodule-dir ./ckpts/nodule \
    --output     nodule_mask.npz            # (1, H, W, D) uint8 binary
```

**Joint end-to-end** (single 3-class model, no bbox stage):

```bash
huggingface-cli download Kakimaki00/joint-dynunet-3d-ex-lidc --local-dir ./ckpts/joint

python inference.py \
    --ct        path/to/case.npz \
    --joint-dir ./ckpts/joint \
    --output    joint_mask.npz              # (1, H, W, D) uint8 in {0=bg, 1=lung, 2=nodule}
```

The input CT is expected as a `.npz` with a `data` array of shape
`(1, H, W, D)`, float32, values in `[0, 1]` (same normalisation the
training pipeline uses). Two-stage output is a binary nodule mask;
joint output is a 3-class label. `inference.py --help` lists the
alternative flag pairs (`--roi-weights` + `--roi-config`, etc.) if
you'd rather pass paths directly.

## Models

### Nodule (two-stage: ROI → bbox → nodule)

| Config    | Architecture                | Params  | Batch | Split           | Epochs | Best val Dice ‡  | Recall | Precision |
|-----------|-----------------------------|---------|-------|-----------------|--------|-------------------|--------|-----------|
| `v6.yaml` | SegResNet (init_filters=16) | 20.7 M  | 4     | `unified_v2`    | 400    | 0.525            | 0.743  | 0.608     |
| `v7.yaml` | SegResNet (init_filters=32) | 82.7 M  | 2     | `unified` †     | 1000   | **0.589**        | 0.774  | 0.693     |
| `v9.yaml` | DynUNet (6-level)           | 31.2 M  | 2     | `unified_v2`    | 400    | 0.538            | 0.662  | 0.648     |

### Joint end-to-end (single 3-class model, no bbox stage)

| Config                            | Architecture          | Params  | Batch | Split               | Best mIoU (3-cls) | Lung Dice | Nodule Dice |
|-----------------------------------|-----------------------|---------|-------|---------------------|------------------:|----------:|------------:|
| `joint_segresnet_ex_lidc.yaml`    | SegResNet             | 20.7 M  | 4     | `unified_v2_ex_lidc`|      0.8147       |   0.9814  |    0.6546   |
| `joint_segresnet_pseudo.yaml`     | SegResNet             | 20.7 M  | 4     | `unified_v2`        |      0.8057       |   0.9754  |    0.6421   |
| `joint_dynunet_ex_lidc.yaml`      | DynUNet (3D U-Net)    | 31.2 M  | 2     | `unified_v2_ex_lidc`|    **0.8213**     |   0.9800  |  **0.6752** |
| `joint_dynunet_pseudo.yaml`       | DynUNet (3D U-Net)    | 31.2 M  | 2     | `unified_v2`        |      0.8031       |   0.9746  |    0.6364   |

The `ex_lidc` variants train only on NLST + NSCLC (they have GT lung
labels). The `pseudo` variants add LIDC-IDRI back in, using the trained
2D SegResNet ROI model as pseudo-GT for LIDC's missing lung labels.
Across both architectures, **`ex_lidc` outperforms `pseudo`** — the
pseudo-labels' noise slightly hurts the lung head's supervision signal
and the added LIDC diversity does not compensate. Full metric
breakdowns (mIoU, Accuracy, per-class Precision/Recall, per-case Dice
distribution) are in [`reports/joint/`](reports/joint/).

All seven 3D configs use the same shared training recipe: Focal Tversky
(α=0.3, β=0.7, γ=2.0) + weighted CE (λ_ce=0.1, nodule class weight 100),
Adam (lr=1e-5, wd=1e-5), CosineAnnealingLR (T_max=epochs, η_min=1e-6),
bf16 AMP, seed 42. The nodule variants use a 2-class softmax head; the
joint variants use a 3-class softmax head and a K-class generalisation
of the same loss.

**† Note on splits.** v6 and v9 are retrained on the balanced
`unified_v2.json` split (the same split used to train the ROI model), so
the whole pipeline is on a single consistent split. v7 was trained on the
earlier `unified.json` split before we settled on the v2 split; because
retraining it risks not reproducing its peak of 0.589 (which was reached
at epoch 545 on the v1 split, well past the 400-epoch budget we use for
the retrains), we kept the original v7 checkpoint and its `unified.json`
split. Both split JSONs are bundled here.

**‡ Metric aggregation.** *Best val Dice* is the per-case mean Dice on
the val split, matching the training loop's `nodule_dice`. *Recall* and
*Precision* are voxel-level, micro-averaged over the whole val split;
they are the numbers a strict "per-voxel classifier" reading of the
model produces. Full metric breakdown (mIoU, per-case distribution) is
in [`reports/`](reports/).

**3D SwinUNETR for the nodule task was tested (`feature_size=48`, 62 M
params, gradient checkpointing) but underperformed both the wider
SegResNet (v7: 0.589) and DynUNet (v9), while being ~2.5× slower per
epoch than either. Not retrained; not reported in the paper. The
*2D* SwinUNETR appears only in the ROI section below and performs
well.**

## Data layout

`DATA_ROOT` must point at a directory laid out like this:

```
$DATA_ROOT/
  ct_3d/<series_uid>.npz               ['data']  (1, H, W, D)  float32 [0,1]
  nodule_sem_seg_3d/<series_uid>.npz   ['data']  (1, H, W, D)  binary   {0,1}
  # optional, extra for ROI + joint training:
  ct_2d/<series_uid>_<NNNN>.npz        ['data']  (1, H, W)     float32 [0,1]
  roi_sem_seg_2d/<series_uid>_<NNNN>.npz ['data'] (1, H, W)    binary   {0,1}
  # produced by scripts/build_lung_3d.py, required for joint training:
  lung_sem_seg_3d/<series_uid>.npz     ['data']  (1, H, W, D)  binary   {0,1}
```

All file series use the same series-UID basenames. The train/val split
JSONs and the per-series lung bboxes are bundled in this repo — see below.

## Bundled artifacts

- `data/splits/unified.json` — original 70/15/15 patient-grouped
  train/val/test split (1609 / 345 / 351 series). Used by v7 only.
- `data/splits/unified_v2.json` — balanced re-split (1683 / 297 / 325
  series). Used by v6, v9, ROI, and the joint `pseudo` variants. This
  is the split the paper reports on.
- `data/splits/unified_v2_ex_lidc.json` — same split with LIDC-IDRI
  filtered out (1110 / 196 / 129 series). Used by the joint `ex_lidc`
  variants.
- `processed/bboxes_unified.json` — per-series lung bboxes. NLST + NSCLC
  cases use bboxes derived from the ground-truth 2D lung ROI labels; LIDC
  cases (which lack ROI labels) use bboxes from the frozen medium nodule
  model.

All bundled files are the exact ones the reported models were trained
with, so reproduction is byte-for-byte deterministic (given the seed).

**Trained checkpoints are hosted separately on HuggingFace** (private
during paper review — request access if you need them):

| Task        | Model                              | HuggingFace repo |
|-------------|------------------------------------|------------------|
| ROI (2D)    | SegResNet                          | [Kakimaki00/roi-segresnet-2d](https://huggingface.co/Kakimaki00/roi-segresnet-2d) |
| ROI (2D)    | SwinUNETR (small)                  | [Kakimaki00/roi-swinunetr-2d](https://huggingface.co/Kakimaki00/roi-swinunetr-2d) |
| Nodule (3D) | SegResNet, small (v6)              | [Kakimaki00/nodule-segresnet-3d-small](https://huggingface.co/Kakimaki00/nodule-segresnet-3d-small) |
| Nodule (3D) | SegResNet, wide (v7 — paper's best)| [Kakimaki00/nodule-segresnet-3d-wide](https://huggingface.co/Kakimaki00/nodule-segresnet-3d-wide) |
| Nodule (3D) | DynUNet / 3D U-Net (v9)            | [Kakimaki00/nodule-dynunet-3d](https://huggingface.co/Kakimaki00/nodule-dynunet-3d) |
| Joint (3D)  | SegResNet, ex-LIDC                 | [Kakimaki00/joint-segresnet-3d-ex-lidc](https://huggingface.co/Kakimaki00/joint-segresnet-3d-ex-lidc) |
| Joint (3D)  | SegResNet, pseudo-LIDC             | [Kakimaki00/joint-segresnet-3d-pseudo-lidc](https://huggingface.co/Kakimaki00/joint-segresnet-3d-pseudo-lidc) |
| Joint (3D)  | DynUNet, ex-LIDC (best joint)      | [Kakimaki00/joint-dynunet-3d-ex-lidc](https://huggingface.co/Kakimaki00/joint-dynunet-3d-ex-lidc) |
| Joint (3D)  | DynUNet, pseudo-LIDC               | [Kakimaki00/joint-dynunet-3d-pseudo-lidc](https://huggingface.co/Kakimaki00/joint-dynunet-3d-pseudo-lidc) |

Each HF repo contains `model.pth` (weights-only, `torch.save`d
state_dict), the exact `config.yaml` used at training time, and a model
card. See the [ROI model](#roi-model-optional) section for where to
place ROI checkpoints locally.

## ROI model (optional)

Two 2D lung foreground models are shipped as trainable configs:

| Config           | Architecture              | Params  | Val mIoU | Val Dice (micro) | Val Precision | Val Recall |
|------------------|---------------------------|---------|---------:|----------------:|--------------:|-----------:|
| `roi.yaml`       | SegResNet 2D              |  6.9 M  |   0.980  |   0.983         |   0.980       |   0.986    |
| `roi_swin.yaml`  | SwinUNETR 2D (small)      |  6.3 M  | **0.985**| **0.987**       | **0.986**     | **0.988**  |

Both are trained on the same `unified_v2` split as v6/v9 and use the
same 2D dataset (`ct_2d/` + `roi_sem_seg_2d/`). See
[`reports/roi_segresnet.md`](reports/roi_segresnet.md) and
[`reports/roi_swinunetr.md`](reports/roi_swinunetr.md) for full metric
breakdowns.

**You do not need to train or run either ROI model to reproduce
v6 / v7 / v9.** Those nodule configs read the precomputed
`bboxes_unified.json` and never open an ROI checkpoint.

The ROI model matters for two things the demo does *not* cover:

1. **Inference on a new CT.** No GT lung mask exists for an unseen scan,
   so the ROI model produces per-slice lung masks → derive a 3D bbox →
   crop → feed to the nodule model.
2. **Retraining the pipeline on data without GT lung labels.** The ROI
   model is the practical substitute for GT when the corpus lacks
   `roi_sem_seg_2d/` annotations.

### Training an ROI model from scratch

```bash
export DATA_ROOT=/path/to/unified              # needs ct_2d/ and roi_sem_seg_2d/
python train.py --config configs/roi.yaml      # SegResNet 2D
python train.py --config configs/roi_swin.yaml # SwinUNETR 2D (small)
```

The same `train.py` handles both tasks; the `task:` field in the YAML
routes to the 2D ROI path (`Roi2DDataset`, sigmoid + threshold
post-proc, plain `DiceLoss`) instead of the 3D nodule path.

### Downloading a pre-trained ROI checkpoint

Trained ROI checkpoints are hosted on HuggingFace (private during paper
review — request access if you need them):

- SegResNet: [Kakimaki00/roi-segresnet-2d](https://huggingface.co/Kakimaki00/roi-segresnet-2d)
- SwinUNETR: [Kakimaki00/roi-swinunetr-2d](https://huggingface.co/Kakimaki00/roi-swinunetr-2d)

Each repo ships `model.pth` (weights-only). To use with `--resume` in
this repo, download and rename:

```bash
huggingface-cli download Kakimaki00/roi-segresnet-2d model.pth --local-dir checkpoints/roi
mv checkpoints/roi/model.pth checkpoints/roi/best.pth

huggingface-cli download Kakimaki00/roi-swinunetr-2d model.pth --local-dir checkpoints/roi_swin
mv checkpoints/roi_swin/model.pth checkpoints/roi_swin/best_model.pth
```

Note that the HF weights are **weights-only** (`state_dict`, no
optimizer state), so `--resume` will restart the optimizer from scratch
rather than continuing the exact original schedule. For inference-only
use this is fine; for continuing a training run, use the internal
full-checkpoint path instead.

## Joint end-to-end training

The four `joint_*.yaml` configs train a single 3-class model
{bg, lung, nodule} directly on the full CT — no ROI stage, no bbox
crop. Two data variants:

- **`ex_lidc`** — NLST + NSCLC only. Lung labels are ground-truth (2D
  masks in `roi_sem_seg_2d/`, stacked into 3D per series).
- **`pseudo`**  — full corpus (adds LIDC-IDRI). LIDC lacks GT lung
  labels, so they are produced by running the trained 2D SegResNet ROI
  model on each LIDC slice as pseudo-GT.

Both variants read a per-series 3D lung mask from
`$DATA_ROOT/lung_sem_seg_3d/`. That directory is **not bundled** — it
depends on your CT corpus, and (for LIDC) on the ROI model — so
generate it locally once before joint training:

```bash
huggingface-cli download Kakimaki00/roi-segresnet-2d --local-dir ./ckpts/roi
export DATA_ROOT=/path/to/unified

python scripts/build_lung_3d.py \
    --data-root         $DATA_ROOT \
    --out-dir           $DATA_ROOT/lung_sem_seg_3d \
    --roi-config        ./ckpts/roi/config.yaml \
    --roi-checkpoint    ./ckpts/roi/model.pth \
    --split-in          data/splits/unified_v2.json \
    --split-out-ex-lidc data/splits/unified_v2_ex_lidc.json \
    --manifest-out      lung_source_manifest.json
```

This takes ~1 h on an H100 (mostly LIDC inference). The bundled
`data/splits/unified_v2_ex_lidc.json` is the exact output of this
script — regenerating it will overwrite it byte-for-byte.

Then train:

```bash
python train.py --config configs/joint_dynunet_ex_lidc.yaml    # best joint model
python train.py --config configs/joint_segresnet_ex_lidc.yaml
python train.py --config configs/joint_dynunet_pseudo.yaml
python train.py --config configs/joint_segresnet_pseudo.yaml
```

Evaluate:

```bash
python eval_metrics_joint.py \
    --config configs/joint_dynunet_ex_lidc.yaml \
    --ckpt   checkpoints/joint_dynunet_ex_lidc/best_model.pth
```

## Files

```
train.py                            training loop (Adam + cosine LR, bf16, ckpt save/resume)
inference.py                        two-stage OR joint inference on a raw CT
eval_roi.py                         ROI evaluation: mIoU / Accuracy / Precision / Recall
eval_metrics_joint.py               joint (3-class) evaluation, same metric set
model.py                            architecture dispatch (SegResNet / DynUNet / SwinUNETR)
loss.py                             FocalTverskyCELoss (2-cls) + MulticlassFocalTverskyCELoss (K-cls) + DiceLoss
dataset.py                          NoduleFineCropDataset + Roi2DDataset + JointFullVolumeDataset
transforms.py                       train / val transform pipelines
configs/{v6,v7,v9}.yaml             two-stage nodule configs
configs/{roi,roi_swin}.yaml         2D ROI configs
configs/joint_*.yaml                joint (end-to-end) configs (4)
scripts/build_lung_3d.py            preprocessing for joint training (writes lung_sem_seg_3d/)
data/splits/unified.json            v1 split (used by v7)
data/splits/unified_v2.json         v2 balanced split (used by v6, v9, ROI, joint pseudo)
data/splits/unified_v2_ex_lidc.json v2 split with LIDC filtered out (used by joint ex_lidc)
processed/bboxes_unified.json       bundled per-series lung bboxes
reports/                            markdown + PDF metric reports (one per model)
requirements.txt
```

## Compute expectations

Wall-clock on a single H100 94 GB:

| Config                            | Epoch time | Full run |
|-----------------------------------|-----------:|---------:|
| v6 (400)                          |  ~13 min   |  ~3.5 d  |
| v7 (1000)                         |  ~18 min   | ~12 d    |
| v9 (400)                          |  ~13 min   |  ~3.5 d  |
| roi (100)                         |   ~5 min   |  ~8 h    |
| roi_swin (100)                    |  ~18 min   |  ~30 h   |
| joint_segresnet_ex_lidc (400)     |  ~13 min   |  ~3.5 d  |
| joint_segresnet_pseudo (400)      |  ~19 min   |  ~5 d    |
| joint_dynunet_ex_lidc (400)       |  ~13 min   |  ~3.5 d  |
| joint_dynunet_pseudo (400)        |  ~19 min   |  ~5 d    |

Training writes:
- `checkpoints/<v>/best_model.pth` — best val-Dice checkpoint
- `checkpoints/<v>/last.pth` — latest epoch (for `--resume`)
- `checkpoints/<v>/epoch_XXXX.pth` — periodic snapshots (every 50 epochs)
- `runs/<v>/run/` — TensorBoard scalars (`tensorboard --logdir runs/`)

## Notes

- **AMP is bf16 only.** The production loop kept an fp16 GradScaler branch
  for legacy reasons; every reported model was trained in bf16, so the
  fp16 path is not preserved here. If your GPU doesn't support bf16
  (pre-Ampere), set `training.amp: false` in the config.
- **Non-finite skip.** If a batch produces a NaN/Inf loss or grad norm,
  the step is skipped and training continues — same behavior as
  production, without the diagnostic dump.
