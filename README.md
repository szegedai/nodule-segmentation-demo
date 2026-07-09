# Nodule segmentation — reproduction demo

A simplified, non-cluster copy of the production Stage-2 fine nodule
training pipeline. Each of the three bundled configs reproduces one of
the trained models reported in the paper — same model, same loss, same
split, same bboxes, same seed.

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

To run the full two-stage pipeline (ROI → bbox → nodule) on a raw CT
volume, download one ROI checkpoint and one nodule checkpoint from
HuggingFace and hand them to `inference.py`:

```bash
pip install huggingface_hub                     # if not already installed
huggingface-cli login                           # first time only

huggingface-cli download Kakimaki00/roi-swinunetr-2d          --local-dir ./ckpts/roi
huggingface-cli download Kakimaki00/nodule-segresnet-3d-wide  --local-dir ./ckpts/nodule

python inference.py \
    --ct         path/to/case.npz \
    --roi-dir    ./ckpts/roi \
    --nodule-dir ./ckpts/nodule \
    --output     nodule_mask.npz
```

The input CT is expected as a `.npz` with a `data` array of shape
`(1, H, W, D)`, float32, values in `[0, 1]` (same normalisation the
training pipeline uses). The output nodule mask is written as
`{'data': (1, H, W, D) uint8}` at the input's native resolution.
`inference.py --help` lists the two alternative flag pairs
(`--roi-weights` + `--roi-config`) if you'd rather pass paths directly.

## Models

| Config    | Architecture                | Params  | Batch | Split           | Epochs | Best val Dice ‡  | Recall | Precision |
|-----------|-----------------------------|---------|-------|-----------------|--------|-------------------|--------|-----------|
| `v6.yaml` | SegResNet (init_filters=16) | 20.7 M  | 4     | `unified_v2`    | 400    | 0.525            | 0.743  | 0.608     |
| `v7.yaml` | SegResNet (init_filters=32) | 82.7 M  | 2     | `unified` †     | 1000   | **0.589**        | 0.774  | 0.693     |
| `v9.yaml` | DynUNet (6-level)           | 31.2 M  | 2     | `unified_v2`    | 400    | 0.538            | 0.662  | 0.648     |

All three use the same shared training recipe: Focal Tversky + weighted
CE (α=0.3, β=0.7, γ=2.0, λ_ce=0.1, ce_nod=100), Adam (lr=1e-5, wd=1e-5),
CosineAnnealingLR (T_max=epochs, η_min=1e-6), bf16 AMP, seed 42.

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
```

Both file series use the same series-UID basenames. The train/val split
JSONs and the per-series lung bboxes are bundled in this repo — see below.

## Bundled artifacts

- `data/splits/unified.json` — original 70/15/15 patient-grouped
  train/val/test split (1609 / 345 / 351 series). Used by v7 only.
- `data/splits/unified_v2.json` — balanced re-split (1683 / 297 / 325
  series). Used by v6, v9, and the ROI model. This is the split the paper
  reports on.
- `processed/bboxes_unified.json` — per-series lung bboxes. NLST + NSCLC
  cases use bboxes derived from the ground-truth 2D lung ROI labels; LIDC
  cases (which lack ROI labels) use bboxes from the frozen medium nodule
  model.

All bundled files are the exact ones the reported models were trained
with, so reproduction is byte-for-byte deterministic (given the seed).

**Trained checkpoints are hosted separately on HuggingFace** (private
during paper review — request access if you need them):

| Model  | HuggingFace repo |
|--------|------------------|
| ROI SegResNet 2D          | [Kakimaki00/roi-segresnet-2d](https://huggingface.co/Kakimaki00/roi-segresnet-2d) |
| ROI SwinUNETR 2D          | [Kakimaki00/roi-swinunetr-2d](https://huggingface.co/Kakimaki00/roi-swinunetr-2d) |
| Nodule SegResNet 3D (v6)  | [Kakimaki00/nodule-segresnet-3d-small](https://huggingface.co/Kakimaki00/nodule-segresnet-3d-small) |
| Nodule SegResNet 3D (v7)  | [Kakimaki00/nodule-segresnet-3d-wide](https://huggingface.co/Kakimaki00/nodule-segresnet-3d-wide) |
| Nodule DynUNet 3D (v9)    | [Kakimaki00/nodule-dynunet-3d](https://huggingface.co/Kakimaki00/nodule-dynunet-3d) |

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

## Files

```
train.py                       training loop (Adam + cosine LR, bf16, ckpt save/resume)
inference.py                   end-to-end ROI → bbox → nodule pipeline on a raw CT
eval_roi.py                    ROI evaluation: mIoU / Accuracy / Precision / Recall
model.py                       architecture dispatch (SegResNet / DynUNet / SwinUNETR)
loss.py                        FocalTverskyCELoss + DiceLoss + build_loss factory
dataset.py                     NoduleFineCropDataset + Roi2DDataset + factory
transforms.py                  train / val transform pipelines
configs/{v6,v7,v9}.yaml        one per nodule model variant
configs/{roi,roi_swin}.yaml    ROI (2D lung foreground) training configs
data/splits/unified.json       v1 split (used by v7)
data/splits/unified_v2.json    v2 balanced split (used by v6, v9, ROI)
processed/bboxes_unified.json  bundled per-series lung bboxes
reports/                       markdown + PDF metric reports (one per model)
requirements.txt
```

## Compute expectations

Wall-clock on a single H100 94 GB:

| Config          | Epoch time | Full run |
|-----------------|-----------:|---------:|
| v6 (400)        |  ~13 min   |  ~3.5 d  |
| v7 (1000)       |  ~18 min   | ~12 d    |
| v9 (400)        |  ~13 min   |  ~3.5 d  |
| roi (100)       |   ~5 min   |  ~8 h    |
| roi_swin (100)  |  ~18 min   |  ~30 h   |

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
