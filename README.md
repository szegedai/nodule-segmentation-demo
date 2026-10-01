# Nodule segmentation — reproduction demo

A simplified, non-cluster copy of the production training pipeline
covering all three tasks the paper reports:

- **ROI** — 2D per-slice lung foreground segmentation (feeds the two-stage pipeline)
- **Nodule** — 3D nodule segmentation on lung-bbox-cropped input
- **Joint (end-to-end)** — 3D full-volume 3-class {bg, lung, nodule} model, no bbox stage

Each bundled config reproduces one of the trained models reported in the
paper — same model, same loss, same split, same bboxes, same seed. All
nine trained checkpoints are hosted on HuggingFace under
[`szabopeter`](https://huggingface.co/szabopeter).

## Quickstart

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/unified          # dir with ct_3d/, nodule_sem_seg_3d/
python train.py --config configs/segresnet_wide_sw_ce10.yaml   # reproduce the paper's best 3D baseline
```

That's it. Every hyperparameter, the persisted train/val split, and the
precomputed lung bboxes are all bundled — the only external input is the
unified CT corpus itself.

To resume a run after interruption:

```bash
python train.py --config configs/segresnet_wide.yaml --resume
```

## Inference on a new CT

Two pipelines supported by `inference.py`:

**Two-stage** (ROI → bbox → nodule) — matches how the two-stage nodule models were trained:

```bash
pip install huggingface_hub                     # if not already installed
huggingface-cli login                           # first time only

huggingface-cli download HalmosiL/roi-swinunetr-2d          --local-dir ./ckpts/roi
huggingface-cli download HalmosiL/nodule-segresnet-3d-wide  --local-dir ./ckpts/nodule

python inference.py \
    --ct         path/to/case.npz \
    --roi-dir    ./ckpts/roi \
    --nodule-dir ./ckpts/nodule \
    --output     nodule_mask.npz            # (1, H, W, D) uint8 binary
```

**Joint end-to-end** (single 3-class model, no bbox stage):

```bash
huggingface-cli download HalmosiL/joint-dynunet-3d-ex-lidc --local-dir ./ckpts/joint

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

## HTTP API

A FastAPI wrapper around the two-stage pipeline is under
[`api/`](api/). It ships the resize-recipe two-stage model combo
(SwinUNETR ROI + DynUNet nodule) and exposes a `POST /predict`
endpoint that takes a CT `.npz` and returns a binary nodule-mask
`.npz`:

```bash
pip install -r api/requirements.txt      # fastapi + uvicorn + huggingface_hub
export HF_TOKEN=...                      # for the private model repos
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@case.npz" -D headers.txt -o nodule_mask.npz
```

See [`api/README.md`](api/README.md) for full documentation, the
Python client example, and deployment notes.

## Models

All numbers below are on the held-out **test split** (see §Metric
aggregation‡). The val split was used only for early stopping and
best-checkpoint selection during training.

### Nodule (two-stage: ROI → bbox → nodule)

| Config                          | Architecture                | Params  | Batch | Split           | Epochs | Test mIoU | Test Recall | Test Precision |
|---------------------------------|-----------------------------|---------|-------|-----------------|--------|----------:|------------:|---------------:|
| `segresnet_small.yaml`       | SegResNet (init_filters=16) | 20.7 M  | 4     | `unified`    | 400    |    0.7391 |       0.812 |          0.539 |
| `segresnet_wide.yaml` §      | SegResNet (init_filters=32) | 82.7 M  | 2     | `unified`    | 1000   |    0.7482 |       0.782 |          0.577 |
| `dynunet.yaml`               | DynUNet (6-level)           | 31.2 M  | 2     | `unified`    | 400    | **0.7594**|       0.757 |          0.624 |

**§ Per-case Dice.** On voxel-level metrics (mIoU / Dice-micro / the
table above), `dynunet` is the best model on the test set. On
*per-case mean Dice* (each patient weighted equally, regardless of
nodule size), `segresnet_wide` wins at **0.5706** vs dynunet's
0.5560 and segresnet_small's 0.5240. Which model is "best" depends
on whether you want to reward large-nodule accuracy (micro) or overall
patient coverage (per-case).

### Joint end-to-end (single 3-class model, no bbox stage)

Same layout as the nodule table above — *Test Recall* and *Test
Precision* are for the **nodule** class specifically (the sparse
class), matching the two-stage numbers. Lung-class P/R sit at ~0.98
for all four and appear in the individual reports.

| Config                            | Architecture          | Params  | Batch | Split               | Test mIoU (3-cls) | Lung Dice | Nodule Dice | Nodule Recall | Nodule Precision |
|-----------------------------------|-----------------------|---------|-------|---------------------|------------------:|----------:|------------:|--------------:|-----------------:|
| `joint_segresnet_ex_lidc.yaml`    | SegResNet             | 20.7 M  | 4     | `unified_ex_lidc`|    **0.8240**     |   0.9768  |  **0.6856** |     0.819     |    **0.590**     |
| `joint_segresnet_pseudo.yaml`     | SegResNet             | 20.7 M  | 4     | `unified`        |      0.8102       |   0.9694  |    0.6647   |     0.836     |      0.552       |
| `joint_dynunet_ex_lidc.yaml`      | DynUNet (3D U-Net)    | 31.2 M  | 2     | `unified_ex_lidc`|      0.8016       |   0.9762  |    0.6261   |     0.828     |      0.503       |
| `joint_dynunet_pseudo.yaml`       | DynUNet (3D U-Net)    | 31.2 M  | 2     | `unified`        |      0.7886       |   0.9682  |    0.6027   |     0.785     |      0.493       |

The `ex_lidc` variants train only on NLST + NSCLC (they have GT lung
labels). The `pseudo` variants add LIDC-IDRI back in, using the trained
2D SegResNet ROI model as pseudo-GT for LIDC's missing lung labels.
Across both architectures, **`ex_lidc` outperforms `pseudo`** — the
pseudo-labels' noise slightly hurts the lung head's supervision signal
and the added LIDC diversity does not compensate. Full metric
breakdowns (Accuracy, per-class Precision/Recall, per-case Dice
distribution) are in [`reports/joint/`](reports/joint/).

All seven 3D configs use the same shared training recipe: Focal Tversky
(α=0.3, β=0.7, γ=2.0) + weighted CE (λ_ce=0.1, nodule class weight 100),
Adam (lr=1e-5, wd=1e-5), CosineAnnealingLR (T_max=epochs, η_min=1e-6),
bf16 AMP, seed 42. The nodule variants use a 2-class softmax head; the
joint variants use a 3-class softmax head and a K-class generalisation
of the same loss.

**‡ Split methodology.** The corpus is patient-grouped (a given patient
never appears in more than one of train/val/test) and
dataset-stratified (NLST, NSCLC, LIDC each get their own train/val/test
split, concatenated). *Val* was used during training for early stopping
and best-checkpoint selection, so those numbers would be optimistically
biased; the tables above and every report in [`reports/`](reports/)
therefore report on **test only** — a completely held-out split the
model never saw during training or model selection. All *Recall* /
*Precision* values are voxel-level, micro-averaged over the whole test
split. For a full metric breakdown per model (mIoU, Accuracy,
per-class breakdown, per-case Dice distribution) see the corresponding
report under `reports/`. Regenerate with `python eval_metrics_nodule.py
--config <cfg> --ckpt <ckpt> --split test` (or `eval_roi.py` /
`eval_metrics_joint.py` for the other tasks).

**3D SwinUNETR for the nodule task was tested (`feature_size=48`, 62 M
params, gradient checkpointing) but underperformed both the wider
SegResNet (`segresnet_wide`) and DynUNet (`dynunet`), while
being ~2.5× slower per epoch than either. Not retrained; not reported
in the paper. The *2D* SwinUNETR appears only in the ROI section below
and performs well.**

### Sliding-window variants (the paper's baselines)

Nine `*_sw.yaml` configs — the two 2D ROI models plus the same three two-stage nodule
architectures and four joint variants as above, but trained with
**random 128³ positive-biased patches** at native resolution and
evaluated with MONAI's `sliding_window_inference` instead of a single
256³ resize forward. **The dataset paper reports the sliding-window
ce10 models (and the SW ROI models) as its baselines**; the resize
models are the earlier recipe, kept for comparison. These runs keep the baseline
nodule CE weight of 100 (the `*_sw_ce10.yaml` variants below lower it
to 10).

The mode is a `training.mode: sliding_window` toggle in the YAML;
default `resize` reproduces the earlier resize recipe byte-for-byte. See any
`configs/*_sw.yaml` for the extra keys (`patch_size`,
`patches_per_volume`, `pos_neg_ratio` / `class_sample_ratios`,
`inference.sw_overlap`).

**Test-set summary** (all seven vs. their 256³ paper counterpart):

| Model | 256³ nodule Dice (resize recipe) | SW (ce=100) nodule Dice | Δ |
|---|---:|---:|---:|
| `segresnet_small_sw` | 0.6478 | 0.5192 | −0.129 |
| `segresnet_wide_sw`  | 0.6637 | 0.5387 | −0.125 |
| `dynunet_sw`         | 0.6838 | 0.4767 | −0.207 |
| `joint_segresnet_ex_lidc_sw` | 0.6856 | 0.4996 | −0.186 |
| `joint_segresnet_pseudo_sw`  | 0.6647 | 0.5008 | −0.164 |
| `joint_dynunet_ex_lidc_sw`   | 0.6261 | 0.3517 | −0.274 |
| `joint_dynunet_pseudo_sw`    | 0.6027 | 0.4021 | −0.201 |

Sliding-window at 128³ preserves small-nodule resolution but each patch
sees only ~15% of a lung volume, so the model can't use anatomy-
conditional context to rule out false positives. Every SW model comes
in below its resize counterpart on nodule Dice, with much higher recall
and much lower precision (aggressive over-prediction). Lung Dice on the
joint variants stays at ~0.98 regardless. Full per-model breakdowns in
[`reports/nodule_*_sw.md`](reports/) and
[`reports/joint/joint_*_sw.md`](reports/joint/).

#### Reduced-CE-weight retrain (`*_sw_ce10.yaml`)

To test whether the SW precision collapse was driven by the loss
weighting (nodule CE weight 100, tuned for the resize regime, applied
to positive-biased sparse patches), all seven SW models were retrained
identically but with the nodule CE weight lowered 100 → 10. Test-set
nodule Dice (micro), same split as above:

| Model | SW ce=100 | SW ce=10 | Δ | Precision ce=100 → ce=10 |
|---|---:|---:|---:|---|
| `segresnet_small_sw_ce10` | 0.5192 | 0.5370 | +0.018 | 0.377 → 0.465 |
| `segresnet_wide_sw_ce10`  | 0.5387 | 0.6064 | +0.068 | 0.414 → 0.493 |
| `dynunet_sw_ce10`         | 0.4767 | 0.4966 | +0.020 | 0.338 → 0.363 |
| `joint_segresnet_ex_lidc_sw_ce10` | 0.4996 | 0.4978 | −0.002 | 0.347 → 0.349 |
| `joint_segresnet_pseudo_sw_ce10`  | 0.5008 | 0.5024 | +0.002 | 0.352 → 0.355 |
| `joint_dynunet_ex_lidc_sw_ce10`   | 0.3517 | 0.4309 | +0.079 | 0.219 → 0.290 |
| `joint_dynunet_pseudo_sw_ce10`    | 0.4021 | 0.4403 | +0.038 | 0.263 → 0.300 |

Pre-trained weights of the ce10 variants are on HuggingFace:
[`HalmosiL/nodule-segresnet-3d-small-sw`](https://huggingface.co/HalmosiL/nodule-segresnet-3d-small-sw),
[`HalmosiL/nodule-segresnet-3d-wide-sw`](https://huggingface.co/HalmosiL/nodule-segresnet-3d-wide-sw),
[`HalmosiL/nodule-dynunet-3d-sw`](https://huggingface.co/HalmosiL/nodule-dynunet-3d-sw).

The lower weight helps consistently (best SW model is now
`segresnet_wide_sw_ce10` at 0.606) but closes only part of the gap to
the 256³ resize baselines — the loss weighting explains some, not all,
of the SW deficit. Per-model reports in
[`reports/*_sw_ce10.md`](reports/) and
[`reports/joint/*_sw_ce10.md`](reports/joint/).

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

## Data preparation (raw → `DATA_ROOT`)

The dataset itself is not bundled (≈0.9 TB). To rebuild it from the raw
sources, the pipeline is:

```
raw DICOM (ct/ + seg/ + manifest.csv)
  └─ scripts/dicom_to_npz.py        → ct_3d/ + nodule_sem_seg_3d/
       (HU clip [-1000, 400] → [0,1], 1 mm isotropic, channel-first npz)
3D volumes + lung masks / ROI checkpoint
  └─ scripts/make_roi_2d.py           → ct_2d/ + roi_sem_seg_2d/
       (axial slicing; masks from GT lung volumes with --lung_dir, or
        predicted per slice with --roi_config/--roi_ckpt for LIDC)
per-slice 2D ROI lung masks (roi_sem_seg_2d/)
  └─ scripts/precompute_bboxes_unified.py → per-series lung bboxes JSON
       (union of the 2D masks + 20 vox padding; model-free)
  └─ scripts/build_lung_3d.py       → lung_sem_seg_3d/ (GT where available,
       ROI-model pseudo-labels for LIDC; also emits the ex_lidc split)
series catalog (nodule_catalog.csv)
  └─ scripts/build_unified_split.py → patient-grouped, dataset-stratified
       70/15/15 split JSON (seed 42, deterministic)
```

The outputs of the last two steps are already bundled
(`data/splits/*.json`, `processed/bboxes_unified.json`), so these scripts
are only needed to regenerate the dataset from scratch or to extend it
with a new corpus.

## Bundled artifacts

- `data/splits/unified.json` — balanced patient-grouped train/val/test
  split (1683 / 297 / 325 series). Used by every bundled model on the
  full corpus (`segresnet_small`, `segresnet_wide`, `dynunet`,
  ROI, and the joint `pseudo` variants). This is the split the paper
  reports on.
- `data/splits/unified_ex_lidc.json` — same split with LIDC-IDRI
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

| Task        | Model                                    | HuggingFace repo |
|-------------|------------------------------------------|------------------|
| ROI (2D)    | SegResNet                                | [HalmosiL/roi-segresnet-2d](https://huggingface.co/HalmosiL/roi-segresnet-2d) |
| ROI (2D)    | SwinUNETR (small)                        | [HalmosiL/roi-swinunetr-2d](https://huggingface.co/HalmosiL/roi-swinunetr-2d) |
| Nodule (3D) | SegResNet small                | [HalmosiL/nodule-segresnet-3d-small](https://huggingface.co/HalmosiL/nodule-segresnet-3d-small) |
| Nodule (3D) | SegResNet **wide** (resize recipe) | [HalmosiL/nodule-segresnet-3d-wide](https://huggingface.co/HalmosiL/nodule-segresnet-3d-wide) |
| Nodule (3D) | DynUNet / 3D U-Net             | [HalmosiL/nodule-dynunet-3d](https://huggingface.co/HalmosiL/nodule-dynunet-3d) |
| Joint (3D)  | SegResNet, ex-LIDC                       | [HalmosiL/joint-segresnet-3d-ex-lidc](https://huggingface.co/HalmosiL/joint-segresnet-3d-ex-lidc) |
| Joint (3D)  | SegResNet, pseudo-LIDC                   | [HalmosiL/joint-segresnet-3d-pseudo-lidc](https://huggingface.co/HalmosiL/joint-segresnet-3d-pseudo-lidc) |
| Joint (3D)  | DynUNet, ex-LIDC (best joint)            | [HalmosiL/joint-dynunet-3d-ex-lidc](https://huggingface.co/HalmosiL/joint-dynunet-3d-ex-lidc) |
| Joint (3D)  | DynUNet, pseudo-LIDC                     | [HalmosiL/joint-dynunet-3d-pseudo-lidc](https://huggingface.co/HalmosiL/joint-dynunet-3d-pseudo-lidc) |

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

Both are trained on the same `unified` split as the nodule models and use the
same 2D dataset (`ct_2d/` + `roi_sem_seg_2d/`). See
[`reports/roi_segresnet.md`](reports/roi_segresnet.md) and
[`reports/roi_swinunetr.md`](reports/roi_swinunetr.md) for full metric
breakdowns.

**You do not need to train or run either ROI model to reproduce the
two-stage nodule configs.** They read the precomputed
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

### Sliding-window ROI variants (`roi_sw.yaml`, `roi_swin_sw.yaml`)

Native-resolution recipe: 256×256 patch training + 2D sliding-window
inference (overlap 0.5, Gaussian blending) instead of resizing each
slice. Held-out test split (45,751 slices):

| Model | Dice (micro) | Precision | Recall | Per-slice mean | Per-slice p05 |
|---|---:|---:|---:|---:|---:|
| `roi_sw` (SegResNet) | 0.9823 | 0.9832 | 0.9814 | 0.9511 | 0.8352 |
| `roi_swin_sw` (SwinUNETR) | **0.9834** | **0.9836** | **0.9832** | **0.9612** | **0.8692** |

Both models avoid the resize-recipe SegResNet's per-slice p05 = 0.0
failure on near-empty apex/base slices. Pre-trained weights:
[`HalmosiL/roi-segresnet-2d-sw`](https://huggingface.co/HalmosiL/roi-segresnet-2d-sw),
[`HalmosiL/roi-swinunetr-2d-sw`](https://huggingface.co/HalmosiL/roi-swinunetr-2d-sw).
All models of the project are collected at
[huggingface.co/collections/HalmosiL/medical-image-segmentation](https://huggingface.co/collections/HalmosiL/medical-image-segmentation-6a71c92de761a2fc4ce68162). Full reports:
[`reports/roi_segresnet_sw.md`](reports/roi_segresnet_sw.md),
[`reports/roi_swinunetr_sw.md`](reports/roi_swinunetr_sw.md); metric
JSONs under `results_eval/`.

### Downloading a pre-trained ROI checkpoint

Trained ROI checkpoints are hosted on HuggingFace (private during paper
review — request access if you need them):

- SegResNet: [HalmosiL/roi-segresnet-2d](https://huggingface.co/HalmosiL/roi-segresnet-2d)
- SwinUNETR: [HalmosiL/roi-swinunetr-2d](https://huggingface.co/HalmosiL/roi-swinunetr-2d)

Each repo ships `model.pth` (weights-only). To use with `--resume` in
this repo, download and rename:

```bash
huggingface-cli download HalmosiL/roi-segresnet-2d model.pth --local-dir checkpoints/roi
mv checkpoints/roi/model.pth checkpoints/roi/best.pth

huggingface-cli download HalmosiL/roi-swinunetr-2d model.pth --local-dir checkpoints/roi_swin
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
huggingface-cli download HalmosiL/roi-segresnet-2d --local-dir ./ckpts/roi
export DATA_ROOT=/path/to/unified

python scripts/build_lung_3d.py \
    --data-root         $DATA_ROOT \
    --out-dir           $DATA_ROOT/lung_sem_seg_3d \
    --roi-config        ./ckpts/roi/config.yaml \
    --roi-checkpoint    ./ckpts/roi/model.pth \
    --split-in          data/splits/unified.json \
    --split-out-ex-lidc data/splits/unified_ex_lidc.json \
    --manifest-out      lung_source_manifest.json
```

This takes ~1 h on an H100 (mostly LIDC inference). The bundled
`data/splits/unified_ex_lidc.json` is the exact output of this
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
api/                                FastAPI HTTP wrapper (POST /predict → nodule mask)
eval_roi.py                         ROI evaluation: mIoU / Accuracy / Precision / Recall
eval_metrics_nodule.py              two-stage nodule evaluation, same metric set
eval_metrics_joint.py               joint (3-class) evaluation, same metric set
model.py                            architecture dispatch (SegResNet / DynUNet / SwinUNETR)
loss.py                             FocalTverskyCELoss (2-cls) + MulticlassFocalTverskyCELoss (K-cls) + DiceLoss
dataset.py                          NoduleFineCropDataset + Roi2DDataset + JointFullVolumeDataset
transforms.py                       train / val transform pipelines
configs/segresnet_small.yaml     two-stage nodule: small SegResNet
configs/segresnet_wide.yaml      two-stage nodule: wide SegResNet (resize recipe)
configs/dynunet.yaml             two-stage nodule: DynUNet (3D U-Net)
configs/{roi,roi_swin}.yaml         2D ROI configs
configs/joint_*.yaml                joint (end-to-end) configs (4)
scripts/build_lung_3d.py            preprocessing for joint training (writes lung_sem_seg_3d/)
scripts/dicom_to_npz.py             raw DICOM → normalized npz volumes
scripts/build_unified_split.py      catalog → patient-grouped split JSON
scripts/precompute_bboxes_unified.py 2D ROI masks → per-series lung bbox JSON
scripts/make_roi_2d.py              3D volumes → 2D ROI training slices (GT or predicted)
scripts/instance_metrics.py         per-nodule instance recall/precision (26-connectivity)
scripts/gen_report.py               regenerates model reports (resize + SW/ce10) from metrics JSONs
scripts/md_to_pdf.py                renders report md → pdf (needs markdown + weasyprint)
container/nodule-seg.def            Apptainer recipe (+ pinned requirements_train.txt)
slurm/{train,eval}.sh               generic SLURM launchers (site-specific headers)
data/splits/unified.json         balanced split (used by everything except joint ex_lidc)
data/splits/unified_ex_lidc.json same split with LIDC filtered out (used by joint ex_lidc)
processed/bboxes_unified.json       bundled per-series lung bboxes
reports/                            markdown + PDF metric reports (one per model)
requirements.txt
```

## Cluster training (SLURM + Apptainer)

`container/nodule-seg.def` builds the training container (NGC PyTorch
24.05 base + pinned deps from `container/requirements_train.txt`):

```bash
cd container && apptainer build --fakeroot nodule-seg.sif nodule-seg.def
```

`slurm/train.sh` and `slurm/eval.sh` are generic launchers around
`train.py` / `eval_metrics_*.py`:

```bash
export DATA_ROOT=/path/to/dataset
export SIF=/path/to/nodule-seg.sif
sbatch --job-name=segresnet_small slurm/train.sh configs/segresnet_small.yaml
SPLIT=test sbatch slurm/eval.sh nodule configs/segresnet_small.yaml checkpoints/segresnet_small/best_model.pth
```

The `#SBATCH` headers carry SZTE-supercomputer-specific values
(`--account`, `--partition`, and the explicit per-GPU bundle of
32 CPUs / 182 GB — it is not applied automatically there); adjust them
for other sites.

## Compute expectations

Wall-clock on a single H100 94 GB:

| Config                            | Epoch time | Full run |
|-----------------------------------|-----------:|---------:|
| `segresnet_small` (400)        |  ~13 min   |  ~3.5 d  |
| `segresnet_wide`  (1000)       |  ~18 min   | ~12 d    |
| `dynunet`         (400)        |  ~13 min   |  ~3.5 d  |
| `roi`                (100)        |   ~5 min   |  ~8 h    |
| `roi_swin`           (100)        |  ~18 min   |  ~30 h   |
| `joint_segresnet_ex_lidc` (400)   |  ~13 min   |  ~3.5 d  |
| `joint_segresnet_pseudo`  (400)   |  ~19 min   |  ~5 d    |
| `joint_dynunet_ex_lidc`   (400)   |  ~13 min   |  ~3.5 d  |
| `joint_dynunet_pseudo`    (400)   |  ~19 min   |  ~5 d    |

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
