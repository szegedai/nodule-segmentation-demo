"""Generate MD metric reports for the nodule and joint models.

Covers both recipes — the config's `training.mode` selects the
template branch: `resize` (paper baseline; adds the HuggingFace
"How to load & run" section for the published models) and
`sliding_window` (SW / SW-ce10 variants). The bundled
reports/*_sw*.md files were produced by this script and regenerate
byte-identically; the three legacy resize baseline reports predate it
(hand-maintained), but any re-evaluated resize model can be reported
with it. Render the PDF with scripts/md_to_pdf.py.

Usage:
    gen_sw_reports.py <metrics.json> <config.yaml> <out.md>

Handles both binary nodule reports (stage2_fine/eval_metrics.py output) and
3-class joint reports (stage3_joint/eval_metrics.py output). Detects task
from the number of classes in the config.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml


HF_REPOS = {
    "segresnet_small": "szabopeter/nodule-segresnet-3d-small",
    "segresnet_wide":  "szabopeter/nodule-segresnet-3d-wide",
    "dynunet":         "szabopeter/nodule-dynunet-3d",
    "joint_segresnet_ex_lidc": "szabopeter/joint-segresnet-3d-ex-lidc",
    "joint_segresnet_pseudo":  "szabopeter/joint-segresnet-3d-pseudo-lidc",
    "joint_dynunet_ex_lidc":   "szabopeter/joint-dynunet-3d-ex-lidc",
    "joint_dynunet_pseudo":    "szabopeter/joint-dynunet-3d-pseudo-lidc",
}


def hf_section(stem):
    repo = HF_REPOS.get(stem)
    if not repo:
        return ""
    return f"""## How to load & run

Pre-trained weights are hosted at
[`{repo}`](https://huggingface.co/{repo}):

```bash
huggingface-cli download {repo} --local-dir ./ckpt
```

See the repo README ("Inference on a new CT") for running the full
pipeline with `inference.py`.

"""


def load(cfg_path, m_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    m = json.load(open(m_path))
    return cfg, m


def nodule_report(cfg_path, m_path):
    cfg, m = load(cfg_path, m_path)
    arch = cfg["model"]["name"]
    init_f = cfg["model"].get("init_filters", "n/a")
    is_sw  = cfg.get("training", {}).get("mode", "resize") == "sliding_window"
    patch = tuple(cfg["preprocessing"].get("patch_size", [128, 128, 128]))
    tsize = tuple(cfg["preprocessing"].get("target_size", [256, 256, 256]))
    epochs = cfg["training"]["epochs"]
    batch  = cfg["training"]["batch_size"]
    ppv    = cfg["training"].get("patches_per_volume", 1)
    ratio  = cfg["training"].get("pos_neg_ratio", 2)
    ovl    = cfg.get("inference", {}).get("sw_overlap", 0.5)
    swbs   = cfg.get("inference", {}).get("sw_batch", 4)
    split  = Path(cfg["data"]["split_json"]).stem
    ckpt_name = Path(m["_meta"]["checkpoint"]).name
    ep     = m["_meta"].get("epoch", "?")
    n_params = m["_meta"].get("params", "?")

    d = m["Dice_per_case"]
    variant = "small" if init_f == 16 else ("wide" if init_f == 32 else str(init_f))
    loss_cfg = cfg.get("loss", {})
    nod_w = int(float(loss_cfg.get("ce_nodule_weight", 100)))
    is_ce10 = Path(cfg_path).stem.endswith("_ce10")
    title_suffix = ""
    base_stem = Path(cfg_path).stem.replace("_ce10", "").replace("_sw", "_v2")
    diff_txt = ("the differences are a reduced nodule CE weight (10 instead of 100) and the inference recipe:"
                if is_ce10 else "the only difference is inference recipe:")
    stem = Path(cfg_path).stem
    hf_sec = "" if is_sw else hf_section(stem)
    if is_sw:
        title_recipe = f" — sliding-window (128³), CE weight {nod_w}"
        task_intro = ("lung-bbox-cropped. Same task as the paper's `" + base_stem +
            "`\nmodel — " + diff_txt + " **sliding-window over 128³\n"
            "patches at native resolution** rather than a single forward on a 256³\nresample.")
        input_bullets = (
            "- **Input**   `(1, H, W, D)` CT crop, intensity-normalised to `[0, 1]`,\n"
            "              lung-bbox + 20-voxel padding, kept at native voxel grid.\n"
            "- **Training input** 128³ patches sampled via `RandCropByPosNegLabeld` at\n"
            f"              a ratio of `pos:neg = {ratio}:1`, `num_samples = {ppv}` per volume.\n"
            f"- **Inference** MONAI `sliding_window_inference`, ROI `{patch}`,\n"
            f"              `overlap = {ovl}`, `sw_batch_size = {swbs}`, Gaussian\n"
            "              blending, single output at the input resolution.")
        crop_bullet = "medium-model fallback), + 20-vox padding. No resampling."
        batch_row = (f"| Batch size         | {batch} volumes × {ppv} patches = {batch*ppv} patches / step |\n"
                     f"| Patch size         | {patch[0]}³ (positive-biased random crop, `pos:neg = {ratio}:1`) |")
        pred_src = "`sliding_window_inference`"
        interp = ("Sliding-window inference at 128³ trades global context (a 256³ resample\n"
            "sees the whole lung in one pass) for preserved resolution (no down-\n"
            "sampling of small nodules). On this unified corpus the trade tends to\n"
            "be neutral-to-negative on nodule Dice — small-nodule gains do not\n"
            "compensate for lost global context on typical test cases.")
    else:
        title_recipe = f" — resize ({tsize[0]}³)"
        task_intro = (f"lung-bbox-cropped, on the paper's baseline recipe: the bbox crop is\n"
            f"trilinearly resampled to {tsize[0]}³ and segmented in a single forward\npass.")
        input_bullets = (
            "- **Input**   `(1, H, W, D)` CT crop, intensity-normalised to `[0, 1]`,\n"
            "              lung-bbox + 20-voxel padding.\n"
            "- **Training / inference input** the crop trilinearly resampled to\n"
            f"              `{tsize}`; one forward pass per volume.")
        crop_bullet = f"medium-model fallback), + 20-vox padding, resampled to {tsize[0]}³."
        batch_row = (f"| Batch size         | {batch} volumes |\n"
                     f"| Input size         | {tsize[0]}³ (trilinear resample of the bbox crop) |")
        pred_src = f"a single {tsize[0]}³ forward"
        interp = (f"The {tsize[0]}³ resample gives the model the whole lung in one pass\n"
            "(global anatomical context) at the cost of downsampling resolution for\n"
            "small nodules in large crops — the trade the sliding-window variants\n"
            "try to reverse, so far unsuccessfully (see the `*_sw*` reports).")
    title_arch = "DynUNet / 3D U-Net" if arch == "dynunet" else f"SegResNet ({variant})"

    return f"""# Lung Nodule Segmentation — {title_arch}{title_recipe}{title_suffix}

## 1. Task

Voxel-level segmentation of pulmonary nodules in 3D chest CT volumes,
{task_intro}

{input_bullets}
- **Output**  `(2, H, W, D)` softmax logits: 0 = background, 1 = nodule.

## 2. Model

| Property                | Value |
|-------------------------|-------|
| Architecture family     | {title_arch} |
| Spatial dims            | 3 |
| Input channels          | 1 |
| Output channels         | 2 (softmax) |
| {"Init. feature width   " if arch == "segresnet" else "6-level auto-configured  "}| {init_f if arch == "segresnet" else "strides=[1,2,2,2,2], kernel=3"} |
| Trainable parameters    | **{n_params:,}** |

## 3. Data

Same unified corpus + split as the paper's 256³ baselines:

| Source          | Role |
|-----------------|------|
| NLST            | train + val + test |
| NSCLC-Radiomics | train + val + test |
| LIDC-IDRI       | train + val + test |

- **Split**: `{split}.json` — patient-grouped, dataset-stratified.
  1 683 train / 297 val / 325 test series.
- **Lung crop**: per-series 3D bbox from `bboxes_unified.json` (lung ROI +
  {crop_bullet}
- **Class imbalance**: nodule voxels are ~10⁻⁵ of the total — see §7.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Focal Tversky + weighted CE (α=0.3, β=0.7, γ=2.0, λ_CE=0.1 / 0.3, nodule class weight = {nod_w}) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻⁵ |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = {epochs}, η_min = 1 × 10⁻⁶ |
{batch_row}
| Epochs             | {epochs} |
| Random seed        | 42 |
| Mixed precision    | bf16 (autocast, no GradScaler) |
| Augmentation       | 3D flips, 90° rotations, elastic rotation, zoom, intensity scale/shift, Gaussian noise/blur, contrast |
| Hardware           | 1 × NVIDIA H100 80 GB (fat01) |

Best checkpoint recorded at **epoch {ep} / {epochs}** (see `_meta` in
metrics JSON).

## 5. Evaluation protocol

Evaluated on the `{split}` **test** split ({m['n_cases']} series held
out — never seen during training or model selection). Predictions taken
as `argmax` over the 2-channel softmax output from
{pred_src}. All voxel-level metrics are micro-averaged
over the whole split.

## 6. Results

### Ticket-required metrics

| Metric             | Value      |
|--------------------|-----------:|
| **mean IoU**       | **{m['mIoU']:.4f}** |
| **Accuracy**       | **{m['Accuracy']:.4f}** |
| **Precision**      | **{m['Precision']:.4f}** |
| **Recall**         | **{m['Recall']:.4f}** |

### Supplementary

| Metric                          | Value      |
|---------------------------------|-----------:|
| IoU (foreground / nodule class) | {m['IoU_fg']:.4f} |
| IoU (background class)          | {m['IoU_bg']:.4f} |
| Dice / F1 (micro)               | {m['Dice_micro']:.4f} |

### Per-case Dice distribution ({m['n_cases']} test cases)

| Statistic | Value  |
|-----------|-------:|
| Mean      | {d['mean']:.4f} |
| Std       | {d['std']:.4f} |
| Min       | {d['min']:.4f} |
| p05       | {d['p05']:.4f} |
| p25       | {d['p25']:.4f} |
| Median    | {d['median']:.4f} |
| p75       | {d['p75']:.4f} |
| p95       | {d['p95']:.4f} |
| Max       | {d['max']:.4f} |

Voxels evaluated: {m['n_voxels']:,}. Cases: {m['n_cases']}.

## 7. Notes on interpretation

{interp}

Class-imbalance caveats are unchanged from the 256³ reports: Accuracy
is trivially ~1, mIoU is dominated by IoU_bg, and Dice / F1 (micro) is
the honest voxel-level summary. Per-case mean weights each patient
equally regardless of nodule volume.

{hf_sec}## 8. Reproducibility

- Config       `{cfg_path}`
- Checkpoint   `{m['_meta']['checkpoint']}`
- Metrics JSON `{m_path}`
- Command      `python stage2_fine/eval_metrics.py --config {cfg_path} --checkpoint {m['_meta']['checkpoint']} --split test`
"""


def joint_report(cfg_path, m_path):
    cfg, m = load(cfg_path, m_path)
    arch = cfg["model"]["name"]
    init_f = cfg["model"].get("init_filters", "n/a")
    is_sw  = cfg.get("training", {}).get("mode", "resize") == "sliding_window"
    patch = tuple(cfg["preprocessing"].get("patch_size", [128, 128, 128]))
    tsize = tuple(cfg["preprocessing"].get("target_size", [256, 256, 256]))
    epochs = cfg["training"]["epochs"]
    batch  = cfg["training"]["batch_size"]
    ppv    = cfg["training"].get("patches_per_volume", 1)
    ratios = cfg["training"].get("class_sample_ratios", [0.2, 0.3, 0.5])
    ovl    = cfg.get("inference", {}).get("sw_overlap", 0.5)
    swbs   = cfg.get("inference", {}).get("sw_batch", 4)
    split  = Path(cfg["data"]["split_json"]).stem
    ep     = m["_meta"].get("epoch", "?")
    n_params = m["_meta"].get("params", "?")

    title_arch = "DynUNet / 3D U-Net" if arch == "dynunet" else "SegResNet"
    variant = "ex-LIDC (NLST + NSCLC only)" if "ex_lidc" in split else "pseudo-LIDC (full corpus w/ pseudo lung labels)"
    loss_cfg = cfg.get("loss", {})
    nod_w = int(float((loss_cfg.get("class_weights") or [1, 1, 100])[-1]))
    is_ce10 = Path(cfg_path).stem.endswith("_ce10")
    title_suffix = ""
    diff_txt = ("the differences are a reduced nodule CE\nweight (10 instead of 100) and the inference recipe:"
                if is_ce10 else "the only difference is\ninference recipe:")
    stem = Path(cfg_path).stem
    hf_sec = "" if is_sw else hf_section(stem)
    base_stem = stem.replace('_ce10','').replace('_sw','')
    if is_sw:
        title_recipe = f" — sliding-window (128³), CE weight {nod_w}"
        task_intro = ("Same task as the paper's\n`" + base_stem + "` — " + diff_txt +
            " **sliding-window over 128³ patches at native\n"
            "resolution** rather than a single forward on a 256³ resample.")
        input_bullets = (
            "- **Input**   `(1, H, W, D)` CT at native voxel grid.\n"
            "- **Training input** 128³ patches sampled via `RandCropByLabelClassesd`\n"
            f"              with class-balanced ratios `[bg, lung, nodule] = {ratios}`,\n"
            f"              `num_samples = {ppv}` per volume.\n"
            f"- **Inference** MONAI `sliding_window_inference`, ROI `{patch}`,\n"
            f"              `overlap = {ovl}`, `sw_batch_size = {swbs}`, Gaussian\n"
            "              blending, output at input resolution.")
        batch_row = (f"| Batch size         | {batch} volumes × {ppv} patches = {batch*ppv} patches / step |\n"
                     f"| Patch size         | {patch[0]}³ (class-biased random crop, ratios `[bg, lung, nodule] = {ratios}`) |")
        interp = ("Sliding-window inference at 128³ trades global lung context for\n"
            "preserved local resolution. On this corpus the trade tends to be\n"
            "neutral-to-negative on nodule Dice compared to the 256³ resize\n"
            "baseline — patches see only ~15% of a typical lung volume, which\n"
            "limits anatomy-conditional reasoning.")
    else:
        title_recipe = f" — resize ({tsize[0]}³)"
        task_intro = (f"The paper's baseline recipe: the full CT is trilinearly\n"
            f"resampled to {tsize[0]}³ (no ROI stage, no bbox crop) and segmented\nin a single forward pass.")
        input_bullets = (
            f"- **Input / inference input**   `(1, H, W, D)` CT trilinearly resampled to `{tsize}`.\n"
            "- **Output**  argmax over 3-class softmax, resampled back to the native grid.")
        batch_row = (f"| Batch size         | {batch} volumes |\n"
                     f"| Input size         | {tsize[0]}³ (trilinear resample of the full CT) |")
        interp = (f"The {tsize[0]}³ full-volume resample gives the model global lung\n"
            "context in one pass at the cost of small-nodule resolution — the\n"
            "trade the sliding-window variants try to reverse, so far\n"
            "unsuccessfully (see the `*_sw*` reports).")

    # Per-class metrics for the joint task
    per_class = m["per_class"]
    bg   = per_class.get("background", {})
    lung = per_class.get("lung",       {})
    nod  = per_class.get("nodule",     {})

    return f"""# Joint 3-class Lung Segmentation — {title_arch}, {variant}{title_recipe}{title_suffix}

## 1. Task

End-to-end 3-class semantic segmentation of a full chest CT into
{{background, lung, nodule}}. {task_intro}

{input_bullets}
- **Output**  `(3, H, W, D)` softmax logits: 0 = bg, 1 = lung, 2 = nodule.

## 2. Model

| Property                | Value |
|-------------------------|-------|
| Architecture family     | {title_arch} |
| Spatial dims            | 3 |
| Input channels          | 1 |
| Output channels         | 3 (softmax) |
| {"Init. feature width   " if arch == "segresnet" else "6-level auto-configured  "}| {init_f if arch == "segresnet" else "strides=[1,2,2,2,2], kernel=3"} |
| Trainable parameters    | **{n_params:,}** |

## 3. Data

| Source          | Role |
|-----------------|------|
| NLST            | train + val + test |
| NSCLC-Radiomics | train + val + test |
{"| LIDC-IDRI       | train + val + test (lung labels are pseudo — from 2D ROI SegResNet) |" if "pseudo" in split else "| LIDC-IDRI       | excluded (no GT lung labels available) |"}

- **Split**: `{split}.json` — patient-grouped, dataset-stratified.
- **Preprocessing**: none beyond intensity clipping / normalization; the
  full CT enters the sliding-window inference at its native voxel grid.

## 4. Training

| Setting            | Value |
|--------------------|-------|
| Loss               | Multiclass Focal Tversky + weighted CE (α=0.3, β=0.7, γ=2.0, λ_CE=0.1, class weights [1, 1, {nod_w}]) |
| Optimizer          | Adam |
| Learning rate      | 1 × 10⁻⁵ |
| Weight decay       | 1 × 10⁻⁵ |
| LR schedule        | Cosine annealing, T_max = {epochs}, η_min = 1 × 10⁻⁶ |
{batch_row}
| Epochs             | {epochs} |
| Random seed        | 42 |
| Mixed precision    | bf16 |
| Augmentation       | 3D flips, 90° rotations, elastic rotation, zoom, intensity scale/shift, Gaussian noise/blur, contrast |
| Hardware           | 1 × NVIDIA H100 80 GB (fat01) |

Best checkpoint recorded at **epoch {ep} / {epochs}**.

## 5. Evaluation protocol

Evaluated on the `{split}` **test** split ({m['n_cases']} series). All
voxel-level metrics micro-averaged over the whole split. Per-class
metrics computed against argmax predictions.

## 6. Results

### Ticket-required metrics (3-class average)

| Metric                     | Value      |
|----------------------------|-----------:|
| **mean IoU (3-cls)**       | **{m['mIoU']:.4f}** |
| **mean IoU (lung + nodule)** | **{m['mIoU_no_bg']:.4f}** |
| **Accuracy**               | **{m['Accuracy']:.6f}** |
| **Macro Precision (lung + nodule)** | **{m['Precision_no_bg']:.4f}** |
| **Macro Recall (lung + nodule)**    | **{m['Recall_no_bg']:.4f}** |

### Per-class breakdown

| Class      | Dice (micro) | Recall | Precision | IoU    |
|------------|-------------:|-------:|----------:|-------:|
| background | {bg.get('Dice_micro', 0):.4f} | {bg.get('Recall', 0):.4f} | {bg.get('Precision', 0):.4f} | {bg.get('IoU', 0):.4f} |
| lung       | {lung.get('Dice_micro', 0):.4f} | {lung.get('Recall', 0):.4f} | {lung.get('Precision', 0):.4f} | {lung.get('IoU', 0):.4f} |
| nodule     | {nod.get('Dice_micro', 0):.4f} | {nod.get('Recall', 0):.4f} | {nod.get('Precision', 0):.4f} | {nod.get('IoU', 0):.4f} |

### Per-case Dice distribution — nodule class ({m['n_cases']} test cases)

| Statistic | Value  |
|-----------|-------:|
| Mean      | {nod.get('Dice_per_case', {}).get('mean', 0):.4f} |
| Std       | {nod.get('Dice_per_case', {}).get('std', 0):.4f} |
| Min       | {nod.get('Dice_per_case', {}).get('min', 0):.4f} |
| p25       | {nod.get('Dice_per_case', {}).get('p25', 0):.4f} |
| Median    | {nod.get('Dice_per_case', {}).get('median', 0):.4f} |
| p75       | {nod.get('Dice_per_case', {}).get('p75', 0):.4f} |
| Max       | {nod.get('Dice_per_case', {}).get('max', 0):.4f} |

Voxels evaluated: {m['n_voxels']:,}. Cases: {m['n_cases']}.

## 7. Notes on interpretation

Lung Dice at ≈ 0.98 is essentially saturated across all architectures
and inference recipes — the lung is a large, well-defined foreground
class. Nodule Dice is the real signal.

{interp}

{hf_sec}## 8. Reproducibility

- Config       `{cfg_path}`
- Checkpoint   `{m['_meta']['checkpoint']}`
- Metrics JSON `{m_path}`
- Command      `python stage3_joint/eval_metrics.py --config {cfg_path} --checkpoint {m['_meta']['checkpoint']} --split test`
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("config")
    ap.add_argument("out")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    n_classes = cfg["model"].get("out_channels", 2)
    if n_classes == 3:
        md = joint_report(args.config, args.metrics)
    else:
        md = nodule_report(args.config, args.metrics)
    Path(args.out).write_text(md)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
