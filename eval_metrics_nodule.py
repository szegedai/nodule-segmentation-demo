"""Evaluate a two-stage nodule model on its val split.

Computes voxel-level mIoU / Accuracy / Precision / Recall + Dice
(micro) and a per-case Dice histogram — the same metric set as
`eval_roi.py` and `eval_metrics_joint.py`, adapted to the 2-class
{background, nodule} softmax head.

Because nodule voxels are ~10⁻⁵ of the total, voxel-level Accuracy
looks trivially perfect and mIoU is dominated by `IoU_background` — the
useful signal is in `IoU_foreground`, `Dice_micro`, `Precision`,
`Recall`, and the per-case Dice distribution. All are reported.

Usage:
    export DATA_ROOT=/path/to/unified
    python eval_metrics_nodule.py \\
        --config configs/v6.yaml \\
        --ckpt   checkpoints/v6/best_model.pth
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset    import build_eval_dataset
from model      import build_model, count_parameters
from transforms import build_transforms


def build_predictor(model, cfg):
    """Return a callable `predictor(x) -> logits` matching cfg's mode."""
    training = cfg.get("training", {})
    if training.get("mode", "resize") != "sliding_window":
        return model
    preproc  = cfg.get("preprocessing", {})
    inf_cfg  = cfg.get("inference",     {})
    roi_size = tuple(preproc.get("patch_size", [128, 128, 128]))
    ovlp     = float(inf_cfg.get("sw_overlap", 0.5))
    swbs     = int(inf_cfg.get("sw_batch",   4))
    def _predict(x):
        return sliding_window_inference(
            inputs=x, roi_size=roi_size, sw_batch_size=swbs,
            predictor=model, overlap=ovlp, mode="gaussian",
        )
    return _predict


def load_config(path):
    if not os.environ.get("DATA_ROOT"):
        raise SystemExit("DATA_ROOT is not set.  Run:  export DATA_ROOT=/path/to/unified")
    with open(path) as f:
        text = os.path.expandvars(f.read())
    return yaml.safe_load(text)


def load_weights(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ("model_state_dict" in ck or "model" in ck):
        state = ck.get("model_state_dict", ck.get("model"))
    else:
        state = ck
    model.load_state_dict(state)
    epoch = ck.get("epoch", -1) if isinstance(ck, dict) else -1
    return epoch


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, predictor=None):
    model.eval()
    forward = predictor if predictor is not None else model
    tp = fp = fn = tn = 0
    per_case_dice = []
    per_case_files = []
    t0 = time.time()
    n_batches = len(loader)

    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            logits = forward(img)

        pred = logits.float().argmax(dim=1, keepdim=True).to(torch.uint8)  # (B,1,H,W,D)
        gt   = (lbl > 0.5).to(torch.uint8)

        p1 = pred.eq(1)
        g1 = gt.eq(1)
        tp += int((p1 &  g1).sum())
        fp += int((p1 & ~g1).sum())
        fn += int((~p1 &  g1).sum())
        tn += int((~p1 & ~g1).sum())

        # per-case Dice
        p_sum = pred.sum(dim=(1, 2, 3, 4)).float()
        g_sum = gt.sum(dim=(1, 2, 3, 4)).float()
        inter = (pred & gt).sum(dim=(1, 2, 3, 4)).float()
        denom = p_sum + g_sum
        dice_case = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
        per_case_dice.extend(dice_case.cpu().tolist())

        per_case_files.extend(batch.get("filename", [""] * pred.shape[0]))

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            dt = time.time() - t0
            print(f"  case {i+1}/{n_batches}  ({dt:.1f} s elapsed)", flush=True)

    total  = tp + fp + fn + tn
    iou_fg = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou   = 0.5 * (iou_fg + iou_bg)
    acc    = (tp + tn) / max(total, 1)
    prec   = tp / max(tp + fp, 1)
    rec    = tp / max(tp + fn, 1)
    dice_micro = 2 * tp / max(2 * tp + fp + fn, 1)

    d = np.array(per_case_dice)
    return {
        "n_cases":  len(per_case_dice),
        "n_voxels": total,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mIoU":       miou,
        "IoU_fg":     iou_fg,
        "IoU_bg":     iou_bg,
        "Accuracy":   acc,
        "Precision":  prec,
        "Recall":     rec,
        "Dice_micro": dice_micro,
        "Dice_per_case": {
            "mean":   float(d.mean()) if len(d) else 0.0,
            "std":    float(d.std())  if len(d) else 0.0,
            "min":    float(d.min())  if len(d) else 0.0,
            "median": float(np.median(d)) if len(d) else 0.0,
            "max":    float(d.max())  if len(d) else 0.0,
            "p05":    float(np.percentile(d,  5)) if len(d) else 0.0,
            "p25":    float(np.percentile(d, 25)) if len(d) else 0.0,
            "p75":    float(np.percentile(d, 75)) if len(d) else 0.0,
            "p95":    float(np.percentile(d, 95)) if len(d) else 0.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--split",  default="val", choices=["train", "val", "test"],
                    help="Which split to evaluate on (default: val).")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg  = load_config(args.config)
    task = cfg.get("task", "nodule")
    if task != "nodule":
        sys.exit(f"eval_metrics_nodule.py only supports task=nodule, got {task!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_tf, val_tf = build_transforms(task, cfg)
    eval_ds = build_eval_dataset(task, cfg, val_tf, split_name=args.split)
    print(f"{args.split} cases: {len(eval_ds):,}", flush=True)

    val_loader = DataLoader(eval_ds, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(cfg["model"]).to(device).eval()
    print(f"model: {cfg['model'].get('name', 'segresnet')}  "
          f"params={count_parameters(model):,}", flush=True)
    ep = load_weights(model, args.ckpt)
    print(f"loaded ckpt {args.ckpt} (epoch={ep})", flush=True)

    amp_dtype = torch.bfloat16 if cfg.get("training", {}).get("amp", True) else None
    predictor = build_predictor(model, cfg)
    mode = cfg.get("training", {}).get("mode", "resize")
    print(f"inference mode: {mode}", flush=True)

    print("evaluating…", flush=True)
    m = evaluate(model, val_loader, device, amp_dtype, predictor=predictor)

    m["_meta"] = {
        "config":       args.config,
        "checkpoint":   args.ckpt,
        "eval_split":   args.split,
        "epoch":        ep,
        "model_name":   cfg["model"].get("name", "segresnet"),
        "params":       count_parameters(model),
        "split_json":   cfg["data"].get("split_json"),
    }

    print("\n" + "=" * 60)
    print(f"  Nodule eval — {cfg['model'].get('name', 'segresnet')}   split={args.split}")
    print(f"  checkpoint: {args.ckpt}")
    print("=" * 60)
    print(f"  val cases evaluated: {m['n_cases']:,}")
    print(f"  voxels evaluated:    {m['n_voxels']:,}")
    print(f"  ────  Ticket-required metrics  ────")
    print(f"  mIoU:                {m['mIoU']:.4f}")
    print(f"    IoU foreground:    {m['IoU_fg']:.4f}    ← nodule (informative)")
    print(f"    IoU background:    {m['IoU_bg']:.4f}    ← ~1.0 (class imbalance)")
    print(f"  Accuracy:            {m['Accuracy']:.6f}    ← ~1.0 (class imbalance)")
    print(f"  Precision:           {m['Precision']:.4f}")
    print(f"  Recall:              {m['Recall']:.4f}")
    print(f"  ────  supplementary  ────")
    print(f"  Dice (micro):        {m['Dice_micro']:.4f}    ← = F1 = 2·IoU_fg / (1 + IoU_fg)")
    d = m["Dice_per_case"]
    print(f"  Dice per case:       mean={d['mean']:.4f}  std={d['std']:.4f}")
    print(f"                       min={d['min']:.4f}  p05={d['p05']:.4f}  p25={d['p25']:.4f}")
    print(f"                       median={d['median']:.4f}  p75={d['p75']:.4f}  p95={d['p95']:.4f}  max={d['max']:.4f}")

    out = args.output or str(Path(args.ckpt).parent / f"eval_metrics_nodule_{args.split}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(m, indent=2))
    print(f"\nWrote metrics → {out}")


if __name__ == "__main__":
    main()
