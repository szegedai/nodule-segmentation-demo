"""Evaluate a 2D ROI model on the validation split.

Reports the metrics the Jira ticket asks for:
    mIoU, Accuracy, Precision, Recall
plus Dice (for continuity with the training loop).

All metrics are computed at the per-pixel level, aggregated across the
entire validation split (micro-averaging) with a 0.5 threshold on the
sigmoid output. Also reports a per-case Dice histogram (min / mean / max)
so a single terrible case does not silently drag mean scores.

Usage:
    python eval_roi.py --config configs/roi.yaml       --ckpt checkpoints/roi/best.pth
    python eval_roi.py --config configs/roi_swin.yaml  --ckpt checkpoints/roi_swin/best_model.pth
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset   import build_datasets
from model     import build_model
from transforms import build_transforms


def load_config(path):
    text = Path(path).read_text()
    text = os.path.expandvars(text)
    return yaml.safe_load(text)


def load_ckpt_tolerant(path, model, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck.get("model"))
    model.load_state_dict(state)
    epoch = ck.get("epoch", -1)
    val   = ck.get("val_dice", ck.get("best_dice", None))
    return epoch, val


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    per_case_dice = []
    per_case_uid  = []
    t0 = time.time()
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(img)
        prob = torch.sigmoid(logits.float())
        pred = (prob > threshold).to(torch.uint8)
        gt   = (lbl > 0.5).to(torch.uint8)

        # global counts (int64 to avoid overflow — ~1.3M pixels/slice × 30k slices)
        p1 = pred.eq(1)
        g1 = gt.eq(1)
        tp += (p1 &  g1).sum().item()
        fp += (p1 & ~g1).sum().item()
        fn += (~p1 &  g1).sum().item()
        tn += (~p1 & ~g1).sum().item()

        # per-slice Dice for the histogram
        # pred, gt shape (B, 1, H, W) — reduce over (1,2,3)
        p_sum = pred.sum(dim=(1, 2, 3)).float()
        g_sum = gt.sum(dim=(1, 2, 3)).float()
        inter = (pred & gt).sum(dim=(1, 2, 3)).float()
        # avoid div-by-zero for empty-gt slices (background-only): treat as 1.0 if both empty
        denom = p_sum + g_sum
        dice_slice = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
        per_case_dice.extend(dice_slice.cpu().tolist())

        # keep filenames for context
        fns = batch.get("filename", batch.get("meta", [""] * pred.shape[0]))
        if isinstance(fns, (list, tuple)):
            per_case_uid.extend(fns)

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            dt = time.time() - t0
            print(f"  batch {i+1}/{n_batches}  ({dt:.1f} s elapsed)", flush=True)

    total = tp + fp + fn + tn
    smooth = 1e-9
    iou_fg = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    miou   = 0.5 * (iou_fg + iou_bg)
    acc    = (tp + tn) / max(total, 1)
    prec   = tp / max(tp + fp, 1)
    rec    = tp / max(tp + fn, 1)
    dice_micro = 2 * tp / max(2 * tp + fp + fn, 1)

    d = np.array(per_case_dice)
    return {
        "n_pixels":  total,
        "n_slices":  len(per_case_dice),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mIoU":      miou,
        "IoU_fg":    iou_fg,
        "IoU_bg":    iou_bg,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "Dice_micro": dice_micro,
        "Dice_per_slice": {
            "mean":   float(d.mean()) if len(d) else 0.0,
            "std":    float(d.std())  if len(d) else 0.0,
            "min":    float(d.min())  if len(d) else 0.0,
            "median": float(np.median(d)) if len(d) else 0.0,
            "max":    float(d.max())  if len(d) else 0.0,
            "p05":    float(np.percentile(d,  5)) if len(d) else 0.0,
            "p95":    float(np.percentile(d, 95)) if len(d) else 0.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--output", default=None, help="Where to write JSON metrics (default: alongside ckpt)")
    ap.add_argument("--batch",  type=int, default=None, help="override batch size for eval")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    task = cfg.get("task", "roi")
    if task != "roi":
        sys.exit(f"eval_roi.py only supports task=roi, got {task!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_tf, val_tf = build_transforms(task, cfg)
    _, val_ds = build_datasets(task, cfg, train_tf, val_tf)
    print(f"val slices: {len(val_ds):,}", flush=True)

    batch = args.batch or cfg.get("training", {}).get("batch_size", 16)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(cfg["model"]).to(device)
    ep, val = load_ckpt_tolerant(args.ckpt, model, device)
    print(f"loaded ckpt {args.ckpt}  (epoch={ep}, val_dice_at_save={val})", flush=True)

    amp_dtype = torch.bfloat16 if cfg.get("training", {}).get("amp", True) else None

    print(f"evaluating…", flush=True)
    metrics = evaluate(model, val_loader, device, amp_dtype)

    metrics["_meta"] = {
        "config":       args.config,
        "checkpoint":   args.ckpt,
        "epoch":        ep,
        "val_dice_at_save": val,
        "model_name":   cfg["model"].get("name"),
        "spatial_dims": cfg["model"].get("spatial_dims"),
    }

    print("\n" + "=" * 60)
    print(f"  ROI eval — {cfg['model'].get('name')}  ({args.ckpt})")
    print("=" * 60)
    print(f"  slices evaluated:   {metrics['n_slices']:,}")
    print(f"  pixels evaluated:   {metrics['n_pixels']:,}")
    print(f"  ────  Jira-required metrics  ────")
    print(f"  mIoU:               {metrics['mIoU']:.4f}")
    print(f"    IoU foreground:   {metrics['IoU_fg']:.4f}")
    print(f"    IoU background:   {metrics['IoU_bg']:.4f}")
    print(f"  Accuracy:           {metrics['Accuracy']:.4f}")
    print(f"  Precision:          {metrics['Precision']:.4f}")
    print(f"  Recall:             {metrics['Recall']:.4f}")
    print(f"  ────  supplementary  ────")
    print(f"  Dice (micro):       {metrics['Dice_micro']:.4f}")
    d = metrics["Dice_per_slice"]
    print(f"  Dice per slice:     mean={d['mean']:.4f}  std={d['std']:.4f}  "
          f"min={d['min']:.4f}  median={d['median']:.4f}  max={d['max']:.4f}")
    print(f"                      p05={d['p05']:.4f}  p95={d['p95']:.4f}")

    if args.output is None:
        args.output = str(Path(args.ckpt).with_suffix("").parent / f"eval_{cfg['model'].get('name')}.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote metrics → {args.output}")


if __name__ == "__main__":
    main()
