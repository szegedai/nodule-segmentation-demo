"""Evaluate a joint (3-class) model on the val split.

Reports the ticket metric set (mIoU / Accuracy / Precision / Recall) with
a per-class breakdown for {background, lung, nodule}, plus per-case Dice
statistics for context.

Usage:
    export DATA_ROOT=/path/to/unified
    python eval_metrics_joint.py \\
        --config configs/joint_segresnet_ex_lidc.yaml \\
        --ckpt   checkpoints/joint_segresnet_ex_lidc/best_model.pth
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset    import build_eval_dataset
from model      import build_model, count_parameters
from transforms import build_transforms

from monai.inferers import sliding_window_inference

CLASS_NAMES = ["background", "lung", "nodule"]


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
def evaluate(model, loader, device, amp_dtype, num_classes=3, predictor=None):
    model.eval()
    forward = predictor if predictor is not None else model
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    tn = [0] * num_classes
    n_correct = 0
    n_total   = 0
    per_case_dice = {c: [] for c in range(num_classes)}
    per_case_uid  = []

    t0 = time.time()
    n_batches = len(loader)
    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            logits = forward(img)

        pred = logits.float().argmax(dim=1)               # (B, H, W, D)
        gt   = lbl[:, 0].long() if lbl.ndim == 5 else lbl.long()

        n_correct += int((pred == gt).sum())
        n_total   += int(gt.numel())

        for c in range(num_classes):
            p_c = pred.eq(c)
            g_c = gt.eq(c)
            tp[c] += int((p_c &  g_c).sum())
            fp[c] += int((p_c & ~g_c).sum())
            fn[c] += int((~p_c &  g_c).sum())
            tn[c] += int((~p_c & ~g_c).sum())

            spatial = tuple(range(1, gt.ndim))
            inter = (p_c & g_c).sum(dim=spatial).float()
            denom = p_c.sum(dim=spatial).float() + g_c.sum(dim=spatial).float()
            dice_c = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
            per_case_dice[c].extend(dice_c.cpu().tolist())

        per_case_uid.extend(batch.get("filename", [""] * pred.shape[0]))

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            dt = time.time() - t0
            print(f"  case {i+1}/{n_batches}  ({dt:.1f} s elapsed)", flush=True)

    metrics = {"num_classes": num_classes, "per_class": {}}
    iou_list, prec_list, rec_list = [], [], []
    for c in range(num_classes):
        iou_c  = tp[c] / max(tp[c] + fp[c] + fn[c], 1)
        prec_c = tp[c] / max(tp[c] + fp[c], 1)
        rec_c  = tp[c] / max(tp[c] + fn[c], 1)
        dice_c = 2 * tp[c] / max(2 * tp[c] + fp[c] + fn[c], 1)
        d = np.array(per_case_dice[c])
        metrics["per_class"][CLASS_NAMES[c]] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c], "tn": tn[c],
            "IoU":       iou_c,
            "Precision": prec_c,
            "Recall":    rec_c,
            "Dice_micro": dice_c,
            "Dice_per_case": {
                "mean":   float(d.mean()) if len(d) else 0.0,
                "std":    float(d.std())  if len(d) else 0.0,
                "min":    float(d.min())  if len(d) else 0.0,
                "median": float(np.median(d)) if len(d) else 0.0,
                "max":    float(d.max())  if len(d) else 0.0,
                "p25":    float(np.percentile(d, 25)) if len(d) else 0.0,
                "p75":    float(np.percentile(d, 75)) if len(d) else 0.0,
            },
        }
        iou_list.append(iou_c)
        prec_list.append(prec_c)
        rec_list.append(rec_c)

    metrics["mIoU"]              = float(np.mean(iou_list))
    metrics["mIoU_no_bg"]        = float(np.mean(iou_list[1:]))
    metrics["Accuracy"]          = n_correct / max(n_total, 1)
    metrics["Precision_macro"]   = float(np.mean(prec_list))
    metrics["Recall_macro"]      = float(np.mean(rec_list))
    metrics["Precision_no_bg"]   = float(np.mean(prec_list[1:]))
    metrics["Recall_no_bg"]      = float(np.mean(rec_list[1:]))
    metrics["n_voxels"]          = n_total
    metrics["n_cases"]           = len(per_case_uid)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--split",  default="val", choices=["train", "val", "test"],
                    help="Which split to evaluate on (default: val).")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    task = cfg.get("task", "joint")
    if task != "joint":
        sys.exit(f"eval_metrics_joint.py only supports task=joint, got {task!r}")

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
    m = evaluate(model, val_loader, device, amp_dtype,
                 num_classes=cfg["model"]["out_channels"], predictor=predictor)

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
    print(f"  Joint eval — {cfg['model'].get('name', 'segresnet')}   split={args.split}")
    print(f"  checkpoint: {args.ckpt}")
    print("=" * 60)
    print(f"  val cases:      {m['n_cases']:,}")
    print(f"  voxels:         {m['n_voxels']:,}")
    print(f"  ────  Ticket-required metrics  ────")
    print(f"  mIoU (all 3 classes):  {m['mIoU']:.4f}")
    print(f"  mIoU (lung + nodule):  {m['mIoU_no_bg']:.4f}")
    print(f"  Accuracy:              {m['Accuracy']:.6f}")
    print(f"  Precision (macro):     {m['Precision_macro']:.4f}    (no-bg: {m['Precision_no_bg']:.4f})")
    print(f"  Recall    (macro):     {m['Recall_macro']:.4f}    (no-bg: {m['Recall_no_bg']:.4f})")
    print(f"  ────  per-class  ────")
    for c in CLASS_NAMES:
        p = m["per_class"][c]
        d = p["Dice_per_case"]
        print(f"  {c:>10}   IoU={p['IoU']:.4f}   P={p['Precision']:.4f}   R={p['Recall']:.4f}   "
              f"Dice(micro)={p['Dice_micro']:.4f}   "
              f"Dice(per-case): mean={d['mean']:.4f} median={d['median']:.4f}")

    out = args.output or str(Path(args.ckpt).parent / f"eval_metrics_joint_{args.split}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(m, indent=2))
    print(f"\nWrote metrics → {out}")


if __name__ == "__main__":
    main()
