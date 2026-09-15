"""Instance-level (per-nodule) recall and precision on a held-out split.

Definitions (26-connectivity connected components):
  - instance recall    = #GT nodules touched by >=1 predicted voxel
                         / #GT nodules
  - instance precision = #predicted components touching >=1 GT voxel
                         / #predicted components

Counts are pooled over all cases of the split (instance-micro). Works for
the two-stage nodule models (task: nodule, foreground = class 1) and the
joint models (task: joint, nodule = class 2). Same preprocessing grid as
eval_metrics_*: resize-mode models are scored on the 256-cube grid, SW
models on the native grid.

Usage:
    export DATA_ROOT=/path/to/unified
    python scripts/instance_metrics.py --config configs/segresnet_small.yaml \\
        --ckpt checkpoints/segresnet_small/best_model.pth --split test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import build_eval_dataset                                # noqa: E402
from eval_metrics_nodule import build_predictor, load_config, load_weights  # noqa: E402
from model import build_model, count_parameters                       # noqa: E402
from transforms import build_transforms                               # noqa: E402

STRUCT = ndimage.generate_binary_structure(3, 3)   # 26-connectivity


def instance_stats(pred, gt):
    """pred, gt: binary 3D uint8 arrays → per-case instance counts."""
    gt_lab, n_gt = ndimage.label(gt, structure=STRUCT)
    pr_lab, n_pr = ndimage.label(pred, structure=STRUCT)
    gt_hit = 0
    if n_gt:
        hit_ids = np.unique(gt_lab[pred > 0])
        gt_hit = int((hit_ids > 0).sum())
    pr_hit = 0
    if n_pr:
        hit_ids = np.unique(pr_lab[gt > 0])
        pr_hit = int((hit_ids > 0).sum())
    return n_gt, gt_hit, n_pr, pr_hit


@torch.no_grad()
def run(model, loader, device, amp_dtype, nodule_class, task, predictor=None):
    forward = predictor if predictor is not None else model
    tot = {"n_gt": 0, "n_gt_hit": 0, "n_pred": 0, "n_pred_hit": 0}
    per_case = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device.type == "cuda")):
            logits = forward(img)
        pred = logits.float().argmax(dim=1).eq(nodule_class)[0].cpu().numpy().astype(np.uint8)
        # Both tasks yield (B, 1, H, W, D) labels: binary float mask for the
        # nodule task, {0,1,2} int map for joint — select the nodule class
        # by task, never by shape.
        lab = lbl[0, 0] if lbl.ndim == 5 else lbl[0]
        gt = (lab.eq(nodule_class) if task == "joint"
              else (lab > 0.5)).cpu().numpy().astype(np.uint8)

        n_gt, gt_hit, n_pr, pr_hit = instance_stats(pred, gt)
        tot["n_gt"] += n_gt; tot["n_gt_hit"] += gt_hit
        tot["n_pred"] += n_pr; tot["n_pred_hit"] += pr_hit
        per_case.append({"filename": batch.get("filename", [""])[0],
                         "n_gt": n_gt, "n_gt_hit": gt_hit,
                         "n_pred": n_pr, "n_pred_hit": pr_hit})
        if (i + 1) % 20 == 0 or (i + 1) == len(loader):
            print(f"  case {i+1}/{len(loader)}  ({time.time()-t0:.1f} s)", flush=True)

    return {
        **tot,
        "instance_recall":    tot["n_gt_hit"]   / max(tot["n_gt"], 1),
        "instance_precision": tot["n_pred_hit"] / max(tot["n_pred"], 1),
        "connectivity": 26,
        "per_case": per_case,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--split",  default="test", choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg  = load_config(args.config)
    task = cfg.get("task", "nodule")
    if task not in ("nodule", "joint"):
        sys.exit(f"instance metrics are defined for nodule/joint tasks, got {task!r}")
    nodule_class = 1 if task == "nodule" else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = build_transforms(task, cfg)
    ds = build_eval_dataset(task, cfg, val_tf, split_name=args.split)
    print(f"{args.split} cases: {len(ds):,}", flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_model(cfg["model"]).to(device).eval()
    ep = load_weights(model, args.ckpt)
    print(f"model={cfg['model'].get('name')} params={count_parameters(model):,} "
          f"ckpt={args.ckpt} (epoch={ep})", flush=True)

    amp_dtype = torch.bfloat16 if cfg.get("training", {}).get("amp", True) else None
    m = run(model, loader, device, amp_dtype, nodule_class, task,
            predictor=build_predictor(model, cfg))
    m["_meta"] = {"config": args.config, "checkpoint": args.ckpt,
                  "eval_split": args.split, "epoch": ep, "task": task,
                  "split_json": cfg["data"].get("split_json")}

    print(f"\n  GT nodules:        {m['n_gt']:,}  hit: {m['n_gt_hit']:,}")
    print(f"  Instance recall:    {m['instance_recall']:.4f}")
    print(f"  Pred components:   {m['n_pred']:,}  hit: {m['n_pred_hit']:,}")
    print(f"  Instance precision: {m['instance_precision']:.4f}")

    out = args.output or (Path(args.ckpt).parent /
                          f"instance_metrics_{Path(args.config).stem}_{args.split}.json")
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
