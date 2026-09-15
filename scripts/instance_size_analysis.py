"""Component-size breakdown behind the instance metrics.

For every test case, records the voxel size of each predicted component
(split into GT-hitting vs false-positive) and each GT nodule (hit vs
missed). Post-process to answer: are the false-positive components real
over-segmentation or tiny specks that a minimum-size filter removes,
and what does such a filter cost in instance recall?

Usage: like scripts/instance_metrics.py (same args), writes
<out>.json with per-case size lists.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import build_eval_dataset                                # noqa: E402
from eval_metrics_nodule import build_predictor, load_config, load_weights  # noqa: E402
from model import build_model                                          # noqa: E402
from transforms import build_transforms                                # noqa: E402

STRUCT = ndimage.generate_binary_structure(3, 3)


def sizes(pred, gt):
    gt_lab, n_gt = ndimage.label(gt, structure=STRUCT)
    pr_lab, n_pr = ndimage.label(pred, structure=STRUCT)
    out = {"pred_hit": [], "pred_fp": [], "gt_hit": [], "gt_miss": []}
    if n_pr:
        pr_sizes = np.bincount(pr_lab.ravel())[1:]
        hit_ids = set(np.unique(pr_lab[gt > 0])) - {0}
        for i, sz in enumerate(pr_sizes, start=1):
            out["pred_hit" if i in hit_ids else "pred_fp"].append(int(sz))
    if n_gt:
        gt_sizes = np.bincount(gt_lab.ravel())[1:]
        hit_ids = set(np.unique(gt_lab[pred > 0])) - {0}
        for i, sz in enumerate(gt_sizes, start=1):
            out["gt_hit" if i in hit_ids else "gt_miss"].append(int(sz))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split",  default="test")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg  = load_config(args.config)
    task = cfg.get("task", "nodule")
    nodule_class = 1 if task == "nodule" else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_tf = build_transforms(task, cfg)
    ds = build_eval_dataset(task, cfg, val_tf, split_name=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    model = build_model(cfg["model"]).to(device).eval()
    load_weights(model, args.ckpt)
    predictor = build_predictor(model, cfg)
    amp = torch.bfloat16 if cfg.get("training", {}).get("amp", True) else None

    per_case = []
    for i, batch in enumerate(loader):
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp,
                            enabled=(amp is not None and device.type == "cuda")):
            logits = predictor(img)
        pred = logits.float().argmax(dim=1).eq(nodule_class)[0].cpu().numpy().astype(np.uint8)
        lab = lbl[0, 0] if lbl.ndim == 5 else lbl[0]
        gt = (lab.eq(nodule_class) if task == "joint" else (lab > 0.5)).cpu().numpy().astype(np.uint8)
        rec = sizes(pred, gt)
        rec["filename"] = batch.get("filename", [""])[0]
        per_case.append(rec)
        if (i + 1) % 40 == 0 or (i + 1) == len(loader):
            print(f"  {i+1}/{len(loader)}", flush=True)

    with open(args.output, "w") as f:
        json.dump({"per_case": per_case,
                   "_meta": {"config": args.config, "checkpoint": args.ckpt,
                             "split": args.split, "task": task}}, f)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
