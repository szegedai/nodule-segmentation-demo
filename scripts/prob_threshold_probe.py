"""Does raising the decision threshold remove the speck FPs?

Runs the SW model once per test case, keeps the class-1 softmax
probability map, and for each threshold counts predicted components,
FP components, and hit GT nodules. If specks sit near p=0.5 they are a
calibration/loss-weight artefact; if they persist at p=0.99 the model
is confidently wrong from local appearance.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import build_eval_dataset
from eval_metrics_nodule import build_predictor, load_config, load_weights
from model import build_model
from transforms import build_transforms

STRUCT = ndimage.generate_binary_structure(3, 3)
THS = [0.5, 0.7, 0.9, 0.95, 0.99]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config); task = cfg.get("task", "nodule")
    device = torch.device("cuda")
    _, val_tf = build_transforms(task, cfg)
    ds = build_eval_dataset(task, cfg, val_tf, split_name="test")
    loader = DataLoader(ds, batch_size=1, num_workers=a.workers, pin_memory=True)
    model = build_model(cfg["model"]).to(device).eval()
    load_weights(model, a.ckpt)
    predictor = build_predictor(model, cfg)
    agg = {t: {"n_pred": 0, "n_fp": 0, "n_gt": 0, "n_gt_hit": 0,
               "fp_small": 0} for t in THS}
    for i, b in enumerate(loader):
        img = b["image"].to(device); lbl = b["label"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = predictor(img)
        prob = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
        gt = (lbl[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        gl, ng = ndimage.label(gt, structure=STRUCT)
        for t in THS:
            pred = (prob >= t).astype(np.uint8)
            pl, npd = ndimage.label(pred, structure=STRUCT)
            if npd:
                sizes = np.bincount(pl.ravel())[1:]
                hit = set(np.unique(pl[gt > 0])) - {0}
                fp = [k for k in range(1, npd + 1) if k not in hit]
                agg[t]["n_pred"] += npd; agg[t]["n_fp"] += len(fp)
                agg[t]["fp_small"] += sum(1 for k in fp if sizes[k - 1] <= 27)
            agg[t]["n_gt"] += ng
            if ng:
                agg[t]["n_gt_hit"] += len(set(np.unique(gl[pred > 0])) - {0})
        if (i + 1) % 40 == 0 or i + 1 == len(loader):
            print(f"  {i+1}/{len(loader)}", flush=True)
    json.dump({str(t): v for t, v in agg.items()}, open(a.output, "w"), indent=2)
    print("wrote", a.output)

if __name__ == "__main__":
    main()
