"""Render axial slices showing small-FP (speck) overprediction for an SW model.

For each requested case: run inference, label components, find FP
components (no GT overlap), pick the slices with the most small FPs
(size <= SMALL), and save a 2-panel PNG per slice:
left = CT, right = CT + GT contour (green) + prediction (red) with
small FPs circled (yellow) and large FPs marked (orange).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import build_eval_dataset
from eval_metrics_nodule import build_predictor, load_config, load_weights
from model import build_model
from transforms import build_transforms

STRUCT = ndimage.generate_binary_structure(3, 3)
SMALL = 27


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", required=True, help="comma-separated filename keys")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--slices-per-case", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    task = cfg.get("task", "nodule")
    wanted = set(args.cases.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_tf = build_transforms(task, cfg)
    ds = build_eval_dataset(task, cfg, val_tf, split_name="test")
    model = build_model(cfg["model"]).to(device).eval()
    load_weights(model, args.ckpt)
    predictor = build_predictor(model, cfg)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    done = 0
    for idx in range(len(ds)):
        sample = ds[idx]
        fname = sample.get("filename", "")
        if fname not in wanted:
            continue
        img = torch.as_tensor(sample["image"]).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = predictor(img)
        pred = logits.float().argmax(dim=1)[0].eq(1).cpu().numpy().astype(np.uint8)
        gt = (np.asarray(sample["label"])[0] > 0.5).astype(np.uint8)
        ct = np.asarray(sample["image"])[0]

        pr_lab, n_pr = ndimage.label(pred, structure=STRUCT)
        sizes = np.bincount(pr_lab.ravel())[1:]
        hit_ids = set(np.unique(pr_lab[gt > 0])) - {0}
        fp_small, fp_large = [], []
        for i, sz in enumerate(sizes, start=1):
            if i in hit_ids:
                continue
            (fp_small if sz <= SMALL else fp_large).append(i)
        small_mask = np.isin(pr_lab, fp_small)

        # rank axial slices (last axis) by number of distinct small FPs present
        per_slice = [(len(set(np.unique(pr_lab[:, :, z][small_mask[:, :, z]])) - {0}), z)
                     for z in range(pred.shape[2])]
        per_slice.sort(reverse=True)
        uid = Path(fname).stem[-12:]

        for rank, (nsmall, z) in enumerate(per_slice[:args.slices_per_case]):
            if nsmall == 0:
                break
            fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
            for ax in axes:
                ax.imshow(ct[:, :, z].T, cmap="gray", origin="lower")
                ax.set_xticks([]); ax.set_yticks([])
            axes[0].set_title(f"CT  (…{uid}, z={z})")
            axes[1].contour(gt[:, :, z].T, levels=[0.5], colors="lime", linewidths=1.5)
            axes[1].imshow(np.ma.masked_where(pred[:, :, z].T == 0, pred[:, :, z].T),
                           cmap="autumn", alpha=0.55, origin="lower", vmin=0, vmax=1)
            for comp in fp_small:
                ys, xs = np.where(pr_lab[:, :, z] == comp)
                if len(ys):
                    axes[1].add_patch(plt.Circle((ys.mean(), xs.mean()), 9,
                                                 fill=False, color="yellow", lw=1.6))
            for comp in fp_large:
                ys, xs = np.where(pr_lab[:, :, z] == comp)
                if len(ys):
                    axes[1].add_patch(plt.Circle((ys.mean(), xs.mean()), 14,
                                                 fill=False, color="orange", lw=1.6, ls="--"))
            axes[1].set_title(f"GT (zöld) + predikció (piros) — {nsmall} kis FP e szeleten "
                              f"(sárga: FP ≤{SMALL} vox, narancs szaggatott: nagy FP)")
            fig.tight_layout()
            fig.savefig(out / f"speck_{uid}_z{z}.png", dpi=110)
            plt.close(fig)
        n_gt = ndimage.label(gt, structure=STRUCT)[1]
        print(f"{fname}: GT={n_gt}, pred komp={n_pr}, kis FP={len(fp_small)}, nagy FP={len(fp_large)}", flush=True)
        done += 1
        if done == len(wanted):
            break


if __name__ == "__main__":
    main()
