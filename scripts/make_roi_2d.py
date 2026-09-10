"""Build the 2D ROI training data (ct_2d/ + roi_sem_seg_2d/) from 3D volumes.

Slices every $DATA_ROOT/ct_3d/<uid>.npz along the axial (last) axis into
ct_2d/<uid>_<NNNN>.npz, and produces the matching per-slice lung mask in
roi_sem_seg_2d/ from one of two sources:

  --lung_dir DIR      slice existing 3D lung masks (GT case: NLST, NSCLC-R)
  --roi_config Y --roi_ckpt P
                      predict per slice with a trained ROI model
                      (pseudo-label case: LIDC-IDRI)

Output format matches Roi2DDataset: key 'data', CT (1, H, W) float32 [0,1],
mask (1, H, W) uint8 {0,1}, filenames '<series_uid>_<NNNN>.npz'.

Usage:
    python scripts/make_roi_2d.py --data_root $DATA_ROOT --lung_dir $DATA_ROOT/lung_sem_seg_3d
    python scripts/make_roi_2d.py --data_root $DATA_ROOT \
        --roi_config configs/roi.yaml --roi_ckpt checkpoints/roi/best.pth \
        --only_missing
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Dataset root with ct_3d/; outputs go to ct_2d/ and roi_sem_seg_2d/")
    p.add_argument("--lung_dir", default=None,
                   help="Dir with 3D lung masks (<uid>.npz) to slice as GT")
    p.add_argument("--roi_config", default=None, help="ROI model config yaml")
    p.add_argument("--roi_ckpt", default=None, help="ROI model checkpoint")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=32, help="slices per forward (predict mode)")
    p.add_argument("--only_missing", action="store_true",
                   help="Skip series whose mask slices already exist (e.g. GT series)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_roi_model(cfg_path, ckpt_path, device):
    from model import build_model
    cfg = yaml.safe_load(os.path.expandvars(open(cfg_path).read()))
    model = build_model(cfg["model"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state)
    model.eval()
    size = tuple(cfg.get("preprocessing", {}).get("target_size", [256, 256]))
    return model, size


@torch.no_grad()
def predict_masks(model, size, ct3d, batch, threshold, device):
    """ct3d (H, W, D) -> per-slice bool masks (D, H, W) at native resolution."""
    h, w, d = ct3d.shape
    stack = torch.from_numpy(ct3d).permute(2, 0, 1).unsqueeze(1)          # (D,1,H,W)
    out = torch.zeros((d, h, w), dtype=torch.bool)
    for i in range(0, d, batch):
        chunk = stack[i:i + batch].to(device)
        chunk = F.interpolate(chunk, size=size, mode="bilinear", align_corners=False)
        mask = torch.sigmoid(model(chunk)) > threshold                    # (b,1,*size)
        mask = F.interpolate(mask.float(), size=(h, w), mode="nearest")
        out[i:i + batch] = mask[:, 0].bool().cpu()
    return out.numpy()


def main():
    args = parse_args()
    if (args.lung_dir is None) == (args.roi_config is None):
        sys.exit("choose exactly one mask source: --lung_dir OR --roi_config/--roi_ckpt")
    if args.roi_config and not args.roi_ckpt:
        sys.exit("--roi_ckpt is required with --roi_config")

    root = Path(args.data_root)
    ct2d_dir = root / "ct_2d"
    roi2d_dir = root / "roi_sem_seg_2d"
    ct2d_dir.mkdir(parents=True, exist_ok=True)
    roi2d_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.roi_config:
        model, size = load_roi_model(args.roi_config, args.roi_ckpt, args.device)

    volumes = sorted((root / "ct_3d").glob("*.npz"))
    print(f"{len(volumes)} series in {root / 'ct_3d'}")
    for n, path in enumerate(volumes, 1):
        uid = path.stem
        if args.only_missing and (roi2d_dir / f"{uid}_0000.npz").exists():
            continue
        ct = np.load(path)["data"][0].astype(np.float32)                  # (H, W, D)

        if model is not None:
            masks = predict_masks(model, size, ct, args.batch, args.threshold, args.device)
        else:
            lung_path = Path(args.lung_dir) / path.name
            if not lung_path.exists():
                print(f"  [skip] no lung mask for {uid}")
                continue
            lung = np.load(lung_path)["data"][0]                          # (H, W, D)
            masks = np.moveaxis(lung > 0, -1, 0)                          # (D, H, W)

        for i in range(ct.shape[-1]):
            fname = f"{uid}_{i:04d}.npz"
            np.savez_compressed(ct2d_dir / fname, data=ct[None, :, :, i])
            np.savez_compressed(roi2d_dir / fname, data=masks[i][None].astype(np.uint8))
        print(f"  [{n}/{len(volumes)}] {uid}: {ct.shape[-1]} slices")


if __name__ == "__main__":
    main()
