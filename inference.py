"""End-to-end nodule inference: raw CT → nodule mask via ROI + nodule models.

Two-stage pipeline that mirrors the training preprocessing exactly:

    1. Load a raw CT volume (npz, {'data': (1, H, W, D) float32 in [0, 1]}).
    2. Run the 2D ROI model slice-by-slice → per-slice lung mask → stacked
       into a 3D lung mask at native resolution.
    3. Derive a 3D lung bounding box from that mask (+ padding).
    4. Crop the CT to the bbox and resample to 256³ (trilinear).
    5. Run the 3D nodule model on the 256³ crop → argmax → binary nodule
       mask at 256³.
    6. Resample the nodule mask back to bbox size (nearest) and place it
       into an (H, W, D) full-frame canvas of zeros.
    7. Save as {'data': (1, H, W, D) uint8}.

Model checkpoints are expected to be either
  (a) local path to a .pth file (flat state_dict or wrapped) plus a
      config path, OR
  (b) a directory containing `model.pth` + `config.yaml` — this is the
      layout `huggingface-cli download` produces when pulling one of the
      companion HF repos.

Example:

    huggingface-cli download Kakimaki00/roi-swinunetr-2d       --local-dir ./ckpts/roi
    huggingface-cli download Kakimaki00/nodule-segresnet-3d-wide --local-dir ./ckpts/nodule
    python inference.py \\
        --ct         path/to/case.npz \\
        --roi-dir    ./ckpts/roi \\
        --nodule-dir ./ckpts/nodule \\
        --output     nodule_mask.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model


def load_config(path):
    text = Path(path).read_text()
    text = os.path.expandvars(text)
    return yaml.safe_load(text)


def load_weights(model, path):
    """Load a state_dict into `model`, tolerating both flat and wrapped formats."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ("model_state_dict" in ck or "model" in ck):
        state = ck.get("model_state_dict", ck.get("model"))
    else:
        state = ck
    model.load_state_dict(state)
    return model


def resolve_model_paths(arg, config_arg):
    """`arg` may be a directory (from huggingface-cli download) or a .pth file.

    Returns (config_path, weights_path).
    """
    p = Path(arg)
    if p.is_dir():
        cfg = p / "config.yaml"
        weights = p / "model.pth"
        if not cfg.exists() or not weights.exists():
            sys.exit(f"expected {cfg} and {weights} in {arg}")
        return str(cfg), str(weights)
    if p.is_file():
        if config_arg is None:
            sys.exit(f"--config required when the model arg is a .pth file: {arg}")
        return config_arg, arg
    sys.exit(f"model arg is neither a file nor a directory: {arg}")


@torch.no_grad()
def run_roi(model, ct3d, device, roi_size=256, batch_size=32):
    """CT (H, W, D) → per-slice lung mask (H, W, D) uint8.

    The model was trained on 256x256 slices resized via bilinear
    interpolation; we reproduce the exact preprocessing here.
    """
    H, W, D = ct3d.shape

    # (D, 1, H, W) — one channel-first slice per axial position
    slices = torch.from_numpy(ct3d).permute(2, 0, 1).unsqueeze(1).float()

    # Resize slices to the model's native (256, 256)
    slices_small = F.interpolate(
        slices, size=(roi_size, roi_size),
        mode="bilinear", align_corners=False,
    ).to(device)

    masks_small = torch.empty(D, 1, roi_size, roi_size, dtype=torch.bool, device=device)
    for i in range(0, D, batch_size):
        logits = model(slices_small[i:i + batch_size])
        masks_small[i:i + batch_size] = torch.sigmoid(logits) > 0.5

    # Resize per-slice masks back to (H, W) via nearest — binary → binary
    masks_full = F.interpolate(
        masks_small.float(), size=(H, W), mode="nearest",
    ).squeeze(1).to(torch.uint8)              # (D, H, W)

    return masks_full.permute(1, 2, 0).cpu().numpy()   # (H, W, D)


def compute_bbox(mask3d, padding, min_lung_voxels=1000):
    """(H, W, D) uint8 → (h0, h1, w0, w1, d0, d1) axis-aligned bbox with padding."""
    n = int(mask3d.sum())
    if n < min_lung_voxels:
        sys.exit(f"ROI model predicted only {n} lung voxels — too little to derive a bbox")
    idx = np.argwhere(mask3d > 0)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    H, W, D = mask3d.shape
    h0 = max(0, lo[0] - padding); h1 = min(H, hi[0] + padding)
    w0 = max(0, lo[1] - padding); w1 = min(W, hi[1] + padding)
    d0 = max(0, lo[2] - padding); d1 = min(D, hi[2] + padding)
    return h0, h1, w0, w1, d0, d1


@torch.no_grad()
def run_nodule(model, ct3d, bbox, device, nod_size=256):
    """CT (H, W, D) + bbox → full-frame nodule mask (H, W, D) uint8.

    The nodule model was trained on 256³ crops (trilinear-resized from
    the bbox); we mirror that preprocessing.
    """
    h0, h1, w0, w1, d0, d1 = bbox
    crop = ct3d[h0:h1, w0:w1, d0:d1]                  # (h, w, d)
    ch, cw, cd = crop.shape

    x = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).float()   # (1, 1, h, w, d)
    x = F.interpolate(x, size=(nod_size, nod_size, nod_size),
                      mode="trilinear", align_corners=False).to(device)

    logits = model(x)                                   # (1, 2, s, s, s)
    pred_small = logits.argmax(dim=1, keepdim=True).float()  # (1, 1, s, s, s)

    # Resize prediction back to the crop's native shape via nearest
    pred_bbox = F.interpolate(pred_small, size=(ch, cw, cd),
                              mode="nearest").squeeze().to(torch.uint8).cpu().numpy()

    H, W, D = ct3d.shape
    full = np.zeros((H, W, D), dtype=np.uint8)
    full[h0:h1, w0:w1, d0:d1] = pred_bbox
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ct", required=True,
                    help="input CT npz with 'data' key of shape (1, H, W, D), float32 in [0, 1]")
    ap.add_argument("--roi-dir",     help="directory containing config.yaml + model.pth (from `huggingface-cli download`)")
    ap.add_argument("--roi-weights", help="alternative to --roi-dir: path to model.pth")
    ap.add_argument("--roi-config",  help="alternative to --roi-dir: path to config.yaml")
    ap.add_argument("--nodule-dir",     help="directory containing config.yaml + model.pth (from `huggingface-cli download`)")
    ap.add_argument("--nodule-weights", help="alternative to --nodule-dir: path to model.pth")
    ap.add_argument("--nodule-config",  help="alternative to --nodule-dir: path to config.yaml")
    ap.add_argument("--output", required=True, help="output nodule mask npz")
    ap.add_argument("--bbox-padding", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roi-batch", type=int, default=32, help="batch size for ROI stage")
    args = ap.parse_args()

    # Resolve model paths.
    if args.roi_dir:
        roi_cfg_path, roi_w_path = resolve_model_paths(args.roi_dir, args.roi_config)
    elif args.roi_weights:
        if not args.roi_config:
            sys.exit("--roi-config required when using --roi-weights")
        roi_cfg_path, roi_w_path = args.roi_config, args.roi_weights
    else:
        sys.exit("provide either --roi-dir or (--roi-weights + --roi-config)")

    if args.nodule_dir:
        nod_cfg_path, nod_w_path = resolve_model_paths(args.nodule_dir, args.nodule_config)
    elif args.nodule_weights:
        if not args.nodule_config:
            sys.exit("--nodule-config required when using --nodule-weights")
        nod_cfg_path, nod_w_path = args.nodule_config, args.nodule_weights
    else:
        sys.exit("provide either --nodule-dir or (--nodule-weights + --nodule-config)")

    device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    # Load CT.
    npz = np.load(args.ct)
    if "data" not in npz:
        sys.exit(f"{args.ct} has no 'data' key; keys={list(npz.keys())}")
    ct = npz["data"]
    if ct.ndim != 4 or ct.shape[0] != 1:
        sys.exit(f"expected CT shape (1, H, W, D), got {ct.shape}")
    ct3d = ct[0].astype(np.float32)                       # (H, W, D)
    print(f"CT:      shape={ct.shape}  dtype={ct.dtype}  range=[{ct3d.min():.3f}, {ct3d.max():.3f}]", flush=True)

    # ── ROI stage ─────────────────────────────────────────────────────────────
    roi_cfg = load_config(roi_cfg_path)
    roi_model = build_model(roi_cfg["model"]).to(device).eval()
    load_weights(roi_model, roi_w_path)
    roi_size = tuple(roi_cfg.get("preprocessing", {}).get("target_size", [256, 256]))
    print(f"ROI:     {roi_cfg['model'].get('name', 'segresnet')}  "
          f"({sum(p.numel() for p in roi_model.parameters())/1e6:.1f} M params)", flush=True)

    lung_mask = run_roi(roi_model, ct3d, device,
                        roi_size=roi_size[0], batch_size=args.roi_batch)
    print(f"         lung voxels predicted: {int(lung_mask.sum()):,}  "
          f"({100 * lung_mask.mean():.2f} % of volume)", flush=True)

    # Free ROI model before loading the nodule model — nodule models are larger.
    del roi_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Bbox derivation.
    bbox = compute_bbox(lung_mask, padding=args.bbox_padding)
    h0, h1, w0, w1, d0, d1 = bbox
    print(f"bbox:    h[{h0}:{h1}] w[{w0}:{w1}] d[{d0}:{d1}]  "
          f"crop shape=({h1-h0}, {w1-w0}, {d1-d0})", flush=True)

    # ── nodule stage ─────────────────────────────────────────────────────────
    nod_cfg = load_config(nod_cfg_path)
    nod_model = build_model(nod_cfg["model"]).to(device).eval()
    load_weights(nod_model, nod_w_path)
    nod_size = tuple(nod_cfg.get("preprocessing", {}).get("target_size", [256, 256, 256]))
    print(f"nodule:  {nod_cfg['model'].get('name', 'segresnet')}  "
          f"({sum(p.numel() for p in nod_model.parameters())/1e6:.1f} M params)", flush=True)

    nodule_mask = run_nodule(nod_model, ct3d, bbox, device, nod_size=nod_size[0])
    print(f"         nodule voxels predicted: {int(nodule_mask.sum()):,}  "
          f"({100 * nodule_mask.mean():.4f} % of volume)", flush=True)

    # Save.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, data=nodule_mask[None])   # (1, H, W, D) uint8
    print(f"wrote → {out_path}", flush=True)


if __name__ == "__main__":
    main()
