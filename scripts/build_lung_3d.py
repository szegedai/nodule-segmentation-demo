"""Build per-series 3D lung masks for the joint (end-to-end) nodule task.

For every CT under {data_root}/ct_3d/ produce a matching 3D lung mask
under {out_dir}/<uid>.npz of shape (1, H, W, D) uint8.

Per-series routing:
  • If per-slice masks exist under {data_root}/roi_sem_seg_2d/<uid>_<NNNN>.npz
    → they are GT (NLST + NSCLC): stack them into a 3D mask.
  • Otherwise (LIDC-like): run the 2D ROI SegResNet on each axial slice
    at the model's native 256×256 resolution, resize the prediction back
    to the CT's native (H, W), and stack.

Both branches produce output in the same format so the joint training
pipeline is agnostic to how the mask was produced.

Prerequisites
-------------
- ${DATA_ROOT}/ct_3d/*.npz         — CT volumes
- ${DATA_ROOT}/roi_sem_seg_2d/*_*.npz — per-slice GT lung masks
                                        (only NLST + NSCLC have these)
- A trained 2D ROI checkpoint. Download from HuggingFace:
    huggingface-cli download Kakimaki00/roi-segresnet-2d --local-dir ./ckpts/roi

Usage
-----
    export DATA_ROOT=/path/to/unified
    python scripts/build_lung_3d.py \\
        --data-root         $DATA_ROOT \\
        --out-dir           $DATA_ROOT/lung_sem_seg_3d \\
        --roi-config        ./ckpts/roi/config.yaml \\
        --roi-checkpoint    ./ckpts/roi/model.pth \\
        --split-in          data/splits/unified_v2.json \\
        --split-out-ex-lidc data/splits/unified_v2_ex_lidc.json \\
        --manifest-out      lung_source_manifest.json
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Reuse the demo's model dispatcher without hard-coding sys.path here:
# call the script with `--demo-root /path/to/nodule-training-demo` if it
# lives outside the main repo.


def _add_demo_path(demo_root):
    demo_root = Path(demo_root).resolve()
    if not (demo_root / "model.py").exists():
        sys.exit(f"--demo-root {demo_root} does not contain model.py")
    sys.path.insert(0, str(demo_root))


def load_yaml(path):
    text = Path(path).read_text()
    text = os.path.expandvars(text)
    return yaml.safe_load(text)


def load_roi_state(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ("model_state_dict" in ck or "model" in ck):
        state = ck.get("model_state_dict", ck.get("model"))
    else:
        state = ck
    model.load_state_dict(state)
    return model


def stack_gt_slices(uid, slice_paths):
    """slice_paths already sorted by slice index. Return (H, W, D) uint8."""
    slices = [np.load(p)["data"] for p in slice_paths]  # each is (1, H, W) or (H, W)
    slices_2d = []
    for s in slices:
        if s.ndim == 3 and s.shape[0] == 1:
            slices_2d.append(s[0])
        elif s.ndim == 2:
            slices_2d.append(s)
        else:
            sys.exit(f"unexpected mask shape {s.shape} in {uid}")
    return np.stack(slices_2d, axis=-1).astype(np.uint8)  # (H, W, D)


@torch.no_grad()
def predict_lung_3d(model, ct3d, device, roi_size=256, batch_size=32):
    """CT (H, W, D) float32 [0,1] → (H, W, D) uint8 lung mask."""
    H, W, D = ct3d.shape
    slices = torch.from_numpy(ct3d).permute(2, 0, 1).unsqueeze(1).float()  # (D,1,H,W)
    slices_small = F.interpolate(slices, size=(roi_size, roi_size),
                                 mode="bilinear", align_corners=False).to(device)
    masks_small = torch.empty(D, 1, roi_size, roi_size, dtype=torch.bool, device=device)
    for i in range(0, D, batch_size):
        logits = model(slices_small[i:i + batch_size])
        masks_small[i:i + batch_size] = torch.sigmoid(logits) > 0.5
    masks_full = F.interpolate(masks_small.float(), size=(H, W),
                               mode="nearest").squeeze(1).to(torch.uint8)   # (D, H, W)
    return masks_full.permute(1, 2, 0).cpu().numpy()   # (H, W, D)


SLICE_RE = re.compile(r"^(?P<uid>.+)_(?P<idx>\d+)\.npz$")


def build_roi_2d_index(roi_dir):
    """{uid: [Path, ...] sorted by slice index} for all files in roi_sem_seg_2d/."""
    groups = defaultdict(list)
    for f in Path(roi_dir).iterdir():
        m = SLICE_RE.match(f.name)
        if not m:
            continue
        groups[m["uid"]].append((int(m["idx"]), f))
    for uid in groups:
        groups[uid].sort()
        groups[uid] = [p for _, p in groups[uid]]
    return dict(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",       required=True, help="dir with ct_3d/ and roi_sem_seg_2d/")
    ap.add_argument("--out-dir",         required=True, help="output dir for <uid>.npz files")
    ap.add_argument("--roi-config",      required=True, help="YAML config used at ROI training")
    ap.add_argument("--roi-checkpoint",  required=True, help="path to the 2D ROI checkpoint")
    ap.add_argument("--demo-root",       default=None, help="path to nodule-training-demo (for model.py). "
                    "Defaults to sibling of the main repo.")
    ap.add_argument("--split-in",        default=None,
                    help="input split JSON (unified_v2.json). If given, an ex-LIDC (GT-only) "
                         "variant is emitted alongside --split-out-ex-lidc.")
    ap.add_argument("--split-out-ex-lidc", default=None,
                    help="path for the ex-LIDC split JSON (keeps only series with GT lung "
                         "labels). Requires --split-in.")
    ap.add_argument("--manifest-out",    default=None,
                    help="write a JSON manifest mapping <uid> → 'GT' | 'PRED'.")
    ap.add_argument("--limit", type=int, default=None, help="cap number of series (for smoke tests)")
    ap.add_argument("--skip-existing", action="store_true", help="skip series whose output already exists")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roi-batch", type=int, default=32)
    args = ap.parse_args()

    if args.demo_root is None:
        # In the reproduction demo this script lives at scripts/build_lung_3d.py
        # and model.py sits in the parent (demo root).
        args.demo_root = str(Path(__file__).resolve().parent.parent)
    _add_demo_path(args.demo_root)

    from model import build_model     # from nodule-training-demo/model.py

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ct_dir  = data_root / "ct_3d"
    roi_dir = data_root / "roi_sem_seg_2d"
    if not ct_dir.exists():
        sys.exit(f"missing {ct_dir}")

    # Build ROI model once (needed for LIDC branch only, but cheap to have around).
    device = torch.device(args.device)
    roi_cfg = load_yaml(args.roi_config)
    roi_model = build_model(roi_cfg["model"]).to(device).eval()
    load_roi_state(roi_model, args.roi_checkpoint)
    roi_input_size = tuple(roi_cfg.get("preprocessing", {}).get("target_size", [256, 256]))
    roi_size = roi_input_size[0]

    print(f"[cfg] data_root      = {data_root}",  flush=True)
    print(f"[cfg] out_dir        = {out_dir}",    flush=True)
    print(f"[cfg] roi_config     = {args.roi_config}", flush=True)
    print(f"[cfg] roi_checkpoint = {args.roi_checkpoint}", flush=True)
    print(f"[cfg] roi_input_size = {roi_input_size}", flush=True)
    print(f"[cfg] device         = {device}",     flush=True)

    # Index the 2D per-slice GT masks once (avoids re-globbing per series).
    print("[index] scanning roi_sem_seg_2d/ …", flush=True)
    t0 = time.time()
    roi_index = build_roi_2d_index(roi_dir) if roi_dir.exists() else {}
    print(f"[index] {len(roi_index):,} series with GT lung slices  ({time.time()-t0:.1f} s)",
          flush=True)

    ct_files = sorted(ct_dir.glob("*.npz"))
    if args.limit:
        ct_files = ct_files[:args.limit]

    n_gt = n_pred = n_skipped = n_err = 0
    t_start = time.time()
    manifest = {}       # uid -> "GT" | "PRED"

    for i, ct_path in enumerate(ct_files):
        uid = ct_path.stem
        out_path = out_dir / f"{uid}.npz"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        try:
            npz = np.load(ct_path)
            ct = npz["data"]                             # (1, H, W, D) float32
            if ct.ndim != 4 or ct.shape[0] != 1:
                raise RuntimeError(f"unexpected CT shape {ct.shape}")
            H, W, D = ct.shape[1], ct.shape[2], ct.shape[3]
            ct3d = ct[0].astype(np.float32)              # (H, W, D)

            if uid in roi_index and len(roi_index[uid]) == D:
                mask3d = stack_gt_slices(uid, roi_index[uid])   # (H, W, D)
                if mask3d.shape != (H, W, D):
                    raise RuntimeError(f"stacked GT shape {mask3d.shape} != CT ({H},{W},{D})")
                source = "GT"
                manifest[uid] = "GT"
                n_gt += 1
            elif uid in roi_index and len(roi_index[uid]) != D:
                # NLST/NSCLC series with a partial GT — fall back to model prediction
                # rather than silently mixing GT and unset slices.
                mask3d = predict_lung_3d(roi_model, ct3d, device,
                                         roi_size=roi_size, batch_size=args.roi_batch)
                source = "PRED (partial GT — CT depth mismatch)"
                manifest[uid] = "PRED"
                n_pred += 1
            else:
                mask3d = predict_lung_3d(roi_model, ct3d, device,
                                         roi_size=roi_size, batch_size=args.roi_batch)
                source = "PRED"
                manifest[uid] = "PRED"
                n_pred += 1

            out = mask3d[None].astype(np.uint8)          # (1, H, W, D)
            np.savez_compressed(out_path, data=out)

            if (i + 1) % 25 == 0 or (i + 1) == len(ct_files):
                dt = time.time() - t_start
                per = dt / (i + 1 - n_skipped) if (i + 1 - n_skipped) > 0 else 0
                eta = per * (len(ct_files) - i - 1)
                print(f"[{i+1:>5}/{len(ct_files)}]  gt={n_gt}  pred={n_pred}  "
                      f"skip={n_skipped}  err={n_err}  "
                      f"last={source:<40}  uid={uid}  "
                      f"({dt:.0f} s elapsed, ETA {eta:.0f} s)", flush=True)
        except Exception as e:
            n_err += 1
            print(f"[ERR] {uid}: {e}", flush=True)

    print(f"\nDone. GT-stacked={n_gt}  ROI-predicted={n_pred}  "
          f"skipped={n_skipped}  errors={n_err}   "
          f"total={n_gt + n_pred + n_skipped + n_err} / {len(ct_files)}", flush=True)

    # ── side-artefact: manifest ──────────────────────────────────────────────
    if args.manifest_out:
        import json
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest_out).write_text(json.dumps(manifest, indent=2))
        print(f"[manifest] wrote {len(manifest):,} entries → {args.manifest_out}", flush=True)

    # ── side-artefact: ex-LIDC split JSON (keeps only series with a GT mask) ──
    if args.split_in and args.split_out_ex_lidc:
        import json
        with open(args.split_in) as f:
            split = json.load(f)
        gt_uids = {u for u, s in manifest.items() if s == "GT"}
        # Split keys look like "unified/<uid>.npz"; extract <uid> for the membership test.
        def key_uid(k):
            return Path(k).stem
        filtered = {"meta": {**split.get("meta", {}), "filter": "ex_lidc__gt_lung_only",
                             "source_manifest": args.manifest_out}}
        for k in ("train", "val", "test"):
            if k in split:
                filtered[k] = [s for s in split[k] if key_uid(s) in gt_uids]
                print(f"[split] {k}: {len(split[k]):,} → {len(filtered[k]):,} "
                      f"(dropped {len(split[k]) - len(filtered[k]):,} LIDC-like series)",
                      flush=True)
        Path(args.split_out_ex_lidc).parent.mkdir(parents=True, exist_ok=True)
        Path(args.split_out_ex_lidc).write_text(json.dumps(filtered, indent=2))
        print(f"[split] wrote → {args.split_out_ex_lidc}", flush=True)


if __name__ == "__main__":
    main()
