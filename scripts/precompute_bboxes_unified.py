"""
Compute 3D lung bounding boxes for the unified dataset directly from
roi_sem_seg_2d (per-axial-slice lung masks). No model required.

The output JSON mirrors the format produced by precompute_bboxes.py
(consumed by stage2_fine/dataset.py):
    {
      "<dataset_name>/<series_uid>.npz": {"bbox": [h0, h1, w0, w1, d0, d1]}
    }

Usage:
    python scripts/precompute_bboxes_unified.py \
        --root    /project/AINoduleSeg/radiology/data/processed/unified \
        --name    unified \
        --padding 20 \
        --out     processed/bboxes_unified.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="root containing ct_3d/, nodule_sem_seg_3d/, roi_sem_seg_2d/")
    p.add_argument("--name", default="unified",
                   help="dataset name to prefix in JSON keys")
    p.add_argument("--padding", type=int, default=20,
                   help="voxel padding around the derived bbox")
    p.add_argument("--out", required=True, help="output bbox JSON path")
    return p.parse_args()


def group_slices_by_series(roi_dir: Path) -> dict[str, list[Path]]:
    """Filenames are '<series_uid>_<NNNN>.npz'. Group all slices by series."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in roi_dir.iterdir():
        if not f.name.endswith(".npz"):
            continue
        # last '_<digits>.npz' is the slice index
        stem = f.stem            # 'XXX_NNNN'
        idx_str = stem.rsplit("_", 1)[-1]
        if not idx_str.isdigit():
            continue
        series = stem.rsplit("_", 1)[0]
        groups[series].append(f)
    # Sort slices within each series by slice index
    for s in groups:
        groups[s].sort(key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    return dict(groups)


def bbox_from_slices(slice_paths: list[Path]) -> tuple[int, int, int, int, int, int] | None:
    """Return (h0, h1, w0, w1, d0, d1) in mask-array coordinates (HALF-OPEN ranges).

    h, w are spatial axes of each 2D slice; d is the slice index.
    Returns None if every slice is empty.
    """
    h_min = float("inf")
    h_max = -1
    w_min = float("inf")
    w_max = -1
    z_indices: list[int] = []
    for sp in slice_paths:
        with np.load(sp) as z:
            mask = z["data"]
        if mask.ndim == 3:
            mask = mask[0]      # (1, H, W) → (H, W)
        m = mask > 0.5
        if not m.any():
            continue
        slice_idx = int(sp.stem.rsplit("_", 1)[-1])
        z_indices.append(slice_idx)
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        h_min = min(h_min, int(rows.min()))
        h_max = max(h_max, int(rows.max()) + 1)
        w_min = min(w_min, int(cols.min()))
        w_max = max(w_max, int(cols.max()) + 1)
    if not z_indices:
        return None
    return (int(h_min), int(h_max),
            int(w_min), int(w_max),
            int(min(z_indices)), int(max(z_indices)) + 1)


def main():
    args = parse_args()
    root = Path(args.root)
    ct_dir = root / "ct_3d"
    roi_dir = root / "roi_sem_seg_2d"

    if not ct_dir.is_dir():
        sys.exit(f"missing {ct_dir}")
    if not roi_dir.is_dir():
        sys.exit(f"missing {roi_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Indexing slice files in {roi_dir} …", flush=True)
    groups = group_slices_by_series(roi_dir)
    print(f"  {len(groups)} series have slice files", flush=True)

    # Ensure each series has a matching CT (gives us volume shape for clipping)
    ct_files = {p.stem: p for p in ct_dir.glob("*.npz")}
    matched = [s for s in groups if s in ct_files]
    print(f"  {len(matched)} of those also have a matching CT in ct_3d/", flush=True)

    bboxes: dict[str, dict] = {}
    skipped: list[str] = []
    t0 = time.time()

    for i, series in enumerate(sorted(matched)):
        slice_paths = groups[series]
        bb = bbox_from_slices(slice_paths)
        if bb is None:
            skipped.append(series)
            continue
        h0, h1, w0, w1, d0, d1 = bb

        # Clip to CT volume dimensions (need shape for that)
        with np.load(ct_files[series]) as z:
            ct_shape = z["data"].shape          # (1, H, W, D)
        H, W, D = ct_shape[1], ct_shape[2], ct_shape[3]

        pad = args.padding
        h0 = max(0, h0 - pad);  h1 = min(H, h1 + pad)
        w0 = max(0, w0 - pad);  w1 = min(W, w1 + pad)
        d0 = max(0, d0 - pad);  d1 = min(D, d1 + pad)

        key = f"{args.name}/{series}.npz"
        bboxes[key] = {"bbox": [h0, h1, w0, w1, d0, d1]}

        if (i + 1) % 100 == 0 or (i + 1) == len(matched):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(matched) - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1:5d}/{len(matched)}]  "
                  f"elapsed={elapsed:6.1f}s  rate={rate:5.1f}/s  ETA={eta:5.0f}s",
                  flush=True)

    with open(out_path, "w") as f:
        json.dump(bboxes, f)
    print(f"\nwrote {out_path}  ({len(bboxes)} bboxes, {len(skipped)} skipped)", flush=True)


if __name__ == "__main__":
    main()
