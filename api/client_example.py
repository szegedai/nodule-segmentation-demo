"""Minimal Python client for the Nodule Segmentation API.

Usage:
    python client_example.py path/to/case.npz [--host http://localhost:8000] [--out mask.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ct", help="input CT .npz")
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--out",  default="nodule_mask.npz")
    args = ap.parse_args()

    with open(args.ct, "rb") as f:
        r = requests.post(f"{args.host}/predict", files={"file": f})
    if r.status_code != 200:
        print(f"[ERROR {r.status_code}] {r.text}", file=sys.stderr)
        sys.exit(1)

    Path(args.out).write_bytes(r.content)
    print(f"wrote → {args.out}")
    print(f"  CT shape        : {r.headers.get('X-CT-Shape')}")
    print(f"  bbox            : {r.headers.get('X-Bbox')}")
    print(f"  lung voxels     : {r.headers.get('X-Lung-Voxels')}")
    print(f"  nodule voxels   : {r.headers.get('X-Nodule-Voxels')}")
    print(f"  ROI stage       : {r.headers.get('X-Duration-Roi-Ms')} ms")
    print(f"  nodule stage    : {r.headers.get('X-Duration-Nodule-Ms')} ms")
    print(f"  total           : {r.headers.get('X-Duration-Total-Ms')} ms")

    # Sanity: load the returned .npz to confirm it's well-formed
    out = np.load(args.out)
    print(f"  output shape    : {out['data'].shape}  dtype={out['data'].dtype}  "
          f"nnz={int(out['data'].sum()):,}")


if __name__ == "__main__":
    main()
