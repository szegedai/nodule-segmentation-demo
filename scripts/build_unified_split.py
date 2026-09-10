"""
Build a deterministic, patient-grouped, dataset-stratified train/val/test
split for the unified nodule dataset.

Reads nodule_catalog.csv (one row per nodule), groups by patient_id within
each dataset, and shuffles patient lists with a fixed seed before slicing
70/15/15 (defaults). All series of a given patient land in the same split,
so the same person never appears in both train and test.

Output JSON:
    {
      "meta": {"seed": 42, "ratios": {...}, "datasets": {...}},
      "train": ["unified/<series_uid>.npz", ...],
      "val":   [...],
      "test":  [...]
    }

Usage:
    python scripts/build_unified_split.py \
        --catalog /project/AINoduleSeg/radiology/data/processed/unified/nodule_catalog.csv \
        --name    unified \
        --seed    42 \
        --val     0.15 \
        --test    0.15 \
        --out     data/splits/unified.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", required=True)
    p.add_argument("--name", default="unified",
                   help="dataset name to prefix in JSON keys (e.g. 'unified/<series>.npz')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--out", required=True)
    return p.parse_args()


def split_patients(patients: list[str], val_frac: float, test_frac: float,
                   rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
    """Shuffle the patient list and slice into train/val/test."""
    arr = np.array(patients, dtype=object)
    rng.shuffle(arr)
    n = len(arr)
    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val  = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
    if n_val + n_test >= n:
        sys.exit(f"val+test ({n_val + n_test}) >= patients ({n})")
    test_p = arr[:n_test].tolist()
    val_p  = arr[n_test:n_test + n_val].tolist()
    train_p = arr[n_test + n_val:].tolist()
    return train_p, val_p, test_p


def main():
    args = parse_args()
    cat = pd.read_csv(args.catalog)
    needed = {"patient_id", "series_uid", "dataset"}
    if missing := needed - set(cat.columns):
        sys.exit(f"catalog missing columns: {missing}")

    rng = np.random.default_rng(args.seed)

    # Datasets like nlst_ai and nlst_radiologist annotate the same NLST patients
    # (numeric patient_ids collide across them). For split-grouping we map them
    # to a single cohort so the same NLST patient never ends up in two splits.
    # Prefixed ID schemes (LIDC-IDRI-..., LUNG1-...) don't collide so they
    # stand alone.
    COHORT_NORM = {
        "nlst_ai": "nlst",
        "nlst_radiologist": "nlst",
    }
    def cohort(ds: str) -> str:
        return COHORT_NORM.get(ds, ds)

    # Group: (cohort, patient_id) -> set of series_uids
    pat_series: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ds, pid, series in zip(cat["dataset"], cat["patient_id"], cat["series_uid"]):
        pat_series[(cohort(str(ds)), str(pid))].add(str(series))

    train_keys: list[str] = []
    val_keys: list[str] = []
    test_keys: list[str] = []
    per_cohort_stats: dict[str, dict] = {}

    by_cohort: dict[str, list[str]] = defaultdict(list)
    for (coh, pid) in pat_series:
        by_cohort[coh].append(pid)

    for coh in sorted(by_cohort):
        patients = sorted(by_cohort[coh])
        tr_p, vl_p, te_p = split_patients(patients, args.val, args.test, rng)

        tr_series = sorted({s for p in tr_p for s in pat_series[(coh, p)]})
        vl_series = sorted({s for p in vl_p for s in pat_series[(coh, p)]})
        te_series = sorted({s for p in te_p for s in pat_series[(coh, p)]})

        train_keys += [f"{args.name}/{s}.npz" for s in tr_series]
        val_keys   += [f"{args.name}/{s}.npz" for s in vl_series]
        test_keys  += [f"{args.name}/{s}.npz" for s in te_series]

        per_cohort_stats[coh] = {
            "n_patients": {"train": len(tr_p), "val": len(vl_p), "test": len(te_p)},
            "n_series":   {"train": len(tr_series), "val": len(vl_series), "test": len(te_series)},
        }
    # Backward-compat alias for the stats key name written to JSON.
    per_dataset_stats = per_cohort_stats

    # Sanity: no overlap
    sets = {"train": set(train_keys), "val": set(val_keys), "test": set(test_keys)}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ov = sets[a] & sets[b]
        if ov:
            sys.exit(f"split overlap between {a} and {b}: {len(ov)} series")

    out = {
        "meta": {
            "seed": args.seed,
            "ratios": {"val": args.val, "test": args.test,
                       "train": round(1.0 - args.val - args.test, 4)},
            "catalog": str(args.catalog),
            "dataset_name": args.name,
            "totals": {
                "train": len(train_keys),
                "val":   len(val_keys),
                "test":  len(test_keys),
            },
            "per_dataset": per_dataset_stats,
        },
        "train": sorted(train_keys),
        "val":   sorted(val_keys),
        "test":  sorted(test_keys),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("Per-dataset patient counts:")
    for ds, stats in per_dataset_stats.items():
        p = stats["n_patients"]; s = stats["n_series"]
        print(f"  {ds:20s}  patients tr/vl/te = {p['train']:5d}/{p['val']:4d}/{p['test']:4d}"
              f"   series tr/vl/te = {s['train']:5d}/{s['val']:4d}/{s['test']:4d}")
    print()
    print(f"Total series: train={len(train_keys)}  val={len(val_keys)}  test={len(test_keys)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
