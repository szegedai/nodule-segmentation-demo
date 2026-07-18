"""Datasets for the three-task pipeline.

Nodule3DDataset (== NoduleFineCropDataset)   [task: nodule]
    Loads a CT volume + nodule mask from the unified corpus, crops both to
    the precomputed per-series lung bbox, then hands the crop to the
    transform pipeline (which resizes to 256³).

Roi2DDataset                                 [task: roi]
    One sample per axial slice. Walks ct_2d/ once at construction and
    keeps every '<series_uid>_<NNNN>.npz' whose series UID is in the split.

JointFullVolumeDataset                       [task: joint]
    Loads a CT volume + lung mask + nodule mask WITHOUT bbox cropping. The
    transform pipeline resizes the whole volume to 256³. The lung and
    nodule masks are combined into a 3-class integer label
    {0=background, 1=lung, 2=nodule} — nodule takes precedence over lung.

All three use series-UID basenames; the shared split JSON is a dict with
"train"/"val"/"test" lists of "unified/<uid>.npz" strings.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── 3D nodule segmentation ────────────────────────────────────────────────────

class NoduleFineCropDataset(Dataset):
    def __init__(self, dataset_configs, bbox_json_path, transform=None, filenames=None):
        """
        Parameters
        ----------
        dataset_configs : list of {"name", "ct_dir", "seg_dir"} dicts
        bbox_json_path  : path to the JSON of per-series lung bboxes
        transform       : callable applied to {"image", "label"} dict
        filenames       : list of "<name>/<uid>.npz" keys to include (from split)
        """
        with open(bbox_json_path) as f:
            self.bboxes = json.load(f)
        wanted = set(filenames) if filenames is not None else None
        self.transform = transform
        self.samples = []
        for ds in dataset_configs:
            name    = ds["name"]
            ct_dir  = ds["ct_dir"]
            seg_dir = ds["seg_dir"]
            for fname in sorted(f for f in os.listdir(ct_dir) if f.endswith(".npz")):
                key = f"{name}/{fname}"
                if key not in self.bboxes:
                    continue
                if wanted is not None and key not in wanted:
                    continue
                self.samples.append({
                    "ct_path":  os.path.join(ct_dir,  fname),
                    "seg_path": os.path.join(seg_dir, fname),
                    "key":      key,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        ct  = np.load(s["ct_path"])["data"].astype(np.float32)[:1]        # (1,H,W,D)
        seg = np.load(s["seg_path"])["data"].astype(np.float32)[:1]       # (1,H,W,D)
        h0, h1, w0, w1, d0, d1 = self.bboxes[s["key"]]["bbox"]
        sample = {
            "image":    ct [:, h0:h1, w0:w1, d0:d1],
            "label":    seg[:, h0:h1, w0:w1, d0:d1],
            "filename": s["key"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ── 2D ROI segmentation ───────────────────────────────────────────────────────

class Roi2DDataset(Dataset):
    """One sample per axial slice. Walks ct_2d/ once at construction and keeps
    every '<series_uid>_<NNNN>.npz' whose series UID is in `series_uids`.

    Slices are resized to `target_size` here in __getitem__ (no crop-based
    dependency on a bbox) so the loader yields fixed-shape tensors.
    """

    def __init__(self, root, series_uids, target_size=(256, 256), transform=None):
        root = Path(root)
        self.ct_dir  = root / "ct_2d"
        self.roi_dir = root / "roi_sem_seg_2d"
        self.target  = tuple(target_size)
        self.transform = transform
        wanted = set(series_uids)
        # Filename: '<series_uid>_<NNNN>.npz'. Split on the last '_'.
        self.files = [
            p.name for p in sorted(self.ct_dir.glob("*.npz"))
            if p.name.rsplit("_", 1)[0] in wanted
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ct  = np.load(self.ct_dir  / fname)["data"].astype(np.float32)  # (1, H, W)
        roi = np.load(self.roi_dir / fname)["data"].astype(np.float32)  # (1, H, W)
        ct  = _resize2d(ct,  self.target, mode="bilinear")
        roi = _resize2d(roi, self.target, mode="nearest")
        sample = {
            "image":    ct,
            "label":    roi,
            "filename": fname,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def _resize2d(arr: np.ndarray, target: tuple, mode: str) -> np.ndarray:
    t = torch.from_numpy(arr).float().unsqueeze(0)             # (1,1,H,W)
    kw = {"mode": mode}
    if mode == "bilinear":
        kw["align_corners"] = False
    return F.interpolate(t, size=target, **kw)[0].numpy()


# ── 3D joint (lung + nodule) segmentation ────────────────────────────────────

class JointFullVolumeDataset(Dataset):
    """Full-volume 3-class {bg, lung, nodule} dataset — no bbox crop.

    Parameters
    ----------
    dataset_configs : list of {"name", "ct_dir", "lung_dir", "nodule_dir"} dicts
    transform       : callable applied to {"image", "label"} dict
    filenames       : list of "<name>/<uid>.npz" keys to include (from split)
    """

    def __init__(self, dataset_configs, transform=None, filenames=None):
        wanted = set(filenames) if filenames is not None else None
        self.transform = transform
        self.samples = []
        for ds in dataset_configs:
            name       = ds["name"]
            ct_dir     = ds["ct_dir"]
            lung_dir   = ds["lung_dir"]
            nodule_dir = ds["nodule_dir"]
            for fname in sorted(f for f in os.listdir(ct_dir) if f.endswith(".npz")):
                key = f"{name}/{fname}"
                if wanted is not None and key not in wanted:
                    continue
                lung_p   = os.path.join(lung_dir,   fname)
                nodule_p = os.path.join(nodule_dir, fname)
                if not (os.path.exists(lung_p) and os.path.exists(nodule_p)):
                    continue
                self.samples.append({
                    "ct_path":     os.path.join(ct_dir, fname),
                    "lung_path":   lung_p,
                    "nodule_path": nodule_p,
                    "key":         key,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        ct     = np.load(s["ct_path"])["data"].astype(np.float32)[:1]      # (1, H, W, D)
        lung   = np.load(s["lung_path"])["data"].astype(np.uint8)[:1]      # (1, H, W, D)
        nodule = np.load(s["nodule_path"])["data"].astype(np.uint8)[:1]    # (1, H, W, D)

        # 3-class integer label: nodule > lung > background
        label = np.zeros_like(ct, dtype=np.int64)
        label[lung   > 0] = 1
        label[nodule > 0] = 2

        sample = {"image": ct, "label": label, "filename": s["key"]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ── shared split loading ──────────────────────────────────────────────────────

def load_split(split_json):
    with open(split_json) as f:
        return json.load(f)


def series_uids_from_split(keys):
    """Turn 'unified/<uid>.npz' strings into bare series UIDs (drop dir + ext)."""
    return [Path(k).stem for k in keys]


def build_splits_for_nodule(cfg):
    """Nodule task: return (train_keys, val_keys) intersected with bbox JSON."""
    with open(cfg["data"]["bbox_json"]) as f:
        bboxes = json.load(f)
    split = load_split(cfg["data"]["split_json"])
    train_keys = [k for k in split["train"] if k in bboxes]
    val_keys   = [k for k in split["val"]   if k in bboxes]
    return train_keys, val_keys


# ── task-aware factory ────────────────────────────────────────────────────────

def build_datasets(task, cfg, train_transform, val_transform):
    """Return (train_ds, val_ds) for the given task ('roi' or 'nodule')."""
    if task == "nodule":
        train_keys, val_keys = build_splits_for_nodule(cfg)
        train_ds = NoduleFineCropDataset(cfg["data"]["datasets"], cfg["data"]["bbox_json"],
                                         transform=train_transform, filenames=train_keys)
        val_ds   = NoduleFineCropDataset(cfg["data"]["datasets"], cfg["data"]["bbox_json"],
                                         transform=val_transform,   filenames=val_keys)
        return train_ds, val_ds

    if task == "roi":
        split = load_split(cfg["data"]["split_json"])
        train_uids = series_uids_from_split(split["train"])
        val_uids   = series_uids_from_split(split["val"])
        root       = cfg["data"]["data_root"]
        target     = cfg["preprocessing"]["target_size"]
        train_ds = Roi2DDataset(root, train_uids, target_size=target, transform=train_transform)
        val_ds   = Roi2DDataset(root, val_uids,   target_size=target, transform=val_transform)
        return train_ds, val_ds

    if task == "joint":
        split = load_split(cfg["data"]["split_json"])
        train_ds = JointFullVolumeDataset(cfg["data"]["datasets"],
                                          transform=train_transform,
                                          filenames=split["train"])
        val_ds   = JointFullVolumeDataset(cfg["data"]["datasets"],
                                          transform=val_transform,
                                          filenames=split["val"])
        return train_ds, val_ds

    raise ValueError(f"unknown task: {task!r}")
