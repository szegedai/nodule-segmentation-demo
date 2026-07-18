"""Transforms per task.

Nodule (3D, 2-class softmax):
    Val   : resize CT + label to `target_size` (default 256³).
    Train : resize + aggressive spatial + intensity augmentation. The crop
            is already lung-bounded, so 3D spatial augs can be stronger.

ROI (2D):
    Val   : just EnsureType — the dataset already resizes each slice to
            `target_size` (default 256²) at load time.
    Train : same. Matches the augmentation used to train the bundled ROI
            checkpoint (checkpoints/roi/best.pth).

Joint (3D, 3-class softmax):
    Same shape recipe as nodule, but the label is a 3-class integer
    tensor (`int64`) so it needs a distinct EnsureTyped dtype.
"""

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Resized,
)

KEYS = ["image", "label"]
_DEFAULT_NODULE_SIZE = (256, 256, 256)


def _nodule_target(preproc_cfg):
    return tuple((preproc_cfg or {}).get("target_size", list(_DEFAULT_NODULE_SIZE)))


def _nodule_resize(target_size):
    return Resized(keys=KEYS, spatial_size=target_size, mode=["trilinear", "nearest"])


# ── nodule (3D) ──────────────────────────────────────────────────────────────

def get_nodule_val_transforms(preproc_cfg=None):
    return Compose([
        _nodule_resize(_nodule_target(preproc_cfg)),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


def get_nodule_train_transforms(preproc_cfg=None):
    t = _nodule_target(preproc_cfg)
    return Compose([
        _nodule_resize(t),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 2)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandRotated(keys=KEYS, range_x=0.3, range_y=0.3, range_z=0.3,
                    prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
        RandZoomd(keys=KEYS, prob=0.4, min_zoom=0.85, max_zoom=1.15,
                  mode=["trilinear", "nearest"], padding_mode="constant", keep_size=True),
        RandScaleIntensityd(keys=["image"], factors=0.2,        prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.15,       prob=0.5),
        RandGaussianNoised( keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5), retain_stats=True),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── ROI (2D) ─────────────────────────────────────────────────────────────────
# The Roi2DDataset already resizes each slice in __getitem__, so the transform
# just needs to ensure tensor+dtype. Matches how the bundled ROI checkpoint
# was trained.

def get_roi_val_transforms(preproc_cfg=None):
    return Compose([
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


def get_roi_train_transforms(preproc_cfg=None):
    return Compose([
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── joint (3D, 3-class) ──────────────────────────────────────────────────────
# Same shape recipe as nodule; only the label dtype differs (int64 for the
# 3-class integer label instead of float32 for the binary nodule mask).

def get_joint_val_transforms(preproc_cfg=None):
    return Compose([
        _nodule_resize(_nodule_target(preproc_cfg)),
        EnsureTyped(keys=["image"], dtype="float32", track_meta=False),
        EnsureTyped(keys=["label"], dtype="int64",   track_meta=False),
    ])


def get_joint_train_transforms(preproc_cfg=None):
    t = _nodule_target(preproc_cfg)
    return Compose([
        _nodule_resize(t),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 2)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandRotated(keys=KEYS, range_x=0.3, range_y=0.3, range_z=0.3,
                    prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
        RandZoomd(keys=KEYS, prob=0.4, min_zoom=0.85, max_zoom=1.15,
                  mode=["trilinear", "nearest"], padding_mode="constant", keep_size=True),
        RandScaleIntensityd(keys=["image"], factors=0.2,        prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.15,       prob=0.5),
        RandGaussianNoised( keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5), retain_stats=True),
        EnsureTyped(keys=["image"], dtype="float32", track_meta=False),
        EnsureTyped(keys=["label"], dtype="int64",   track_meta=False),
    ])


# ── task-aware factory ───────────────────────────────────────────────────────

def build_transforms(task, cfg):
    preproc = cfg.get("preprocessing", {}) or {}
    if task == "nodule":
        return get_nodule_train_transforms(preproc), get_nodule_val_transforms(preproc)
    if task == "roi":
        return get_roi_train_transforms(preproc), get_roi_val_transforms(preproc)
    if task == "joint":
        return get_joint_train_transforms(preproc), get_joint_val_transforms(preproc)
    raise ValueError(f"unknown task: {task!r}")


# ── legacy aliases (so train.py doesn't need to know task at import time) ────

get_train_transforms = get_nodule_train_transforms
get_val_transforms   = get_nodule_val_transforms
