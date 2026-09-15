"""Transforms per task.

Nodule (3D, 2-class softmax):
    Two modes chosen by `training.mode` in the YAML:
      resize (default)  — resize CT+label to `preprocessing.target_size`
                          (default 256³), then aggressive augmentation.
      sliding_window    — no resize. RandCropByPosNegLabeld draws
                          `training.patches_per_volume` patches per
                          volume, ratio `pos:neg = training.pos_neg_ratio:1`.
                          Val loader returns full-volume; the training
                          loop uses `sliding_window_inference` at eval time.

ROI (2D):
    Val   : just EnsureType — the dataset already resizes each slice to
            `target_size` (default 256²) at load time.
    Train : same. Matches the augmentation used to train the bundled ROI
            checkpoint (checkpoints/roi/best.pth). No SW mode — ROI is 2D
            and already runs per-slice.

Joint (3D, 3-class softmax):
    Same two modes as nodule, but the "sliding_window" crop uses
    RandCropByLabelClassesd with `training.class_sample_ratios`
    (default [0.2, 0.3, 0.5] for bg / lung / nodule) since the label is
    3-class, not binary.
"""

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Resized,
    ResizeWithPadOrCropd,
)

KEYS = ["image", "label"]
_DEFAULT_NODULE_SIZE = (256, 256, 256)
_DEFAULT_PATCH_SIZE  = (128, 128, 128)
_DEFAULT_CLASS_RATIOS = [0.2, 0.3, 0.5]   # bg / lung / nodule


def _mode(training_cfg):
    return (training_cfg or {}).get("mode", "resize")


def _nodule_target(preproc_cfg):
    return tuple((preproc_cfg or {}).get("target_size", list(_DEFAULT_NODULE_SIZE)))


def _patch(preproc_cfg):
    return tuple((preproc_cfg or {}).get("patch_size", list(_DEFAULT_PATCH_SIZE)))


def _nodule_resize(target_size):
    return Resized(keys=KEYS, spatial_size=target_size, mode=["trilinear", "nearest"])


def _rand_pos_neg_crop(patch_size, num_samples, pos, neg):
    return RandCropByPosNegLabeld(
        keys=KEYS, label_key="label",
        spatial_size=patch_size,
        pos=pos, neg=neg, num_samples=num_samples,
        image_key="image", image_threshold=0.0, allow_smaller=True,
    )


def _rand_class_crop(patch_size, num_samples, ratios):
    return RandCropByLabelClassesd(
        keys=KEYS, label_key="label",
        spatial_size=patch_size,
        num_classes=3, num_samples=num_samples,
        ratios=list(ratios),
        image_key="image", image_threshold=0.0, allow_smaller=True,
    )


def _spatial_augs():
    return [
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
    ]


def _intensity_augs():
    return [
        RandScaleIntensityd(keys=["image"], factors=0.2,        prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.15,       prob=0.5),
        RandGaussianNoised( keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5), retain_stats=True),
    ]


# ── nodule (3D) ──────────────────────────────────────────────────────────────

def get_nodule_val_transforms(preproc_cfg=None, training_cfg=None):
    if _mode(training_cfg) == "sliding_window":
        return Compose([EnsureTyped(keys=KEYS, dtype="float32", track_meta=False)])
    return Compose([
        _nodule_resize(_nodule_target(preproc_cfg)),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


def get_nodule_train_transforms(preproc_cfg=None, training_cfg=None):
    tcfg = training_cfg or {}
    if _mode(training_cfg) == "sliding_window":
        return Compose([
            _rand_pos_neg_crop(
                patch_size  = _patch(preproc_cfg),
                num_samples = int(tcfg.get("patches_per_volume", 4)),
                pos         = float(tcfg.get("pos_neg_ratio", 2.0)),
                neg         = 1.0,
            ),
            *_spatial_augs(),
            *_intensity_augs(),
            EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
        ])
    return Compose([
        _nodule_resize(_nodule_target(preproc_cfg)),
        *_spatial_augs(),
        *_intensity_augs(),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── ROI (2D) ─────────────────────────────────────────────────────────────────
# The Roi2DDataset already resizes each slice in __getitem__, so the transform
# just needs to ensure tensor+dtype. Matches how the bundled ROI checkpoint
# was trained.

def get_roi_val_transforms(preproc_cfg=None, training_cfg=None):
    return Compose([
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


def get_roi_train_transforms(preproc_cfg=None, training_cfg=None):
    tcfg = training_cfg or {}
    if _mode(training_cfg) == "sliding_window":
        patch = tuple((preproc_cfg or {}).get("patch_size", [256, 256]))
        return Compose([
            _rand_pos_neg_crop(
                patch_size  = patch,
                num_samples = int(tcfg.get("patches_per_volume", 4)),
                pos         = float(tcfg.get("pos_neg_ratio", 1.0)),
                neg         = 1.0,
            ),
            # some series have slices smaller than the patch (e.g. 254x254);
            # pad them so every patch in the batch is the same size
            ResizeWithPadOrCropd(keys=KEYS, spatial_size=patch),
            EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
        ])
    return Compose([
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── joint (3D, 3-class) ──────────────────────────────────────────────────────

def _joint_typecast():
    return [
        EnsureTyped(keys=["image"], dtype="float32", track_meta=False),
        EnsureTyped(keys=["label"], dtype="int64",   track_meta=False),
    ]


def get_joint_val_transforms(preproc_cfg=None, training_cfg=None):
    if _mode(training_cfg) == "sliding_window":
        return Compose(_joint_typecast())
    return Compose([_nodule_resize(_nodule_target(preproc_cfg)), *_joint_typecast()])


def get_joint_train_transforms(preproc_cfg=None, training_cfg=None):
    tcfg = training_cfg or {}
    if _mode(training_cfg) == "sliding_window":
        return Compose([
            _rand_class_crop(
                patch_size  = _patch(preproc_cfg),
                num_samples = int(tcfg.get("patches_per_volume", 4)),
                ratios      = tcfg.get("class_sample_ratios", _DEFAULT_CLASS_RATIOS),
            ),
            *_spatial_augs(),
            *_intensity_augs(),
            *_joint_typecast(),
        ])
    return Compose([
        _nodule_resize(_nodule_target(preproc_cfg)),
        *_spatial_augs(),
        *_intensity_augs(),
        *_joint_typecast(),
    ])


# ── task-aware factory ───────────────────────────────────────────────────────

def build_transforms(task, cfg):
    preproc  = cfg.get("preprocessing", {}) or {}
    training = cfg.get("training",      {}) or {}
    if task == "nodule":
        return (get_nodule_train_transforms(preproc, training),
                get_nodule_val_transforms(  preproc, training))
    if task == "roi":
        return (get_roi_train_transforms(preproc, training),
                get_roi_val_transforms(  preproc, training))
    if task == "joint":
        return (get_joint_train_transforms(preproc, training),
                get_joint_val_transforms(  preproc, training))
    raise ValueError(f"unknown task: {task!r}")


# ── legacy aliases (so importers that pre-date the factory keep working) ─────

get_train_transforms = get_nodule_train_transforms
get_val_transforms   = get_nodule_val_transforms
