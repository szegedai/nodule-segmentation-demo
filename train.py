"""Three-task training loop — trains ROI (2D lung), nodule (3D cropped
bbox, 2-class), or joint end-to-end (3D full volume, 3-class).

The task is chosen by the `task:` field in the YAML config
(roi | nodule | joint).

Nodule pipeline (v6, v7, v9):
    - 3D SegResNet / DynUNet (out_channels=2, softmax)
    - FocalTverskyCE loss
    - Val metric = softmax→argmax→one-hot Dice; also nodule sens/prec/detection

ROI pipeline (roi.yaml, roi_swin.yaml):
    - 2D SegResNet / SwinUNETR (out_channels=1, sigmoid)
    - DiceLoss(sigmoid=True, squared_pred=True)
    - Val metric = sigmoid→threshold Dice; also foreground sens/prec

Joint pipeline (joint_*.yaml):
    - 3D SegResNet / DynUNet (out_channels=3, softmax) — no bbox crop
    - MulticlassFocalTverskyCELoss
    - Val metric = softmax→argmax→one-hot Dice (mean of lung + nodule); also
      nodule sens/prec/detection

Usage:
    export DATA_ROOT=/path/to/unified
    python train.py --config configs/v7.yaml           # first run
    python train.py --config configs/joint_dynunet_ex_lidc.yaml
    python train.py --config configs/v7.yaml --resume  # resume from last.pth
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils import set_determinism
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataset    import build_datasets
from loss       import build_loss
from model      import build_model, count_parameters
from transforms import build_transforms


# ── args + config ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Resume from <checkpoint_dir>/last.pth if present.")
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load YAML, expanding ${DATA_ROOT} (and any other env var) in strings."""
    if not os.environ.get("DATA_ROOT"):
        raise SystemExit("DATA_ROOT is not set. Run:  export DATA_ROOT=/path/to/unified")
    with open(path) as f:
        text = os.path.expandvars(f.read())
    return yaml.safe_load(text)


# ── checkpoints ──────────────────────────────────────────────────────────────

def save_ckpt(path, epoch, model, optim, sched, best_dice, cfg):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "val_dice":             best_dice,
        "config":               cfg,
    }, path)


def load_ckpt(path, model, optim, sched, device):
    """Load a checkpoint. Tolerates both the current format (model_state_dict /
    optimizer_state_dict / scheduler_state_dict / val_dice) and the older
    training_demo format (model / optim / best_dice, no scheduler)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict",     ck.get("model")))
    optim.load_state_dict(ck.get("optimizer_state_dict", ck.get("optim")))
    if "scheduler_state_dict" in ck:
        sched.load_state_dict(ck["scheduler_state_dict"])
    return ck.get("epoch", 0), ck.get("val_dice", ck.get("best_dice", 0.0))


# ── task-aware post-processing (train + val) ─────────────────────────────────

class NodulePostproc:
    """2-class softmax → argmax → one-hot. For DiceMetric(include_background=False)."""

    def __init__(self):
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)

    def label_to_device(self, labels, device):
        return labels.long().to(device)

    def pred_pos(self, logits):
        """(pred, label)-shaped foreground channel of one-hot preds."""
        preds_oh = torch.stack([self.post_pred(logits[i]) for i in range(len(logits))])
        return preds_oh                                          # (B, 2, ...)

    def metric_inputs(self, logits, labels):
        preds_oh  = torch.stack([self.post_pred(logits[i])  for i in range(len(logits))])
        labels_oh = torch.stack([self.post_label(labels[i]) for i in range(len(labels))])
        return preds_oh, labels_oh, preds_oh[:, 1], labels_oh[:, 1]


class RoiPostproc:
    """1-channel sigmoid → threshold at 0.5. For DiceMetric(include_background=True)."""

    def label_to_device(self, labels, device):
        return labels.to(device).float()

    def metric_inputs(self, logits, labels):
        pred = (torch.sigmoid(logits) > 0.5).float()             # (B, 1, ...)
        # Return the same tensor 4×: whole-onehot for metric + foreground for TP/FP/FN.
        return pred, labels, pred[:, 0], labels[:, 0]


class JointPostproc:
    """3-class softmax → argmax → one-hot. Foreground channel used for TP/FP/FN
    tracking is the *nodule* channel (class 2) — the sparse, clinically
    important class. DiceMetric(include_background=False) will average lung +
    nodule Dice per case."""

    def __init__(self, num_classes: int = 3, foreground_class: int = 2):
        self.num_classes      = num_classes
        self.foreground_class = foreground_class
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

    def label_to_device(self, labels, device):
        return labels.long().to(device)

    def metric_inputs(self, logits, labels):
        preds_oh  = torch.stack([self.post_pred(logits[i])  for i in range(len(logits))])
        labels_oh = torch.stack([self.post_label(labels[i]) for i in range(len(labels))])
        fg = self.foreground_class
        return preds_oh, labels_oh, preds_oh[:, fg], labels_oh[:, fg]


def make_postproc(task, cfg=None):
    if task == "nodule":
        return NodulePostproc()
    if task == "roi":
        return RoiPostproc()
    if task == "joint":
        C = int((cfg or {}).get("model", {}).get("out_channels", 3))
        return JointPostproc(num_classes=C, foreground_class=C - 1)
    raise ValueError(f"unknown task: {task!r}")


def make_dice_metric(task):
    # Nodule: 2-ch onehot, skip background. ROI: 1-ch, count it.
    # Joint : 3-ch onehot, skip background → mean of (lung, nodule) Dice.
    include_bg = (task == "roi")
    return DiceMetric(include_background=include_bg, reduction="mean")


# ── validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def make_predictor(task, model, cfg):
    """Return a callable `predictor(x) -> logits` for validation / inference.

    In `training.mode: resize` this is just the model. In
    `training.mode: sliding_window` (for 3D nodule and joint tasks) it wraps
    the model in MONAI's `sliding_window_inference` — same recipe used at
    final eval time.
    """
    training = cfg.get("training", {}) if cfg else {}
    if training.get("mode", "resize") != "sliding_window" or task == "roi":
        return model
    preproc  = cfg.get("preprocessing", {})
    inf_cfg  = cfg.get("inference",     {})
    roi_size = tuple(preproc.get("patch_size", [128, 128, 128]))
    ovlp     = float(inf_cfg.get("sw_overlap", 0.5))
    swbs     = int(inf_cfg.get("sw_batch",   4))
    def _predict(x):
        return sliding_window_inference(
            inputs=x, roi_size=roi_size, sw_batch_size=swbs,
            predictor=model, overlap=ovlp, mode="gaussian",
        )
    return _predict


def validate(task, model, loader, criterion, device, cfg=None, predictor=None):
    """Return dict of val metrics. Task-shape aware."""
    model.eval()
    forward = predictor if predictor is not None else model
    post = make_postproc(task, cfg=cfg)
    metric = make_dice_metric(task)

    total_loss = 0.0; n_batches = 0
    tp = fp = fn = 0.0
    n_with_pos = n_detected = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = post.label_to_device(batch["label"], device)

        logits = forward(images)
        # Loss expects the same label dtype/shape it was trained with.
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches  += 1

        pred_full, label_full, pred_fg, label_fg = post.metric_inputs(logits, labels)
        metric(y_pred=pred_full, y=label_full)

        tp += float((pred_fg * label_fg).sum())
        fp += float((pred_fg * (1 - label_fg)).sum())
        fn += float(((1 - pred_fg) * label_fg).sum())

        for b in range(pred_fg.shape[0]):
            if float(label_fg[b].sum()) > 0:
                n_with_pos += 1
                if float(pred_fg[b].sum()) > 0:
                    n_detected += 1

    dice = float(metric.aggregate())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    model.train()
    return {
        "loss":       total_loss / max(n_batches, 1),
        "dice":       dice,
        "sens":       sens,
        "prec":       prec,
        "n_detected": n_detected,
        "n_with_pos": n_with_pos,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    task = cfg.get("task", "nodule")
    if task not in ("nodule", "roi", "joint"):
        raise SystemExit(f"config task must be 'nodule', 'roi', or 'joint', got {task!r}")

    set_determinism(seed=cfg["training"]["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    if task == "nodule":
        title = "Stage 2 Fine — Nodule Training (256³ crop, 2-class)"
    elif task == "roi":
        title = "Stage 1 — ROI Training (2D axial slice, binary lung mask)"
    else:
        title = "Stage 3 Joint — Full-Volume Training (256³, 3-class softmax)"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Device: {device}   Epochs: {cfg['training']['epochs']}")
    print(f"  Batch : {cfg['training']['batch_size']}   LR: {cfg['training']['lr']}")
    print(f"{'='*60}\n")

    # ── data ────────────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms(task, cfg)
    train_ds, val_ds = build_datasets(task, cfg, train_tf, val_tf)
    print(f"  Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    bs        = cfg["training"]["batch_size"]
    n_workers = cfg["data"].get("num_workers", 0)
    pin = n_workers > 0 and torch.cuda.is_available()

    # Optional per-epoch subsampling — used for ROI, whose train set is ~475k
    # slices, so a full epoch is impractical.
    samples_per_epoch = cfg["training"].get("samples_per_epoch")
    # list_data_collate flattens `RandCropBy*Labeld`'s list-of-dicts output
    # into a single batch dim in sliding_window mode; it's a no-op for plain
    # dicts (resize mode / ROI).
    if samples_per_epoch:
        train_sampler = RandomSampler(train_ds, replacement=False,
                                      num_samples=int(samples_per_epoch))
        train_loader  = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                                   num_workers=n_workers, pin_memory=pin,
                                   collate_fn=list_data_collate)
        print(f"  Sampler: RandomSampler(num_samples={samples_per_epoch})")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=n_workers, pin_memory=pin,
                                  collate_fn=list_data_collate)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=n_workers, pin_memory=pin,
                            collate_fn=list_data_collate)

    # ── model / loss / optim ────────────────────────────────────────────────
    model = build_model(cfg["model"]).to(device)
    print(f"  Model : {cfg['model'].get('name', 'segresnet')}   "
          f"params={count_parameters(model):,}")

    criterion = build_loss(cfg, task).to(device)
    print(f"  Loss  : {type(criterion).__name__}")

    optim = Adam(model.parameters(),
                 lr=cfg["training"]["lr"],
                 weight_decay=cfg["training"]["weight_decay"])
    sched = CosineAnnealingLR(optim,
                              T_max=cfg["scheduler"]["T_max"],
                              eta_min=cfg["scheduler"]["eta_min"])

    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    print(f"  AMP   : {'on (bfloat16)' if use_amp else 'off'}\n")

    # ── resume ──────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoints"]["dir"])
    log_dir  = Path(cfg["logging"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)

    start_epoch = 0
    best_dice   = 0.0
    if args.resume and (ckpt_dir / "last.pth").exists():
        start_epoch, best_dice = load_ckpt(ckpt_dir / "last.pth", model, optim, sched, device)
        start_epoch += 1
        print(f"  Resumed from {ckpt_dir/'last.pth'}  epoch={start_epoch-1}  best={best_dice:.4f}\n")

    writer       = SummaryWriter(log_dir=str(log_dir / "run"))
    max_epochs   = cfg["training"]["epochs"]
    val_interval = cfg["training"]["val_interval"]
    save_every   = cfg["checkpoints"]["save_every"]

    post = make_postproc(task, cfg=cfg)
    train_metric = make_dice_metric(task)

    # In resize mode this is the model itself; in sliding_window mode it
    # wraps the model in MONAI's sliding_window_inference — same recipe as
    # final eval.
    predictor = make_predictor(task, model, cfg)

    # ── training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0.0
        train_metric.reset()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = post.label_to_device(batch["label"], device)
            optim.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images)
                    loss   = criterion(logits, labels)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if not torch.isfinite(loss):
                print("  [WARNING] non-finite loss — skipping batch", flush=True)
                optim.zero_grad()
                continue

            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                print(f"  [WARNING] non-finite grad_norm={gn.item()} — skipping step", flush=True)
                optim.zero_grad()
                continue
            optim.step()

            total_loss += loss.item()
            with torch.no_grad():
                pred_full, label_full, _, _ = post.metric_inputs(logits.detach(), labels)
                train_metric(y_pred=pred_full, y=label_full)

        train_loss = total_loss / max(len(train_loader), 1)
        train_dice = float(train_metric.aggregate())
        sched.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Dice/train", train_dice, epoch)
        writer.add_scalar("LR",         sched.get_last_lr()[0], epoch)
        lr_now = sched.get_last_lr()[0]
        print(f"┌─ Epoch {epoch+1:4d}/{max_epochs}   lr={lr_now:.2e}", flush=True)
        print(f"│  TRAIN  loss={train_loss:.4f}   dice={train_dice:.4f}", flush=True)

        if (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1:
            m = validate(task, model, val_loader, criterion, device, cfg=cfg,
                         predictor=predictor)
            writer.add_scalar("Loss/val",       m["loss"], epoch)
            writer.add_scalar("Dice/val",       m["dice"], epoch)
            writer.add_scalar("Val/sensitivity", m["sens"] if m["sens"] == m["sens"] else 0.0, epoch)
            writer.add_scalar("Val/precision",   m["prec"] if m["prec"] == m["prec"] else 0.0, epoch)

            is_best = m["dice"] > best_dice
            best_tag = "   ← BEST" if is_best else ""
            _f = lambda v: f"{v:.4f}" if v == v else "n/a"
            det_str = f"   det={m['n_detected']}/{m['n_with_pos']}" if task in ("nodule", "joint") else ""
            print(f"│  VAL    loss={m['loss']:.4f}   dice={m['dice']:.4f}"
                  f"   sens={_f(m['sens'])}   prec={_f(m['prec'])}{det_str}{best_tag}",
                  flush=True)
            if is_best:
                best_dice = m["dice"]
                save_ckpt(ckpt_dir / "best_model.pth", epoch, model, optim, sched, best_dice, cfg)

        print("└─", flush=True)

        if (epoch + 1) % save_every == 0:
            save_ckpt(ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                      epoch, model, optim, sched, best_dice, cfg)
        save_ckpt(ckpt_dir / "last.pth", epoch, model, optim, sched, best_dice, cfg)

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Training complete.  Best {task} Dice: {best_dice:.4f}")
    print(f"  Best model: {ckpt_dir / 'best_model.pth'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
