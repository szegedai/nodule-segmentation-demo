"""Losses for the three-task pipeline.

Nodule (2-class softmax): FocalTverskyCELoss — recall-biased Tversky + a
light weighted CE. This is the loss v6/v7/v9 were trained with.

ROI    (1-channel sigmoid): plain MONAI DiceLoss(sigmoid=True,
squared_pred=True). Lung foreground is a majority class (~30–50 % of a
mid-thorax slice), so the sparse-class machinery Focal Tversky adds is
overkill; plain Dice is what the bundled ROI checkpoint was trained with.

Joint  (3-class softmax): MulticlassFocalTverskyCELoss — direct
generalisation of the binary version to K classes. The Tversky term is
computed per non-background class and averaged; the CE term uses a
per-class weight vector (default `[1.0, 1.0, 100.0]` for
`[bg, lung, nodule]`, mirroring the binary loss's nodule up-weighting).
"""

import torch
import torch.nn.functional as F


class FocalTverskyCELoss(torch.nn.Module):
    """
    Focal Tversky Loss (recall-biased) + light weighted CE for binary
    (background, nodule) segmentation.

        TI  = (TP + ε) / (TP + α·FP + β·FN + ε)
        FTL = (1 − TI)^γ

    α < β  → penalise false negatives more than false positives (recall-focused)
    γ > 1  → focal exponent: concentrate gradient on samples with low TI

    A small weighted CE term is added to
      (a) provide a non-vanishing per-voxel gradient even when the network
          predicts all-background (the classic Dice/FTL failure mode),
      (b) counter FTL's tendency — when β >> α — to predict nodule everywhere.

    Keep `lambda_ce` small (~0.1) and the CE weight on the nodule class large
    (order of n_background / n_nodule) so the CE contribution is a gentle push,
    not a dominating term.
    """

    def __init__(self,
                 alpha: float = 0.3,
                 beta:  float = 0.7,
                 gamma: float = 2.0,
                 smooth: float = 1e-6,
                 prob_clamp: float = 1e-4,
                 lambda_ce: float = 0.1,
                 ce_bg_weight:     float = 1.0,
                 ce_nodule_weight: float = 500.0):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.gamma      = gamma
        self.smooth     = smooth
        self.prob_clamp = prob_clamp
        self.lambda_ce  = lambda_ce

        self.register_buffer(
            "ce_weights",
            torch.tensor([ce_bg_weight, ce_nodule_weight], dtype=torch.float32),
        )

        self.last_ftl = 0.0
        self.last_ce  = 0.0

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, 2, H, W, D)  unnormalised
        labels : (B, 1, H, W, D) or (B, H, W, D) long/int {0, 1}
        """
        logits_f32 = logits.float()
        probs      = torch.softmax(logits_f32, dim=1).clamp(min=self.prob_clamp)
        prob_nod   = probs[:, 1:2]

        if labels.ndim == 5:
            labels_long = labels[:, 0].long()
        else:
            labels_long = labels.long()
        nodule_mask = (labels_long.unsqueeze(1) == 1).float()

        spatial = (1, 2, 3, 4)
        tp = (prob_nod * nodule_mask).sum(spatial)
        fp = (prob_nod * (1.0 - nodule_mask)).sum(spatial)
        fn = ((1.0 - prob_nod) * nodule_mask).sum(spatial)
        ti  = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        ftl = ((1.0 - ti) ** self.gamma).mean()

        self.last_ftl = ftl.detach().item()
        total = ftl

        if self.lambda_ce > 0:
            ce = F.cross_entropy(logits_f32, labels_long,
                                 weight=self.ce_weights.to(logits.device))
            self.last_ce = ce.detach().item()
            total = total + self.lambda_ce * ce
        else:
            self.last_ce = 0.0

        return total


class MulticlassFocalTverskyCELoss(torch.nn.Module):
    """K-class generalisation of FocalTverskyCELoss.

        TI_k  = (TP_k + ε) / (TP_k + α·FP_k + β·FN_k + ε)
        FTL_k = (1 − TI_k)^γ
        FTL   = mean_{k ∈ non-bg classes} FTL_k

    plus a `lambda_ce · CE` term with per-class weights.
    """

    def __init__(self,
                 num_classes: int = 3,
                 alpha: float = 0.3,
                 beta:  float = 0.7,
                 gamma: float = 2.0,
                 smooth: float = 1e-6,
                 prob_clamp: float = 1e-4,
                 lambda_ce: float = 0.1,
                 class_weights=None,
                 include_background_in_ftl: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.alpha       = alpha
        self.beta        = beta
        self.gamma       = gamma
        self.smooth      = smooth
        self.prob_clamp  = prob_clamp
        self.lambda_ce   = lambda_ce
        self.include_bg  = include_background_in_ftl

        if class_weights is None:
            class_weights = [1.0] * num_classes
        if len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(class_weights)} entries, expected {num_classes}")
        self.register_buffer(
            "ce_weights", torch.tensor(class_weights, dtype=torch.float32),
        )

        self.last_ftl = 0.0
        self.last_ce  = 0.0
        self.last_ftl_per_class = [0.0] * num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, C, H, W, D) unnormalised
        labels : (B, 1, H, W, D) or (B, H, W, D)  integer in {0, …, C-1}
        """
        logits_f32 = logits.float()
        probs = torch.softmax(logits_f32, dim=1).clamp(min=self.prob_clamp)

        if labels.ndim == 5:
            labels_long = labels[:, 0].long()
        else:
            labels_long = labels.long()

        spatial = (1, 2, 3, 4)
        classes = list(range(0 if self.include_bg else 1, self.num_classes))
        ftl_terms = []
        for c in classes:
            gt_c   = (labels_long.unsqueeze(1) == c).float()
            prob_c = probs[:, c:c+1]
            tp = (prob_c * gt_c).sum(spatial)
            fp = (prob_c * (1.0 - gt_c)).sum(spatial)
            fn = ((1.0 - prob_c) * gt_c).sum(spatial)
            ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            ftl_c = ((1.0 - ti) ** self.gamma).mean()
            ftl_terms.append(ftl_c)
            self.last_ftl_per_class[c] = ftl_c.detach().item()

        ftl = torch.stack(ftl_terms).mean()
        self.last_ftl = ftl.detach().item()
        total = ftl

        if self.lambda_ce > 0:
            ce = F.cross_entropy(logits_f32, labels_long,
                                 weight=self.ce_weights.to(logits.device))
            self.last_ce = ce.detach().item()
            total = total + self.lambda_ce * ce
        else:
            self.last_ce = 0.0

        return total


def build_loss(cfg, task):
    """Return the appropriate loss for `task`.

    task="nodule" → FocalTverskyCELoss with cfg["loss"] hyperparams
    task="roi"    → DiceLoss(sigmoid=True, squared_pred=True)
    task="joint"  → MulticlassFocalTverskyCELoss with cfg["loss"] hyperparams
    """
    if task == "nodule":
        loss_cfg = cfg.get("loss", {}) or {}
        return FocalTverskyCELoss(
            alpha            = loss_cfg.get("alpha",            0.3),
            beta             = loss_cfg.get("beta",             0.7),
            gamma            = loss_cfg.get("gamma",            2.0),
            smooth           = loss_cfg.get("smooth",           1e-6),
            prob_clamp       = loss_cfg.get("prob_clamp",       1e-4),
            lambda_ce        = loss_cfg.get("lambda_ce",        0.1),
            ce_bg_weight     = loss_cfg.get("ce_bg_weight",     1.0),
            ce_nodule_weight = loss_cfg.get("ce_nodule_weight", 500.0),
        )
    if task == "roi":
        from monai.losses import DiceLoss
        return DiceLoss(sigmoid=True, squared_pred=True)
    if task == "joint":
        loss_cfg = cfg.get("loss", {}) or {}
        return MulticlassFocalTverskyCELoss(
            num_classes   = int(cfg["model"]["out_channels"]),
            alpha         = loss_cfg.get("alpha",         0.3),
            beta          = loss_cfg.get("beta",          0.7),
            gamma         = loss_cfg.get("gamma",         2.0),
            smooth        = loss_cfg.get("smooth",        1e-6),
            prob_clamp    = loss_cfg.get("prob_clamp",    1e-4),
            lambda_ce     = loss_cfg.get("lambda_ce",     0.1),
            class_weights = loss_cfg.get("class_weights", None),
        )
    raise ValueError(f"unknown task: {task!r}")
