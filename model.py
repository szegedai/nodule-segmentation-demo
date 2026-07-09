"""
Stage 2 Fine — Nodule Model
=============================
Architecture-agnostic builder. Dispatches on cfg["name"] (defaults to
"segresnet" so configs without an explicit name behave exactly as before).

Supported names:
  segresnet  — MONAI SegResNet (CNN, current production architecture).
  swinunetr  — MONAI SwinUNETR (Swin Transformer + U-Net hybrid).
  dynunet    — MONAI DynUNet   (nnU-Net-style self-configuring U-Net).

Input : (B, 1, *target_size)
Output: (B, out_channels, *target_size)
"""

import torch.nn as nn


def _build_segresnet(cfg: dict) -> nn.Module:
    from monai.networks.nets import SegResNet
    return SegResNet(
        spatial_dims = cfg.get("spatial_dims", 3),
        in_channels  = cfg.get("in_channels",  1),
        out_channels = cfg.get("out_channels", 2),
        init_filters = cfg.get("init_filters", 16),
        blocks_down  = tuple(cfg.get("blocks_down", [1, 2, 2, 4, 4])),
        blocks_up    = tuple(cfg.get("blocks_up",   [1, 1, 1, 1])),
        dropout_prob = cfg.get("dropout_prob", 0.1),
    )


def _build_swinunetr(cfg: dict) -> nn.Module:
    from monai.networks.nets import SwinUNETR
    img_size = tuple(cfg.get("img_size", [256, 256, 256]))
    return SwinUNETR(
        img_size      = img_size,
        in_channels   = cfg.get("in_channels",  1),
        out_channels  = cfg.get("out_channels", 2),
        feature_size  = cfg.get("feature_size", 48),
        depths        = tuple(cfg.get("depths",       [2, 2, 2, 2])),
        num_heads     = tuple(cfg.get("num_heads",    [3, 6, 12, 24])),
        drop_rate     = cfg.get("drop_rate",      0.0),
        attn_drop_rate= cfg.get("attn_drop_rate", 0.0),
        dropout_path_rate = cfg.get("dropout_path_rate", 0.0),
        use_checkpoint= cfg.get("use_checkpoint", False),
        spatial_dims  = cfg.get("spatial_dims", 3),
    )


def _build_dynunet(cfg: dict) -> nn.Module:
    from monai.networks.nets import DynUNet
    # nnU-Net-style: each downsampling halves spatial dims. With 256³ input,
    # 6 levels gives a 4³ bottleneck which is a sensible default.
    default_strides = [[1, 1, 1]] + [[2, 2, 2]] * 5
    default_kernel  = [[3, 3, 3]] * len(default_strides)
    strides = cfg.get("strides", default_strides)
    kernel  = cfg.get("kernel_size", default_kernel)
    upsample_kernel = cfg.get("upsample_kernel_size", strides[1:])
    return DynUNet(
        spatial_dims         = cfg.get("spatial_dims", 3),
        in_channels          = cfg.get("in_channels",  1),
        out_channels         = cfg.get("out_channels", 2),
        kernel_size          = kernel,
        strides              = strides,
        upsample_kernel_size = upsample_kernel,
        norm_name            = cfg.get("norm_name", "instance"),
        deep_supervision     = cfg.get("deep_supervision", False),
        deep_supr_num        = cfg.get("deep_supr_num", 1),
        dropout              = cfg.get("dropout_prob", 0.0),
    )


_BUILDERS = {
    "segresnet": _build_segresnet,
    "swinunetr": _build_swinunetr,
    "dynunet":   _build_dynunet,
}


def build_model(cfg: dict) -> nn.Module:
    name = str(cfg.get("name", "segresnet")).lower()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model.name={name!r}; "
                         f"supported: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    import torch, yaml

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/fine_nodule.yaml"
    cfg = yaml.safe_load(open(cfg_path))["model"]
    model = build_model(cfg)
    print(f"name: {cfg.get('name', 'segresnet')}")
    print(f"params: {count_parameters(model):,}")

    x = torch.randn(1, 1, 256, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input : {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    print("PASSED")
