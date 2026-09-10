"""Model factory: builds the architecture selected in the config.

Kind strings: mamba_transformer (default) and baseline_cnn. Both consume the
channels-first Phase 2 tensor shapes (X_hsi, X_dem) and emit 3-class logits.
"""
from __future__ import annotations

from .fusion import FusionModel
from .baseline import BaselineCNN

MODEL_KINDS = ("mamba_transformer", "baseline_cnn")

VERSION = "v1"


def model_version(cfg: dict) -> str:
    return f"{cfg['model']['kind']}-{VERSION}"


def create_model(cfg: dict):
    """Build a model from the full (phase3) config dict."""
    kind = cfg["model"].get("kind", "mamba_transformer")
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")
    num_classes = len(cfg.get("class_names", ["None", "Moderate", "Severe Collapse"]))

    if kind == "mamba_transformer":
        return FusionModel(
            hsi_channels=cfg["data"]["pca_components"],
            dem_channels=1,
            num_classes=num_classes,
            hsi_cfg=cfg["model"]["hsi"],
            dem_cfg=cfg["model"]["dem"],
            head_cfg=cfg["model"]["head"],
        )

    return BaselineCNN(
        in_channels=cfg["data"]["pca_components"] + 1,
        num_classes=num_classes,
        hidden=cfg["model"]["head"]["hidden"],
    )


def create_model_with_channels(cfg: dict, pca_components: int):
    """Same as create_model but with an explicit PCA channel count (pre-fit)."""
    import copy

    cfg = copy.deepcopy(cfg)
    cfg["data"]["pca_components"] = pca_components
    return create_model(cfg)


__all__ = ["create_model", "create_model_with_channels", "model_version", "MODEL_KINDS"]