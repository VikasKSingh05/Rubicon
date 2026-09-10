"""Models for Rubicon Phase 3: Mamba-Transformer fusion + baseline CNN."""
from .mamba import MambaBlock, selective_scan
from .fusion import FusionModel, HsiBranch, DemBranch, TransformerBlock
from .baseline import BaselineCNN, ResidualBlock
from .factory import create_model, create_model_with_channels, model_version

__all__ = [
    "MambaBlock",
    "selective_scan",
    "FusionModel",
    "HsiBranch",
    "DemBranch",
    "TransformerBlock",
    "BaselineCNN",
    "ResidualBlock",
    "create_model",
    "create_model_with_channels",
    "model_version",
]