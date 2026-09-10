"""Phase 3 model unit tests: mamba, transformer, fusion, baseline, factory."""
import copy

import pytest
import torch

from src.model.mamba import MambaBlock, selective_scan
from src.model.fusion import FusionModel, TransformerBlock
from src.model.baseline import BaselineCNN
from src.model.factory import create_model, model_version, MODEL_KINDS

CFG = {
    "model": {
        "kind": "mamba_transformer",
        "hsi": {
            "stem_channels": 8,
            "token_grid": 8,
            "d_model": 8,
            "d_state": 4,
            "d_conv": 3,
            "expand": 2,
            "n_blocks": 1,
            "n_heads": 2,
            "ff_scale": 2,
            "dropout": 0.0,
        },
        "dem": {"channels": [4, 8], "dropout": 0.0},
        "head": {"hidden": 8},
    },
    "data": {"pca_components": 3},
}


def test_selective_scan_shapes():
    B, L, D, N = 2, 8, 4, 2
    x = torch.randn(B, L, D)
    delta = torch.rand(B, L, D) * 0.1
    A = -torch.exp(torch.rand(D, N))
    b, c = torch.randn(B, L, N), torch.randn(B, L, N)
    y = selective_scan(x, delta, A, b, c, torch.ones(D))
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()


def test_mamba_block_forward_backward():
    m = MambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    x = torch.randn(2, 16, 8)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_transformer_block_shape():
    t = TransformerBlock(d=8, n_heads=2, ff_scale=2)
    x = torch.randn(2, 16, 8)
    assert t(x).shape == x.shape


def test_fusion_forward():
    m = FusionModel(3, 1, 3, CFG["model"]["hsi"], CFG["model"]["dem"], CFG["model"]["head"])
    logits = m(torch.randn(4, 3, 32, 32), torch.randn(4, 1, 32, 32))
    assert logits.shape == (4, 3)
    with torch.no_grad():
        assert torch.softmax(logits, dim=1).sum(dim=1).allclose(torch.ones(4))


def test_baseline_forward():
    m = BaselineCNN(in_channels=4, num_classes=3, hidden=8)
    logits = m(torch.randn(2, 3, 32, 32), torch.randn(2, 1, 32, 32))
    assert logits.shape == (2, 3)


def test_fusion_stem_grid_mismatch_raises():
    cfg = copy.deepcopy(CFG)
    cfg["model"]["hsi"]["token_grid"] = 12  # stem stride 2 -> 16x16 grid, mismatch
    m = FusionModel(3, 1, 3, cfg["model"]["hsi"], cfg["model"]["dem"], cfg["model"]["head"])
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 32, 32), torch.randn(1, 1, 32, 32))


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_factory_builds_all_kinds(kind):
    cfg = copy.deepcopy(CFG)
    cfg["model"]["kind"] = kind
    m = create_model(cfg)
    assert m(torch.randn(1, 3, 32, 32), torch.randn(1, 1, 32, 32)).shape == (1, 3)
    assert model_version(cfg) == f"{kind}-v1"