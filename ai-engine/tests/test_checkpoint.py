"""Tests for the shared checkpoint contract (train -> eval -> serve)."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from ai_engine.model.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    DAMAGE_CLASS_NAMES,
    load_checkpoint,
    norm_stats_from_checkpoint,
    save_checkpoint,
    state_dict_hash,
)


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny_model()
    path = tmp_path / "model.pt"
    payload = save_checkpoint(
        path,
        model,
        model_version="test-v1",
        encoder_name="resnet18",
        num_classes=3,
        crop_size=256,
        norm_mean=[0.1, 0.2, 0.3],
        norm_std=[0.4, 0.5, 0.6],
        damage_classes=["background", "a", "b"],
        epoch=2,
        metrics={"xview2_score": 0.5},
        hardware_profile="cpu_fallback",
    )

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["state_dict_hash"] == state_dict_hash(model.state_dict())

    loaded = load_checkpoint(path, map_location="cpu", expected_num_classes=3)
    assert loaded["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert loaded["model_version"] == "test-v1"
    assert loaded["encoder_name"] == "resnet18"
    assert loaded["num_classes"] == 3
    assert loaded["damage_classes"] == ["background", "a", "b"]
    assert loaded["config"]["crop_size"] == 256
    assert loaded["config"]["hardware_profile"] == "cpu_fallback"
    assert loaded["epoch"] == 2
    assert loaded["metrics"]["xview2_score"] == 0.5
    assert norm_stats_from_checkpoint(loaded) == (
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    )

    for key, value in model.state_dict().items():
        assert torch.equal(loaded["state_dict"][key], value)


def test_legacy_checkpoint_is_upgraded(tmp_path):
    model = _tiny_model()
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"encoder": "resnet18", "crop_size": 512},
        },
        path,
    )

    loaded = load_checkpoint(path)
    assert loaded["format_version"] == 1
    assert loaded["encoder_name"] == "resnet18"
    assert loaded["num_classes"] == 5
    assert loaded["damage_classes"] == DAMAGE_CLASS_NAMES
    assert loaded["norm_mean"] == [0.485, 0.456, 0.406]
    assert norm_stats_from_checkpoint(loaded)[1] == [0.229, 0.224, 0.225]
    for key, value in model.state_dict().items():
        assert torch.equal(loaded["state_dict"][key], value)


def test_checkpoint_rejects_wrong_class_count(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, _tiny_model(), num_classes=3)
    with pytest.raises(ValueError):
        load_checkpoint(path, expected_num_classes=5)


def test_checkpoint_missing_weights_raises(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"config": {}}, path)
    with pytest.raises(KeyError):
        load_checkpoint(path)


def test_state_dict_hash_changes_with_weights():
    model = _tiny_model()
    before = state_dict_hash(model.state_dict())
    with torch.no_grad():
        model[0].weight.add_(1.0)
    after = state_dict_hash(model.state_dict())
    assert before != after
    assert state_dict_hash(model.state_dict()) == after
