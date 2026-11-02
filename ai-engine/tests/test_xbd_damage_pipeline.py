import numpy as np
import pytest
import torch

from ai_engine.model.losses import FocalLoss, RubiconLoss, inverse_sqrt_class_weights
from ai_engine.model.metrics import XView2Metrics, decode_damage_logits
from ai_engine.model.siamese_unet import SiameseUNet
from ai_engine.model.checkpoint import save_checkpoint
from ai_engine.model.sat_serving import load_siamese_runtime


def test_damage_loss_ignores_background_and_trains_four_classes():
    criterion = RubiconLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
    loc_logits = torch.zeros((1, 1, 1, 5), requires_grad=True)
    damage_logits = torch.zeros((1, 4, 1, 5), requires_grad=True)
    loc_targets = torch.zeros((1, 1, 1, 5))
    damage_targets = torch.tensor([[[0, 1, 2, 3, 4]]])

    loss, _ = criterion(loc_logits, damage_logits, loc_targets, damage_targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(damage_logits.grad[:, :, :, 0]) == 0
    assert torch.count_nonzero(damage_logits.grad[:, :, :, 1:]) > 0


def test_damage_loss_ignores_legacy_background_logit():
    criterion = RubiconLoss(class_weights=[1.0, 1.0, 1.0, 1.0])
    loc_logits = torch.zeros((1, 1, 1, 2), requires_grad=True)
    damage_logits = torch.zeros((1, 5, 1, 2), requires_grad=True)
    loc_targets = torch.zeros((1, 1, 1, 2))
    damage_targets = torch.tensor([[[0, 4]]])

    loss, _ = criterion(loc_logits, damage_logits, loc_targets, damage_targets)
    loss.backward()

    assert torch.count_nonzero(damage_logits.grad[:, 0]) == 0
    assert torch.count_nonzero(damage_logits.grad[:, 1:]) > 0


def test_inverse_sqrt_weights_use_four_training_class_counts():
    weights = inverse_sqrt_class_weights([100, 25, 4, 1])

    assert weights.shape == (4,)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([2 / 9, 4 / 9, 10 / 9, 20 / 9])
    assert weights.max().item() <= 10.0
    assert inverse_sqrt_class_weights([100, 0, 0, 0]).tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="no labeled"):
        inverse_sqrt_class_weights([0, 0, 0, 0])


def test_focal_class_weight_scales_loss_without_changing_pt():
    logits = torch.tensor([[[[0.0]], [[1.0]], [[0.5]], [[-0.5]]]])
    targets = torch.tensor([[[1]]])
    unweighted = FocalLoss(gamma=2.0, reduction="none")(logits, targets)
    weighted = FocalLoss(
        gamma=2.0,
        weight=torch.tensor([1.0, 3.0, 1.0, 1.0]),
        reduction="none",
    )(logits, targets)

    assert weighted.item() == pytest.approx(unweighted.item() * 3.0)


def test_decode_damage_logits_uses_only_damage_channels():
    logits4 = torch.tensor([[[[0.0]], [[3.0]], [[1.0]], [[2.0]]]])
    assert decode_damage_logits(logits4).item() == 2

    logits5 = torch.tensor([[[[100.0]], [[0.0]], [[3.0]], [[1.0]], [[2.0]]]])
    assert decode_damage_logits(logits5).item() == 2


def test_four_class_checkpoint_loads_in_satellite_runtime(tmp_path):
    model = SiameseUNet(
        encoder_name="resnet18",
        encoder_weights=None,
        num_classes=4,
    )
    save_checkpoint(
        tmp_path / "siamese_unet.pt",
        model,
        encoder_name="resnet18",
        num_classes=4,
        damage_classes=["no-damage", "minor-damage", "major-damage", "destroyed"],
    )

    runtime = load_siamese_runtime(tmp_path)

    assert runtime["model"].num_classes == 4


def test_metrics_score_gt_building_pixels_and_histogram_predicted_buildings():
    metrics = XView2Metrics()
    loc_preds = np.array([[[[0.6, 0.6, 0.1, 0.9, 0.6]]]])
    loc_targets = np.ones_like(loc_preds)
    dmg_preds = np.array([[[4, 1, 2, 2, 4]]])
    dmg_targets = np.array([[[0, 1, 2, 3, 4]]])

    metrics.update(loc_preds, loc_targets, dmg_preds, dmg_targets)
    results = metrics.compute()

    assert results["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    assert results["damage_f1_classes"] == {
        "no-damage": 1.0,
        "minor-damage": 0.6667,
        "major-damage": 0.0,
        "destroyed": 1.0,
    }
    assert results["damage_precision_classes"] == {
        "no-damage": 1.0,
        "minor-damage": 0.5,
        "major-damage": 0.0,
        "destroyed": 1.0,
    }
    assert results["damage_recall_classes"] == {
        "no-damage": 1.0,
        "minor-damage": 1.0,
        "major-damage": 0.0,
        "destroyed": 1.0,
    }
    assert results["damage_f1_macro"] == pytest.approx(0.6667, abs=1e-4)
    assert results["damage_ground_truth_class_histogram"] == {
        "no-damage": 1,
        "minor-damage": 1,
        "major-damage": 1,
        "destroyed": 1,
    }
    assert results["ground_truth_building_pixels"] == 4
    assert results["predicted_building_class_histogram"] == {
        "no-damage": 1,
        "minor-damage": 1,
        "major-damage": 0,
        "destroyed": 2,
    }
    assert results["predicted_building_pixels"] == 4
    assert results["predicted_building_class_histogram_pct"] == {
        "no-damage": 25.0,
        "minor-damage": 25.0,
        "major-damage": 0.0,
        "destroyed": 50.0,
    }
    assert results["xview2_score"] == pytest.approx(
        0.3 * results["loc_f1"] + 0.7 * results["damage_f1_harmonic"],
        abs=1e-4,
    )
