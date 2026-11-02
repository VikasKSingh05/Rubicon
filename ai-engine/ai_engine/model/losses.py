"""Loss functions for Siamese U-Net disaster damage assessment.

Formula:
    Localization loss: BCE + Dice
    Damage loss:       building-only 4-way weighted CrossEntropy + Focal (gamma=2.0)
    Total loss:        0.4 * loc_loss + 0.6 * dmg_loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1).float()

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Multi-class focal loss to counter foreground/background and damage imbalance.

    Class weights are stored as a buffer, so they move to the GPU together
    with the module when `.to(device)` is called. Pixels whose target equals
    `ignore_index` (for example "un-classified") contribute nothing.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.register_buffer("weight", weight)  # None is allowed

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute p_t from unweighted cross entropy; class alpha must not alter p_t.
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.weight is not None:
            class_indices = targets.clamp(min=0, max=self.weight.numel() - 1)
            focal_loss = focal_loss * self.weight[class_indices]

        if self.reduction == "mean":
            valid = (targets != self.ignore_index).sum().clamp(min=1)
            return focal_loss.sum() / valid
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class RubiconLoss(nn.Module):
    """Combined disaster triage loss: 0.4 * Localization + 0.6 * Damage Classification."""

    def __init__(
        self,
        class_weights: torch.Tensor | list[float] | None = None,
        loc_weight: float = 0.4,
        dmg_weight: float = 0.6,
        focal_gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.loc_weight = loc_weight
        self.dmg_weight = dmg_weight
        self.ignore_index = ignore_index

        # 1. Localization loss components (BCE + Dice)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        # 2. Damage classification loss components (Weighted CE + Focal)
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                weights_tensor = class_weights.detach().clone().float()
            else:
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weights_tensor = torch.ones(4, dtype=torch.float32)

        # Buffers move with criterion.to(device); do NOT store weights as plain attributes
        self.register_buffer("class_weights", weights_tensor)
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        self.focal = FocalLoss(
            gamma=focal_gamma,
            weight=self.class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        loc_logits: torch.Tensor,
        dmg_logits: torch.Tensor,
        loc_targets: torch.Tensor,
        dmg_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loc_targets = loc_targets.float()
        dmg_targets = dmg_targets.long()

        # 1. Localization loss: BCE + Dice
        loss_bce = self.bce(loc_logits, loc_targets)
        loss_dice = self.dice(loc_logits, loc_targets)
        loc_loss = loss_bce + loss_dice

        # xBD target IDs are 1-4 for building damage and 0 for background.
        # Background is owned by localization and ignored by the 4-way head.
        building_mask = (dmg_targets >= 1) & (dmg_targets <= 4)
        damage_targets = torch.where(
            building_mask,
            dmg_targets - 1,
            torch.full_like(dmg_targets, self.ignore_index),
        )
        if dmg_logits.shape[1] == 5:
            damage_logits = dmg_logits[:, 1:5]
        elif dmg_logits.shape[1] == 4:
            damage_logits = dmg_logits
        else:
            raise ValueError(
                f"Damage head must have 4 or legacy 5 channels, got {dmg_logits.shape[1]}"
            )

        if self.class_weights.numel() != 4:
            raise ValueError(
                f"Damage class weights must contain 4 values, got {self.class_weights.numel()}"
            )

        # Guard: if there are no building pixels, CE would return NaN.
        if building_mask.any():
            loss_ce = self.ce(damage_logits, damage_targets)
            loss_focal = self.focal(damage_logits, damage_targets)
        else:
            loss_ce = damage_logits.sum() * 0.0
            loss_focal = damage_logits.sum() * 0.0
        dmg_loss = loss_ce + loss_focal

        # 3. Weighted total loss
        total_loss = (self.loc_weight * loc_loss) + (self.dmg_weight * dmg_loss)

        loss_breakdown = {
            "total_loss": total_loss.item(),
            "loc_loss": loc_loss.item(),
            "loc_bce": loss_bce.item(),
            "loc_dice": loss_dice.item(),
            "dmg_loss": dmg_loss.item(),
            "dmg_ce": loss_ce.item(),
            "dmg_focal": loss_focal.item(),
        }

        return total_loss, loss_breakdown


def inverse_sqrt_class_weights(
    class_pixel_counts: torch.Tensor | list[int],
    max_weight: float = 10.0,
) -> torch.Tensor:
    """Return normalized inverse-square-root weights for four damage classes."""
    counts = torch.as_tensor(class_pixel_counts, dtype=torch.float64)
    if counts.shape != (4,):
        raise ValueError(f"Expected four class pixel counts, got shape {tuple(counts.shape)}")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Class pixel counts must be finite and non-negative")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    present = counts > 0
    if not present.any():
        raise ValueError("Training data contains no labeled building damage pixels")

    weights = torch.zeros_like(counts)
    weights[present] = counts[present].rsqrt()
    weights[present] /= weights[present].mean()
    return weights.clamp(max=max_weight).to(dtype=torch.float32)