"""Loss functions for Siamese U-Net disaster damage assessment.

Formula:
    Localization loss: BCE + Dice
    Damage loss:       class-weighted CrossEntropy + Focal (gamma=2.0)
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
        # Per-pixel cross entropy, no reduction; ignored pixels give 0 loss
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            ignore_index=self.ignore_index,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

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
            # Default: background downweighted, damaged classes boosted
            weights_tensor = torch.tensor([0.2, 1.0, 2.0, 2.5, 3.0], dtype=torch.float32)

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

        # 2. Damage loss: class-weighted CE + Focal
        # Guard: if every pixel is ignored, CE would return NaN
        if (dmg_targets != self.ignore_index).any():
            loss_ce = self.ce(dmg_logits, dmg_targets)
            loss_focal = self.focal(dmg_logits, dmg_targets)
        else:
            loss_ce = dmg_logits.sum() * 0.0
            loss_focal = dmg_logits.sum() * 0.0
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