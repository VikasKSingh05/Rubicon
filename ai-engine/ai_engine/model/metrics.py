"""Metrics implementation strictly adhering to the xView2 / xBD evaluation protocol.

Metrics:
    - localization F1: F1 score of binary building footprint presence (threshold 0.5)
    - per-class damage F1: F1 over ground-truth building pixels for classes 1-4
    - harmonic mean of damage F1s: 4 / sum(1 / (F1_c + 1e-6) for c in [1..4])
    - xView2 overall score: 0.3 * loc_F1 + 0.7 * damage_F1_harmonic
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch


DAMAGE_CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def decode_damage_logits(dmg_logits: torch.Tensor) -> torch.Tensor:
    """Decode four damage classes and return xBD IDs 1-4.

    Five-channel legacy checkpoints are supported by ignoring their background
    logit; new checkpoints emit only the four building-damage logits.
    """
    if dmg_logits.ndim != 4:
        raise ValueError(f"Expected damage logits shaped (B,C,H,W), got {tuple(dmg_logits.shape)}")
    if dmg_logits.shape[1] == 5:
        dmg_logits = dmg_logits[:, 1:5]
    elif dmg_logits.shape[1] != 4:
        raise ValueError(f"Expected 4 damage channels (or 5 legacy), got {dmg_logits.shape[1]}")
    return dmg_logits.argmax(dim=1) + 1


class XView2Metrics:
    """Accumulates confusion counts across validation tiles and computes official xView2 metrics."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.reset()

    def reset(self):
        # Localization counts
        self.loc_tp = 0
        self.loc_fp = 0
        self.loc_fn = 0

        # 4x4 matrix over building damage classes (no-damage through destroyed).
        self.confusion_matrix = np.zeros((4, 4), dtype=np.int64)
        self.predicted_building_class_histogram = np.zeros(4, dtype=np.int64)
        self.predicted_building_pixels = 0
        self.damage_ground_truth_class_histogram = np.zeros(4, dtype=np.int64)

    def update(
        self,
        loc_preds: np.ndarray,
        loc_targets: np.ndarray,
        dmg_preds: np.ndarray,
        dmg_targets: np.ndarray,
    ):
        """Update metrics with a batch of predictions and ground truth arrays."""
        # 1. Localization update (binary)
        pred_bin = (loc_preds > 0.5).astype(bool).reshape(-1)
        tgt_bin = (loc_targets > 0.5).astype(bool).reshape(-1)

        self.loc_tp += int(np.logical_and(pred_bin, tgt_bin).sum())
        self.loc_fp += int(np.logical_and(pred_bin, ~tgt_bin).sum())
        self.loc_fn += int(np.logical_and(~pred_bin, tgt_bin).sum())

        # 2. Damage confusion on ground-truth building pixels only.
        dmg_p = dmg_preds.reshape(-1)
        dmg_t = dmg_targets.reshape(-1)

        ground_truth_mask = (dmg_t >= 1) & (dmg_t <= 4)
        self.damage_ground_truth_class_histogram += np.bincount(
            dmg_t[ground_truth_mask] - 1, minlength=4
        )
        valid_mask = (dmg_t >= 1) & (dmg_t <= 4) & (dmg_p >= 1) & (dmg_p <= 4)
        encoded = 4 * (dmg_t[valid_mask] - 1) + (dmg_p[valid_mask] - 1)
        counts = np.bincount(encoded, minlength=16).reshape(4, 4)
        self.confusion_matrix += counts

        predicted_buildings = pred_bin
        if predicted_buildings.size != dmg_p.size:
            raise ValueError("Localization and damage predictions must have matching shapes")
        predicted_class_mask = predicted_buildings & (dmg_p >= 1) & (dmg_p <= 4)
        predicted_classes = dmg_p[predicted_class_mask] - 1
        self.predicted_building_class_histogram += np.bincount(
            predicted_classes, minlength=4
        )
        self.predicted_building_pixels += int(predicted_classes.size)

    def compute(self) -> Dict[str, Any]:
        """Compute all official xView2 metrics and return detailed report dictionary."""
        # 1. Localization F1
        loc_denom = 2 * self.loc_tp + self.loc_fp + self.loc_fn
        loc_f1 = (2.0 * self.loc_tp / max(1, loc_denom)) if loc_denom > 0 else 0.0

        # 2. Per-class Damage F1 on ground-truth building pixels.
        damage_precisions: Dict[str, float] = {}
        damage_recalls: Dict[str, float] = {}
        damage_f1s: Dict[str, float] = {}
        inv_f1_sum = 0.0
        damage_f1_sum = 0.0

        for c, name in enumerate(DAMAGE_CLASS_NAMES):
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            damage_precisions[name] = float(round(precision, 4))
            damage_recalls[name] = float(round(recall, 4))
            damage_f1s[name] = float(round(f1, 4))
            inv_f1_sum += 1.0 / (f1 + self.eps)
            damage_f1_sum += f1

        histogram_pct = {
            name: float(round(100.0 * int(count) / self.predicted_building_pixels, 2))
            if self.predicted_building_pixels
            else 0.0
            for name, count in zip(
                DAMAGE_CLASS_NAMES, self.predicted_building_class_histogram
            )
        }
        ground_truth_building_pixels = int(
            self.damage_ground_truth_class_histogram.sum()
        )
        ground_truth_histogram_pct = {
            name: float(round(100.0 * int(count) / ground_truth_building_pixels, 2))
            if ground_truth_building_pixels
            else 0.0
            for name, count in zip(
                DAMAGE_CLASS_NAMES, self.damage_ground_truth_class_histogram
            )
        }

        # 3. Harmonic mean of damage F1s
        dmg_f1_harmonic = 4.0 / max(inv_f1_sum, self.eps)
        dmg_f1_macro = damage_f1_sum / len(DAMAGE_CLASS_NAMES)

        # 4. Official xView2 Combined Score: 0.3 * loc_F1 + 0.7 * dmg_F1_harmonic
        xview2_score = (0.3 * loc_f1) + (0.7 * dmg_f1_harmonic)

        return {
            "xview2_score": float(round(xview2_score, 4)),
            "loc_f1": float(round(loc_f1, 4)),
            "damage_f1_harmonic": float(round(dmg_f1_harmonic, 4)),
            "damage_f1_macro": float(round(dmg_f1_macro, 4)),
            "damage_precision_classes": damage_precisions,
            "damage_recall_classes": damage_recalls,
            "damage_f1_classes": damage_f1s,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "damage_ground_truth_class_histogram": {
                name: int(count)
                for name, count in zip(
                    DAMAGE_CLASS_NAMES, self.damage_ground_truth_class_histogram
                )
            },
            "damage_ground_truth_class_histogram_pct": ground_truth_histogram_pct,
            "ground_truth_building_pixels": ground_truth_building_pixels,
            "predicted_building_class_histogram": {
                name: int(count)
                for name, count in zip(
                    DAMAGE_CLASS_NAMES, self.predicted_building_class_histogram
                )
            },
            "predicted_building_class_histogram_pct": histogram_pct,
            "predicted_building_pixels": self.predicted_building_pixels,
        }

    def plot_confusion_matrix(self, output_path: str = "docs/results/confusion_matrix.png"):
        """Plot and save normalized confusion matrix for the 4 damage classes."""
        row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
        norm_cm = np.divide(
            self.confusion_matrix,
            row_sums,
            out=np.zeros_like(self.confusion_matrix, dtype=float),
            where=row_sums > 0,
        )

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            norm_cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=DAMAGE_CLASS_NAMES,
            yticklabels=DAMAGE_CLASS_NAMES,
        )
        plt.title("Damage Classification Confusion Matrix (Normalized)")
        plt.xlabel("Predicted Class")
        plt.ylabel("Ground Truth Class")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
