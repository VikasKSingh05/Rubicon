"""Metrics implementation strictly adhering to the xView2 / xBD evaluation protocol.

Metrics:
    - localization F1: F1 score of binary building footprint presence (threshold 0.5)
    - per-class damage F1: F1 score for each damage class (1: no-damage, 2: minor, 3: major, 4: destroyed)
    - harmonic mean of damage F1s: 4 / sum(1 / (F1_c + 1e-6) for c in [1..4])
    - xView2 overall score: 0.3 * loc_F1 + 0.7 * damage_F1_harmonic
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DAMAGE_CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


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

        # 5x5 damage confusion matrix (rows = ground truth, cols = prediction)
        self.confusion_matrix = np.zeros((5, 5), dtype=np.int64)

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

        # 2. Damage confusion matrix update
        dmg_p = dmg_preds.reshape(-1)
        dmg_t = dmg_targets.reshape(-1)

        # Vectorized 2D bincount for 5 classes
        valid_mask = (dmg_t >= 0) & (dmg_t < 5) & (dmg_p >= 0) & (dmg_p < 5)
        encoded = 5 * dmg_t[valid_mask] + dmg_p[valid_mask]
        counts = np.bincount(encoded, minlength=25).reshape(5, 5)
        self.confusion_matrix += counts

    def compute(self) -> Dict[str, Any]:
        """Compute all official xView2 metrics and return detailed report dictionary."""
        # 1. Localization F1
        loc_denom = 2 * self.loc_tp + self.loc_fp + self.loc_fn
        loc_f1 = (2.0 * self.loc_tp / max(1, loc_denom)) if loc_denom > 0 else 0.0

        # 2. Per-class Damage F1 (classes 1, 2, 3, 4)
        damage_f1s: Dict[str, float] = {}
        inv_f1_sum = 0.0

        for c in range(1, 5):
            name = DAMAGE_CLASS_NAMES[c - 1]
            tp = self.confusion_matrix[c, c]
            fp = self.confusion_matrix[:, c].sum() - tp
            fn = self.confusion_matrix[c, :].sum() - tp

            denom = 2 * tp + fp + fn
            f1 = (2.0 * tp / denom) if denom > 0 else 0.0
            damage_f1s[name] = float(round(f1, 4))
            inv_f1_sum += 1.0 / (f1 + self.eps)

        # 3. Harmonic mean of damage F1s
        dmg_f1_harmonic = 4.0 / max(inv_f1_sum, self.eps)

        # 4. Official xView2 Combined Score: 0.3 * loc_F1 + 0.7 * dmg_F1_harmonic
        xview2_score = (0.3 * loc_f1) + (0.7 * dmg_f1_harmonic)

        return {
            "xview2_score": float(round(xview2_score, 4)),
            "loc_f1": float(round(loc_f1, 4)),
            "damage_f1_harmonic": float(round(dmg_f1_harmonic, 4)),
            "damage_f1_classes": damage_f1s,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def plot_confusion_matrix(self, output_path: str = "docs/results/confusion_matrix.png"):
        """Plot and save normalized confusion matrix for the 4 damage classes."""
        sub_cm = self.confusion_matrix[1:5, 1:5]
        row_sums = sub_cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(sub_cm, row_sums, out=np.zeros_like(sub_cm, dtype=float), where=row_sums > 0)

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
