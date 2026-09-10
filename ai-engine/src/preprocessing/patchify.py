"""Tile aligned HSI/DEM/label grids into fixed-size labeled patches.

A patch is kept only if at least `min_labeled` of its pixels carry a label.
The patch label defaults to the majority land-cover class of its labeled pixels
(ties resolved by lowest class id). Severity-proxy remapping happens in dataset.py.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


def sliding_windows(shape, patch_size: int = 32, overlap: float = 0.0) -> Iterator[tuple]:
    """Yield (row0, col0, row1, col1) windows covering the grid.

    If the grid is not evenly divisible by the stride, trailing partial rows and
    last full window are anchored so the bottom-right corner is always covered.
    """
    rows, cols = shape
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    stride = max(1, int(patch_size * (1.0 - overlap)))
    if stride > patch_size:
        stride = patch_size

    anchors_r = list(range(0, rows - patch_size + 1, stride)) or [0]
    anchors_c = list(range(0, cols - patch_size + 1, stride)) or [0]
    if anchors_r[-1] + patch_size < rows:
        anchors_r.append(rows - patch_size)
    if anchors_c[-1] + patch_size < cols:
        anchors_c.append(cols - patch_size)

    for r0 in anchors_r:
        for c0 in anchors_c:
            yield r0, c0, r0 + patch_size, c0 + patch_size


def majority_label(labels: np.ndarray) -> int:
    """Most common positive label in the patch; 0 if no labeled pixels present."""
    valid = labels[labels > 0]
    if valid.size == 0:
        return 0
    counts = np.bincount(valid)
    # break ties toward the smallest class id
    return int(np.flatnonzero(counts == counts.max())[0])


def patch_dataset(
    hsi: np.ndarray,
    dem: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 32,
    overlap: float = 0.0,
    min_labeled: float = 0.5,
) -> dict:
    """Tile aligned HSI/DEM/labels into labeled patches.

    Returns a dict with:
      hsi     (num_patches, patch_size, patch_size, bands) float32
      dem     (num_patches, patch_size, patch_size) float32
      y_class (num_patches,) majority land-cover class id (0 = none)
      coords  (num_patches, 2) top-left (row, col) of each patch
    """
    if hsi.ndim != 3:
        raise ValueError("hsi must be (rows, cols, bands)")
    if dem.shape != hsi.shape[:2]:
        raise ValueError("dem must match hsi spatial dims")
    if labels.shape != hsi.shape[:2]:
        raise ValueError("labels must match hsi spatial dims")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    out_hsi, out_dem, out_y, out_xy = [], [], [], []
    for r0, c0, r1, c1 in sliding_windows(hsi.shape[:2], patch_size, overlap):
        lab = labels[r0:r1, c0:c1]
        if lab.size == 0:
            continue
        labeled = (lab > 0).mean()
        if labeled < min_labeled:
            continue
        out_hsi.append(hsi[r0:r1, c0:c1])
        out_dem.append(dem[r0:r1, c0:c1])
        out_y.append(majority_label(lab))
        out_xy.append((r0, c0))

    if not out_hsi:
        raise ValueError(f"no labeled patches found (min_labeled={min_labeled})")

    return {
        "hsi": np.stack(out_hsi).astype(np.float32),
        "dem": np.stack(out_dem).astype(np.float32),
        "y_class": np.asarray(out_y, dtype=np.int64),
        "coords": np.asarray(out_xy, dtype=np.int64),
    }