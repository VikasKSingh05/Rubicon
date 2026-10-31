"""PyTorch dataset and sampler for xBD satellite tile pairs.

``XBDDataset`` loads pre/post-disaster image pairs with their cached
localisation and damage masks.  It supports random cropping (with a
runtime-adjustable crop size for OOM recovery) and standard
augmentation via albumentations.

``get_weighted_sampler`` builds a ``WeightedRandomSampler`` that
oversamples tiles containing severe damage, compensating for the heavy
class imbalance in xBD (most tiles have no-damage or minor-damage).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .parser import cache_tile_masks, parse_xbd_label, DAMAGE_MAP


class XBDDataset(Dataset):
    """Pre/post satellite tile pairs with localisation + damage masks.

    Parameters
    ----------
    tiles : list of tile descriptor dicts (from ``discover_xbd_tiles``).
    crop_size : spatial extent of the random crop (H = W).
    is_train : enable random flips / rotations for training augmentation.
    cache_dir : directory for caching rendered masks.
    transform : optional albumentations transform (applied to both
        images and masks jointly).
    """

    def __init__(
        self,
        tiles: List[Dict[str, Any]],
        crop_size: int = 512,
        is_train: bool = True,
        cache_dir: str | Path | None = None,
        transform=None,
    ) -> None:
        self.tiles = tiles
        self.crop_size = crop_size
        self.is_train = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def set_crop_size(self, new_size: int) -> None:
        """Halve crop size at runtime to recover from CUDA OOM."""
        self.crop_size = new_size

    # ── internal helpers ─────────────────────────────────────────────

    def _load_image(self, path: Path | str) -> np.ndarray:
        """Read an image as RGB uint8 (H, W, 3)."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_masks(self, tile: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (loc_mask, dmg_mask) for the tile, using cache if available."""
        if self.cache_dir is not None:
            loc_path, dmg_path = cache_tile_masks(
                tile_id=tile["tile_id"],
                pre_json_path=tile["pre_label"],
                post_json_path=tile["post_label"],
                cache_dir=self.cache_dir,
            )
            loc = cv2.imread(str(loc_path), cv2.IMREAD_UNCHANGED)
            dmg = cv2.imread(str(dmg_path), cv2.IMREAD_UNCHANGED)
            return loc, dmg
        # No cache — parse on the fly
        loc, dmg, _ = parse_xbd_label(tile["post_label"])
        return loc, dmg

    def _random_crop(self, *arrays: np.ndarray) -> list[np.ndarray]:
        """Crop all arrays to ``(crop_size, crop_size)`` from a random origin."""
        h, w = arrays[0].shape[:2]
        cs = min(self.crop_size, h, w)
        y = np.random.randint(0, h - cs + 1)
        x = np.random.randint(0, w - cs + 1)
        return [a[y : y + cs, x : x + cs] for a in arrays]

    def _centre_crop(self, *arrays: np.ndarray) -> list[np.ndarray]:
        """Deterministic centre crop for validation."""
        h, w = arrays[0].shape[:2]
        cs = min(self.crop_size, h, w)
        y = (h - cs) // 2
        x = (w - cs) // 2
        return [a[y : y + cs, x : x + cs] for a in arrays]

    # ── __getitem__ ──────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tile = self.tiles[idx]
        pre = self._load_image(tile["pre_img"])
        post = self._load_image(tile["post_img"])
        loc, dmg = self._get_masks(tile)

        # Crop
        if self.is_train:
            pre, post, loc, dmg = self._random_crop(pre, post, loc, dmg)
        else:
            pre, post, loc, dmg = self._centre_crop(pre, post, loc, dmg)

        # Optional albumentations transform
        if self.transform is not None:
            aug = self.transform(
                image=pre,
                image2=post,
                mask=loc,
                mask2=dmg,
            )
            pre, post = aug["image"], aug["image2"]
            loc, dmg = aug["mask"], aug["mask2"]

        # Convert to tensors
        pre_t = torch.from_numpy(pre.transpose(2, 0, 1).astype(np.float32) / 255.0)
        post_t = torch.from_numpy(post.transpose(2, 0, 1).astype(np.float32) / 255.0)
        loc_t = torch.from_numpy(loc[None, ...].astype(np.float32))
        dmg_t = torch.from_numpy(dmg.astype(np.int64))

        return pre_t, post_t, loc_t, dmg_t


def get_weighted_sampler(
    dataset: XBDDataset,
    severe_multiplier: float = 4.0,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples severe-damage tiles.

    Tiles whose post-disaster label contains any class ≥ 3 (major-damage
    or destroyed) receive a weight of ``severe_multiplier``; all others
    get weight 1.0.  This helps counter the extreme class imbalance in
    xBD where ~70 % of buildings are undamaged.

    Parameters
    ----------
    dataset : an XBDDataset instance (tiles are inspected via label files).
    severe_multiplier : sampling weight for tiles with severe damage.

    Returns
    -------
    WeightedRandomSampler — pass directly to DataLoader(sampler=...).
    """
    weights: list[float] = []
    for tile in dataset.tiles:
        _, dmg, _ = parse_xbd_label(tile["post_label"])
        if np.any(dmg >= 3):
            weights.append(severe_multiplier)
        else:
            weights.append(1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
