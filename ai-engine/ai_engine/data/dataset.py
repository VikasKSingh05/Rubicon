"""PyTorch dataset and sampler for xBD satellite tile pairs.

``XBDDataset`` loads pre/post-disaster image pairs with cached localisation
and damage masks.  It supports random cropping (with a runtime-adjustable
crop size for OOM recovery), optional albumentations transforms, and
normalisation statistics fitted on the training split only.

``get_weighted_sampler`` builds a ``WeightedRandomSampler`` that oversamples
tiles containing severe damage, compensating for the heavy class imbalance in
xBD (most tiles have no-damage or minor-damage).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .parser import get_or_create_masks, parse_xbd_label, tile_has_severe_damage

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_norm_stats(
    tiles: Sequence[Dict[str, Any]],
    max_tiles: int = 100,
    sample_size: int = 128,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Fit per-channel RGB mean/std on a deterministic sample of training tiles.

    Statistics are computed on ``uint8 / 255`` pixels.  Sampling is used so
    this stays fast even on the full xBD train split.
    """
    if not tiles:
        return list(IMAGENET_MEAN), list(IMAGENET_STD)

    rng = np.random.default_rng(seed)
    count = min(max_tiles, len(tiles))
    indices = np.sort(rng.choice(len(tiles), size=count, replace=False))

    total = 0
    sum_ = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)

    for idx in indices:
        img = _load_rgb(tiles[int(idx)]["pre_image"])
        img = cv2.resize(img, (sample_size, sample_size), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        total += flat.shape[0]
        sum_ += flat.sum(axis=0)
        sumsq += (flat ** 2).sum(axis=0)

    mean = sum_ / max(total, 1)
    var = np.maximum(sumsq / max(total, 1) - mean ** 2, 1e-8)
    return mean.round(6).tolist(), np.sqrt(var).round(6).tolist()


class XBDDataset(Dataset):
    """Pre/post satellite tile pairs with localisation + damage masks.

    Parameters
    ----------
    tiles : list of tile descriptor dicts (from ``discover_xbd_tiles``).
    crop_size : spatial extent of the random/centre crop (H = W).  ``None``
        or ``0`` keeps the full 1024×1024 tile.
    is_train : enable random crops/flips for training augmentation.
    cache_dir : directory for cached rendered masks.
    norm_stats : optional ``(mean, std)`` RGB triples.  Defaults to ImageNet.
    transform : optional albumentations transform applied to images + masks.
    """

    def __init__(
        self,
        tiles: List[Dict[str, Any]],
        crop_size: int | None = 512,
        is_train: bool = True,
        cache_dir: str | Path | None = None,
        norm_stats: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        transform=None,
    ) -> None:
        self.tiles = tiles
        self.crop_size = crop_size
        self.is_train = is_train
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform

        if norm_stats is None:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = norm_stats
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.tiles)

    def set_crop_size(self, new_size: int | None) -> None:
        """Adjust crop size at runtime to recover from CUDA OOM."""
        self.crop_size = new_size

    # ── internal helpers ─────────────────────────────────────────────

    def _get_masks(self, tile: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_dir is not None:
            return get_or_create_masks(
                tile_id=tile["tile_id"],
                pre_json_path=tile.get("pre_label"),
                post_json_path=tile["post_label"],
                cache_dir=self.cache_dir,
            )
        loc, _, _ = parse_xbd_label(tile.get("pre_label") or tile["post_label"])
        _, dmg, _ = parse_xbd_label(tile["post_label"])
        loc[dmg > 0] = 1
        return loc, dmg

    def _crop(self, arrays: List[np.ndarray], centre: bool) -> List[np.ndarray]:
        if not self.crop_size:
            return arrays
        h, w = arrays[0].shape[:2]
        cs = min(int(self.crop_size), h, w)
        if cs >= h and cs >= w:
            return arrays
        if centre:
            y = (h - cs) // 2
            x = (w - cs) // 2
        else:
            y = int(np.random.randint(0, h - cs + 1))
            x = int(np.random.randint(0, w - cs + 1))
        return [a[y : y + cs, x : x + cs] for a in arrays]

    # ── __getitem__ ──────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tile = self.tiles[idx]
        pre = _load_rgb(tile["pre_image"])
        post = _load_rgb(tile["post_image"])
        if post.shape[:2] != pre.shape[:2]:
            post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)

        loc, dmg = self._get_masks(tile)
        if loc.shape[:2] != pre.shape[:2]:
            loc = cv2.resize(loc, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_NEAREST)
        if dmg.shape[:2] != pre.shape[:2]:
            dmg = cv2.resize(dmg, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_NEAREST)

        pre, post, loc, dmg = self._crop([pre, post, loc, dmg], centre=not self.is_train)

        if self.is_train:
            # Cheap, label-safe augmentation that needs no extra dependency.
            if np.random.rand() < 0.5:
                pre = np.ascontiguousarray(pre[:, ::-1])
                post = np.ascontiguousarray(post[:, ::-1])
                loc = np.ascontiguousarray(loc[:, ::-1])
                dmg = np.ascontiguousarray(dmg[:, ::-1])
            if np.random.rand() < 0.5:
                pre = np.ascontiguousarray(pre[::-1, :])
                post = np.ascontiguousarray(post[::-1, :])
                loc = np.ascontiguousarray(loc[::-1, :])
                dmg = np.ascontiguousarray(dmg[::-1, :])

        if self.transform is not None:
            aug = self.transform(image=pre, image2=post, mask=loc, mask2=dmg)
            pre, post = aug["image"], aug["image2"]
            loc, dmg = aug["mask"], aug["mask2"]

        pre_f = pre.astype(np.float32) / 255.0
        post_f = post.astype(np.float32) / 255.0
        pre_f = (pre_f - self.mean) / self.std
        post_f = (post_f - self.mean) / self.std

        pre_t = torch.from_numpy(np.ascontiguousarray(pre_f.transpose(2, 0, 1)))
        post_t = torch.from_numpy(np.ascontiguousarray(post_f.transpose(2, 0, 1)))
        loc_t = torch.from_numpy(np.ascontiguousarray(loc[None, ...].astype(np.float32)))
        dmg_t = torch.from_numpy(np.ascontiguousarray(dmg.astype(np.int64)))

        return pre_t, post_t, loc_t, dmg_t


def get_weighted_sampler(
    dataset: XBDDataset,
    severe_multiplier: float = 4.0,
) -> WeightedRandomSampler:
    """Oversample tiles whose post-disaster label contains major/destroyed damage."""
    weights: list[float] = []
    for tile in dataset.tiles:
        severe = tile_has_severe_damage(tile["post_label"])
        weights.append(severe_multiplier if severe else 1.0)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
