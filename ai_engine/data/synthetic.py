"""Generate a small synthetic xBD-like dataset for isolated testing.

The real xBD dataset is ~60 GB; this module creates a tiny filesystem
layout that mirrors xBD's directory conventions with randomly generated
satellite tiles and matching GeoJSON label files so that unit tests can
run without any external data download.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

_EVENTS = [
    "hurricane-harvey",
    "hurricane-michael",
    "hurricane-florence",
    "mexico-earthquake",
    "midwest-flooding",
    "palu-tsunami",
]

_DAMAGE_CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

_IMG_SIZE = 1024


def _random_polygon(rng: np.random.Generator, size: int = _IMG_SIZE) -> str:
    """Return a WKT POLYGON with 4–6 vertices inside *size* × *size*."""
    cx, cy = rng.integers(100, size - 100, size=2)
    r = rng.integers(20, 60)
    n_pts = rng.integers(4, 7)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
    pts = []
    for a in angles:
        x = int(cx + r * np.cos(a))
        y = int(cy + r * np.sin(a))
        pts.append(f"{x} {y}")
    pts.append(pts[0])  # close ring
    return f"POLYGON (({', '.join(pts)}))"


def _make_label(
    rng: np.random.Generator,
    n_buildings: int = 5,
    include_damage: bool = True,
) -> dict:
    """Generate a synthetic xBD-style GeoJSON label dict."""
    features: List[dict] = []
    for _ in range(n_buildings):
        wkt = _random_polygon(rng)
        props: dict = {"uid": f"bldg-{rng.integers(0, 1_000_000)}"}
        if include_damage:
            props["subtype"] = rng.choice(_DAMAGE_CLASSES)
        features.append({"wkt": wkt, "properties": props})
    return {"features": {"xy": features}}


def generate_synthetic_dataset(
    output_dir: str | Path = "data/synthetic_xbd",
    tiles_per_event: int = 3,
    n_events: int | None = None,
    seed: int = 42,
) -> Path:
    """Create a minimal xBD-like dataset on disk.

    Directory layout::

        output_dir/
          images/
            <event>/
              <tile_id>_pre_disaster.png
              <tile_id>_post_disaster.png
          labels/
            <event>/
              <tile_id>_pre_disaster.json
              <tile_id>_post_disaster.json

    Parameters
    ----------
    output_dir : root directory for the synthetic dataset.
    tiles_per_event : how many image pairs per disaster event.
    n_events : how many events to include (default: all 6).
    seed : random seed for reproducibility.

    Returns
    -------
    Path to *output_dir* (same value passed in, for convenience).
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    events = _EVENTS if n_events is None else _EVENTS[:n_events]

    for event in events:
        img_dir = output_dir / "images" / event
        lbl_dir = output_dir / "labels" / event
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(tiles_per_event):
            tile_id = f"{event}_{i:04d}"

            # Random RGB satellite tiles
            pre_img = rng.integers(0, 255, (_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            post_img = rng.integers(0, 255, (_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"{tile_id}_pre_disaster.png"), pre_img)
            cv2.imwrite(str(img_dir / f"{tile_id}_post_disaster.png"), post_img)

            # Labels — pre has buildings without damage info, post has damage
            pre_label = _make_label(rng, n_buildings=rng.integers(3, 8), include_damage=False)
            post_label = _make_label(rng, n_buildings=rng.integers(3, 8), include_damage=True)
            with open(lbl_dir / f"{tile_id}_pre_disaster.json", "w") as f:
                json.dump(pre_label, f)
            with open(lbl_dir / f"{tile_id}_post_disaster.json", "w") as f:
                json.dump(post_label, f)

    return output_dir
