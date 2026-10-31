"""xBD label parser — extracts building polygons and damage classes from
the xBD GeoJSON annotation format used by the Rubicon damage-assessment
pipeline.

Key artefacts
-------------
* ``DAMAGE_MAP`` — canonical string → integer class mapping.
* ``parse_xbd_label`` — renders polygons into localisation + damage masks.
* ``cache_tile_masks`` — materialises masks to disk for DataLoader speed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── class contract ───────────────────────────────────────────────────
#   0 = background (no building)
#   1 = no-damage
#   2 = minor-damage
#   3 = major-damage
#   4 = destroyed
# "un-classified" is mapped to 1 (no-damage) as per xBD spec.
DAMAGE_MAP: Dict[str, int] = {
    "no-damage":    1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed":    4,
    "un-classified": 1,
}


def _wkt_to_contour(wkt: str) -> np.ndarray:
    """Convert a POLYGON WKT string to an OpenCV contour (Nx1x2 int32).

    Handles the format ``POLYGON ((x0 y0, x1 y1, ...))`` emitted by xBD
    label files.
    """
    # Strip outer POLYGON (( ... ))
    inner = wkt.replace("POLYGON ((", "").replace("))", "")
    coords = []
    for pair in inner.split(","):
        pair = pair.strip()
        if not pair:
            continue
        x_s, y_s = pair.split()
        coords.append([float(x_s), float(y_s)])
    return np.array(coords, dtype=np.int32).reshape(-1, 1, 2)


def parse_xbd_label(
    json_path: str | Path,
    shape: Tuple[int, int] = (1024, 1024),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Parse a single xBD label JSON and render masks.

    Parameters
    ----------
    json_path : path to the xBD GeoJSON label file.
    shape : (H, W) of the source satellite tile (default 1024×1024).

    Returns
    -------
    loc_mask : uint8 (H, W) — 1 where a building exists, 0 elsewhere.
    dmg_mask : uint8 (H, W) — damage class per pixel (0–4).
    features : list[dict] — raw GeoJSON feature dicts for downstream use.
    """
    json_path = Path(json_path)
    with open(json_path) as fh:
        data = json.load(fh)

    h, w = shape
    loc_mask = np.zeros((h, w), dtype=np.uint8)
    dmg_mask = np.zeros((h, w), dtype=np.uint8)

    features = data.get("features", {}).get("xy", [])
    if isinstance(features, dict):
        features = features.get("features", [])

    for feat in features:
        props = feat.get("properties", {})
        wkt = feat.get("wkt", "")
        if "POLYGON" not in wkt:
            continue
        contour = _wkt_to_contour(wkt)
        # Localisation mask: all buildings → 1
        cv2.fillPoly(loc_mask, [contour], 1)
        # Damage mask: class from DAMAGE_MAP
        subtype = props.get("subtype", "un-classified")
        cls = DAMAGE_MAP.get(subtype, 1)
        cv2.fillPoly(dmg_mask, [contour], cls)

    return loc_mask, dmg_mask, features


def cache_tile_masks(
    tile_id: str,
    pre_json_path: str | Path,
    post_json_path: str | Path,
    cache_dir: str | Path,
    shape: Tuple[int, int] = (1024, 1024),
) -> Tuple[Path, Path]:
    """Render and persist localisation + damage masks for one tile.

    Uses the *post*-disaster label for damage classification and the
    *pre*-disaster label for building localisation.  Results are cached
    as single-channel PNGs under ``cache_dir``.

    Returns
    -------
    (loc_path, dmg_path) — paths to the saved mask files.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    loc_path = cache_dir / f"{tile_id}_loc.png"
    dmg_path = cache_dir / f"{tile_id}_dmg.png"

    if loc_path.exists() and dmg_path.exists():
        return loc_path, dmg_path

    # Post label carries the damage subtype
    loc_mask, dmg_mask, _ = parse_xbd_label(post_json_path, shape=shape)

    cv2.imwrite(str(loc_path), loc_mask)
    cv2.imwrite(str(dmg_path), dmg_mask)
    return loc_path, dmg_path
