"""xBD label parser — extracts building polygons and damage classes from
the xBD GeoJSON annotation format used by the Rubicon damage-assessment
pipeline.

Key artefacts
-------------
* ``DAMAGE_MAP`` / ``INVERSE_DAMAGE_MAP`` — canonical class contract.
* ``parse_xbd_label`` — renders polygons into localisation + damage masks.
* ``get_or_create_masks`` — renders + caches masks for DataLoader speed.
* ``tile_has_severe_damage`` — cheap JSON-only scan for sampler weighting.

Mask contract
-------------
0 = background (no building)
1 = no-damage
2 = minor-damage
3 = major-damage
4 = destroyed

``un-classified`` is mapped to 1 (no-damage), matching the xBD target masks
shipped in ``targets/``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── class contract ───────────────────────────────────────────────────
DAMAGE_MAP: Dict[str, int] = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 1,
}

INVERSE_DAMAGE_MAP: Dict[int, str] = {
    0: "background",
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

DAMAGE_CLASS_NAMES: List[str] = [
    "background",
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

# Bump when rasterization changes so stale cached PNGs are rebuilt.
MASK_CACHE_VERSION = 2

_SEVERE_SUBTYPES = {"major-damage", "destroyed"}


def _wkt_rings(wkt: str) -> List[np.ndarray]:
    """Convert a WKT POLYGON/MULTIPOLYGON to rounded OpenCV contours.

    xBD emits simple ``POLYGON ((x y, ...))`` rings.  We round coordinates to
    the nearest pixel (rather than truncating) because that is what the xBD
    ``targets/`` PNGs use; on the real hold split this raises mask agreement
    from ~99.58% to ~99.98%.
    """
    if not wkt or "(" not in wkt:
        return []

    text = wkt.strip()
    upper = text.upper()
    if upper.startswith("MULTIPOLYGON"):
        bodies = re.findall(r"\(\(([^()]*)\)\)", text)
    else:
        # Everything inside the outer parentheses, e.g. "( (ring) )".
        start = text.find("(")
        end = text.rfind(")")
        bodies = [text[start + 1 : end]]

    rings: List[np.ndarray] = []
    for body in bodies:
        # Split outer ring from any hole rings.
        for ring_text in re.split(r"\)\s*,\s*\(", body.strip().strip("()")):
            points: List[List[int]] = []
            for pair in ring_text.split(","):
                pair = pair.strip().strip("()")
                if not pair:
                    continue
                bits = pair.split()
                if len(bits) < 2:
                    continue
                try:
                    x = int(round(float(bits[0])))
                    y = int(round(float(bits[1])))
                except ValueError:
                    continue
                points.append([x, y])
            if len(points) >= 3:
                rings.append(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2))
    return rings


def _wkt_to_contour(wkt: str) -> np.ndarray:
    """Backwards-compatible helper returning the first (outer) ring."""
    rings = _wkt_rings(wkt)
    return rings[0] if rings else np.empty((0, 1, 2), dtype=np.int32)


def _read_features(json_path: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    features = data.get("features", {}).get("xy", [])
    if isinstance(features, dict):
        features = features.get("features", [])
    return data, features


def parse_xbd_label(
    json_path: str | Path,
    shape: Tuple[int, int] = (1024, 1024),
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Parse a single xBD label JSON and render masks.

    Parameters
    ----------
    json_path : path to the xBD GeoJSON label file.
    shape : (H, W) of the source satellite tile.  The label metadata's
        ``height``/``width`` override this when present.

    Returns
    -------
    loc_mask : uint8 (H, W) — 1 where a building exists, 0 elsewhere.
    dmg_mask : uint8 (H, W) — damage class per pixel (0–4).
    features : list[dict] — raw GeoJSON feature dicts for downstream use.
    """
    data, features = _read_features(json_path)
    meta = data.get("metadata", {}) or {}
    h = int(meta.get("height") or shape[0])
    w = int(meta.get("width") or shape[1])

    loc_mask = np.zeros((h, w), dtype=np.uint8)
    dmg_mask = np.zeros((h, w), dtype=np.uint8)

    for feat in features:
        props = feat.get("properties", {}) or {}
        rings = _wkt_rings(feat.get("wkt", ""))
        if not rings:
            continue
        subtype = props.get("subtype", "un-classified")
        cls = DAMAGE_MAP.get(subtype, 1)

        # Outer ring = building footprint; any extra rings are holes.
        cv2.fillPoly(loc_mask, [rings[0]], 1)
        cv2.fillPoly(dmg_mask, [rings[0]], cls)
        for hole in rings[1:]:
            cv2.fillPoly(loc_mask, [hole], 0)
            cv2.fillPoly(dmg_mask, [hole], 0)

    return loc_mask, dmg_mask, features


def tile_has_severe_damage(json_path: str | Path) -> bool:
    """Return True when a post-disaster label contains major/destroyed damage.

    This is a JSON-only scan (no 1024×1024 rasterization) so building the
    weighted sampler over thousands of tiles stays fast.
    """
    try:
        _, features = _read_features(json_path)
    except (OSError, json.JSONDecodeError):
        return False
    for feat in features:
        subtype = (feat.get("properties", {}) or {}).get("subtype", "")
        if subtype in _SEVERE_SUBTYPES:
            return True
    return False


def _cache_is_fresh(cache_path: Path, sources: List[Optional[str | Path]]) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for src in sources:
        if src is None:
            continue
        src_path = Path(src)
        if not src_path.exists():
            return False
        if src_path.stat().st_mtime > cache_mtime:
            return False
    return True


def get_or_create_masks(
    tile_id: str,
    pre_json_path: Optional[str | Path],
    post_json_path: str | Path,
    cache_dir: str | Path,
    shape: Tuple[int, int] = (1024, 1024),
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render (or load) localisation + damage masks for one tile.

    Localisation comes from the **pre-disaster** label (building footprints
    exist before the event); damage classes come from the **post-disaster**
    label.  Cached PNGs are invalidated when either source JSON is newer.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    loc_path = cache_dir / f"{tile_id}_loc.png"
    dmg_path = cache_dir / f"{tile_id}_dmg.png"
    sources = [pre_json_path, post_json_path]

    if not force and _cache_is_fresh(loc_path, sources) and _cache_is_fresh(dmg_path, sources):
        loc = cv2.imread(str(loc_path), cv2.IMREAD_UNCHANGED)
        dmg = cv2.imread(str(dmg_path), cv2.IMREAD_UNCHANGED)
        if loc is not None and dmg is not None:
            return loc, dmg

    loc_mask, _, _ = parse_xbd_label(pre_json_path or post_json_path, shape=shape)
    _, dmg_mask, _ = parse_xbd_label(post_json_path, shape=shape)

    # Keep the two masks geometrically consistent: any damage pixel must be a
    # building pixel.  This guards against pre/post annotation drift.
    dmg_mask = dmg_mask.astype(np.uint8, copy=False)
    loc_mask = loc_mask.astype(np.uint8, copy=False)
    loc_mask[dmg_mask > 0] = 1

    cv2.imwrite(str(loc_path), loc_mask)
    cv2.imwrite(str(dmg_path), dmg_mask)
    return loc_mask, dmg_mask


def cache_tile_masks(
    tile_id: str,
    pre_json_path: Optional[str | Path],
    post_json_path: str | Path,
    cache_dir: str | Path,
    shape: Tuple[int, int] = (1024, 1024),
    force: bool = False,
) -> Tuple[Path, Path]:
    """Render and persist masks, returning the cached PNG paths."""
    cache_dir = Path(cache_dir)
    get_or_create_masks(
        tile_id=tile_id,
        pre_json_path=pre_json_path,
        post_json_path=post_json_path,
        cache_dir=cache_dir,
        shape=shape,
        force=force,
    )
    return cache_dir / f"{tile_id}_loc.png", cache_dir / f"{tile_id}_dmg.png"
