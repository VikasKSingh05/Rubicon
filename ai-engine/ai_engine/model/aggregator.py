"""H3 hexagonal grid aggregator for Siamese U-Net satellite inference output.

Converts per-pixel damage segmentation predictions from the Siamese U-Net
to H3 hexagonal cell aggregates at resolutions 9 (city-block ~174m) and 10
(~65m) for the /infer/satellite response FeatureCollection.

Severity colour mapping per contract:
  RED    → majority class is "destroyed" (4)
  ORANGE → majority class is "major-damage" (3)
  YELLOW → majority class is "minor-damage" (2)
  GREEN  → majority class is "no-damage" (1)
  GREY   → background only (0) — included for completeness, excluded from alert

Cell attributes exported:
  h3_index, resolution, severity, dominant_class, class_counts,
  mean_confidence, building_pixel_fraction, geometry (GeoJSON Polygon)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import h3
import numpy as np


# ── Severity palette ───────────────────────────────────────────────────────────

SEVERITY_MAP = {
    0: "GREY",
    1: "GREEN",
    2: "YELLOW",
    3: "ORANGE",
    4: "RED",
}

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


@dataclass
class CellStats:
    h3_index: str
    resolution: int
    dominant_class: int
    severity: str
    class_counts: list[int]          # 5 entries, indices 0-4
    building_pixel_fraction: float
    mean_confidence: float = 0.0
    geometry: dict[str, Any] = field(default_factory=dict)


# ── Pixel-to-geographic helpers ───────────────────────────────────────────────

def _pixel_to_latlon(
    row: int, col: int, transform, crs
) -> tuple[float, float]:
    """Convert pixel (row, col) → (lat, lon) using rasterio transform."""
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform

        xs, ys = rasterio.transform.xy(transform, [row], [col])
        if crs and crs.to_epsg() != 4326:
            lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
            return float(lats[0]), float(lons[0])
        return float(ys[0]), float(xs[0])
    except Exception:
        return 0.0, 0.0


def _patch_center_latlons(
    dmg_mask: np.ndarray,
    transform,
    crs,
    patch_size: int = 32,
) -> list[tuple[int, int, float, float]]:
    """Yield (row, col, lat, lon) for the center of each spatial patch."""
    H, W = dmg_mask.shape
    centers = []
    for r0 in range(0, H, patch_size):
        for c0 in range(0, W, patch_size):
            r_c = min(r0 + patch_size // 2, H - 1)
            c_c = min(c0 + patch_size // 2, W - 1)
            lat, lon = _pixel_to_latlon(r_c, c_c, transform, crs)
            centers.append((r0, c0, lat, lon))
    return centers


# ── Core aggregation ──────────────────────────────────────────────────────────

def aggregate_to_h3(
    dmg_mask: np.ndarray,
    conf_map: np.ndarray | None = None,
    transform=None,
    crs=None,
    resolutions: tuple[int, ...] = (9, 10),
    patch_size: int = 32,
) -> dict[int, list[CellStats]]:
    """Aggregate per-pixel damage predictions into H3 hexagonal cells.

    Args:
        dmg_mask: (H, W) uint8 array, values 0-4.
        conf_map: (H, W) float32 confidence per pixel (optional).
        transform: rasterio Affine transform for the image.
        crs: rasterio CRS object.
        resolutions: H3 resolutions to compute (default: 9 and 10).
        patch_size: Spatial granularity in pixels for center sampling.

    Returns:
        Dict mapping resolution → list of CellStats.
    """
    H, W = dmg_mask.shape
    if conf_map is None:
        conf_map = np.zeros((H, W), dtype=np.float32)

    results: dict[int, list[CellStats]] = {r: [] for r in resolutions}

    if transform is None:
        # No georeferencing: build a synthetic lat/lon grid centred on (0, 0)
        # with 1 arcsecond per pixel (~30m), purely for CI/offline mode.
        lat_arr = np.linspace(0.1, -0.1, H)
        lon_arr = np.linspace(-0.1, 0.1, W)
    else:
        lat_arr = None
        lon_arr = None

    # Accumulate per-H3-cell stats per resolution
    for res in resolutions:
        cell_accumulator: dict[str, dict] = {}

        for r0 in range(0, H, patch_size):
            r1 = min(r0 + patch_size, H)
            for c0 in range(0, W, patch_size):
                c1 = min(c0 + patch_size, W)

                patch_dmg = dmg_mask[r0:r1, c0:c1]
                patch_conf = conf_map[r0:r1, c0:c1]

                r_c = (r0 + r1) // 2
                c_c = (c0 + c1) // 2

                if lat_arr is not None:
                    lat = float(lat_arr[min(r_c, H - 1)])
                    lon = float(lon_arr[min(c_c, W - 1)])
                else:
                    lat, lon = _pixel_to_latlon(r_c, c_c, transform, crs)

                if not math.isfinite(lat) or not math.isfinite(lon):
                    continue
                # h3 expects (lat, lon)
                try:
                    h3_idx = h3.latlng_to_cell(lat, lon, res)
                except Exception:
                    continue

                if h3_idx not in cell_accumulator:
                    cell_accumulator[h3_idx] = {
                        "counts": np.zeros(5, dtype=np.int64),
                        "conf_sum": 0.0,
                        "total_pixels": 0,
                    }

                counts = np.bincount(patch_dmg.flatten(), minlength=5)
                cell_accumulator[h3_idx]["counts"] += counts
                cell_accumulator[h3_idx]["conf_sum"] += float(patch_conf.sum())
                cell_accumulator[h3_idx]["total_pixels"] += patch_dmg.size

        # Convert accumulators to CellStats
        for h3_idx, acc in cell_accumulator.items():
            counts = acc["counts"]
            total = acc["total_pixels"]
            building_px = int(counts[1:].sum())
            building_frac = building_px / max(total, 1)

            # Dominant non-background class (or 0 if all bg)
            dom = int(counts[1:].argmax()) + 1 if building_px > 0 else 0
            severity = SEVERITY_MAP[dom]

            mean_conf = acc["conf_sum"] / max(total, 1)

            # H3 cell boundary → GeoJSON Polygon
            try:
                boundary = h3.cell_to_boundary(h3_idx)
                # h3 boundary is [(lat,lon), ...], GeoJSON wants [lon,lat]
                coords = [[lon, lat] for lat, lon in boundary]
                coords.append(coords[0])  # close ring
                geometry = {"type": "Polygon", "coordinates": [coords]}
            except Exception:
                geometry = {}

            results[res].append(CellStats(
                h3_index=h3_idx,
                resolution=res,
                dominant_class=dom,
                severity=severity,
                class_counts=counts.tolist(),
                building_pixel_fraction=round(building_frac, 4),
                mean_confidence=round(float(mean_conf), 4),
                geometry=geometry,
            ))

    return results


def cells_to_geojson(cells: list[CellStats], include_background: bool = False) -> dict:
    """Convert a list of CellStats to a GeoJSON FeatureCollection."""
    features = []
    for c in cells:
        if not include_background and c.severity == "GREY":
            continue
        features.append({
            "type": "Feature",
            "geometry": c.geometry,
            "properties": {
                "h3_index": c.h3_index,
                "resolution": c.resolution,
                "severity": c.severity,
                "dominant_class": CLASS_NAMES[c.dominant_class],
                "class_counts": dict(zip(CLASS_NAMES, c.class_counts)),
                "building_pixel_fraction": c.building_pixel_fraction,
                "mean_confidence": c.mean_confidence,
            },
        })

    return {"type": "FeatureCollection", "features": features}
