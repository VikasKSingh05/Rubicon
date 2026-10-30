"""Rubicon high-resolution drone inspection and damage triage module.

Supports Stage 2 triage:
- Evaluates structural damage on high-resolution drone imagery
- Detects failure modes: roof_collapse, facade_failure, debris_field
- Computes damage area percentages and damage severity classification
- Links inspection results to parent H3 hexagonal cell (from Stage 1 satellite triage)
- Produces GeoJSON contours/bounding polygons for detected damage zones
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import h3
except ImportError:
    h3 = None

logger = logging.getLogger("rubicon.drone-serving")

FAILURE_MODES = ["roof_collapse", "facade_failure", "debris_field"]
DAMAGE_CLASSES = ["none", "minor", "major", "destroyed"]


def analyze_drone_image(
    image_bytes: bytes,
    parent_cell_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    pixel_resolution_m: float = 0.05,
) -> dict[str, Any]:
    """Analyze a high-resolution drone image for disaster damage.

    Parameters
    ----------
    image_bytes : Raw encoded image bytes (PNG / JPEG / TIFF)
    parent_cell_id : Optional parent H3 cell ID (from Stage 1 satellite triage)
    lat, lon : Optional GPS coordinate of the drone capture center
    pixel_resolution_m : Estimated ground resolution per pixel in meters (~0.05m = 5cm)

    Returns
    -------
    dict conforming to POST /infer/drone contract
    """
    start_time = time.perf_counter()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    # Decode image with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image bytes into valid picture.")

    h, w, c = img.shape
    total_pixels = h * w

    # Feature extraction & damage heuristics based on color/texture/edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Edge density & texture variance (rubble / debris exhibits very high Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())

    # Detect high-variance / high-contrast rubble / debris zones
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Detect dark/shadowy collapsed regions (common in roof punctures & structural void collapses)
    _, dark_thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)

    # Combine cues into a synthetic damage likelihood map
    # Normalize laplacian energy
    lap_norm = np.clip(np.abs(laplacian) / 50.0 * 255.0, 0, 255).astype(np.uint8)
    combined = cv2.addWeighted(dilated_edges, 0.4, dark_thresh, 0.3, 0)
    combined = cv2.addWeighted(combined, 0.7, lap_norm, 0.3, 0)

    # Morphological closing to coalesce contiguous damage zones
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    damage_mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, morph_kernel)
    _, damage_binary = cv2.threshold(damage_mask, 80, 255, cv2.THRESH_BINARY)

    damage_pixel_count = int(np.count_nonzero(damage_binary))
    damage_percentage = round(float(damage_pixel_count / total_pixels) * 100.0, 2)

    # Classify overall damage severity
    if damage_percentage < 5.0:
        damage_level = "none"
        severity_label = "GREEN"
    elif damage_percentage < 25.0:
        damage_level = "minor"
        severity_label = "YELLOW"
    elif damage_percentage < 60.0:
        damage_level = "major"
        severity_label = "ORANGE"
    else:
        damage_level = "destroyed"
        severity_label = "RED"

    # Quantify specific failure modes:
    # 1. Roof collapse: concentrated in upper/center quadrants, dark punctures
    center_roi = damage_binary[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    roof_dmg_pct = float(np.count_nonzero(center_roi)) / (center_roi.size + 1e-6)
    roof_collapse = {
        "detected": bool(roof_dmg_pct > 0.15),
        "confidence": round(min(float(roof_dmg_pct * 1.5), 0.99), 3),
        "area_pct": round(float(roof_dmg_pct * 100.0), 2),
    }

    # 2. Debris field: peripheral high-frequency texture / rubble scatter
    debris_pct = float(np.count_nonzero(dilated_edges)) / float(total_pixels)
    debris_field = {
        "detected": bool(debris_pct > 0.10 or lap_var > 350),
        "confidence": round(min(float(debris_pct * 2.2), 0.98), 3),
        "area_pct": round(float(debris_pct * 100.0), 2),
    }

    # 3. Facade failure: vertical edge distortions and irregular boundary cuts
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    facade_energy = float(np.mean(np.abs(sobel_v)))
    facade_detected = bool(damage_percentage > 20.0 and facade_energy > 15.0)
    facade_failure = {
        "detected": facade_detected,
        "confidence": round(min(float(facade_energy / 40.0), 0.95), 3),
        "area_pct": round(min(damage_percentage * 0.6, 100.0), 2),
    }

    # Extract polygon contours for damage regions
    contours, _ = cv2.findContours(damage_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features = []

    # Coordinate mapping: if lat/lon is provided, project pixel coords relative to center
    base_lat = lat if lat is not None else 29.7604
    base_lon = lon if lon is not None else -95.3698
    deg_per_meter_lat = 1.0 / 111139.0
    deg_per_meter_lon = 1.0 / (111139.0 * np.cos(np.radians(base_lat)) + 1e-9)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < (total_pixels * 0.005):  # Filter micro-noise
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue

        # Convert to GeoJSON coordinates
        coords = []
        for pt in approx:
            px, py = pt[0]
            # Offsets in meters from image center
            dx_m = (px - (w / 2.0)) * pixel_resolution_m
            dy_m = ((h / 2.0) - py) * pixel_resolution_m

            pt_lon = round(base_lon + (dx_m * deg_per_meter_lon), 7)
            pt_lat = round(base_lat + (dy_m * deg_per_meter_lat), 7)
            coords.append([pt_lon, pt_lat])

        # Close loop
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        if len(coords) >= 4:
            feature = {
                "type": "Feature",
                "id": f"drone_zone_{i+1}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
                "properties": {
                    "id": f"drone_zone_{i+1}",
                    "area_m2": round(float(area * (pixel_resolution_m**2)), 2),
                    "damage_level": damage_level,
                    "severity": severity_label,
                    "parent_cell_id": parent_cell_id,
                },
            }
            features.append(feature)

    # Validate or compute parent H3 cell
    resolved_cell_id = parent_cell_id
    if not resolved_cell_id and h3 and lat is not None and lon is not None:
        try:
            resolved_cell_id = h3.latlng_to_cell(lat, lon, 9)
        except Exception:
            resolved_cell_id = None

    exec_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    return {
        "image_sha256": image_sha256,
        "parent_cell_id": resolved_cell_id,
        "coordinates": {"lat": base_lat, "lon": base_lon},
        "damage_percentage": damage_percentage,
        "damage_level": damage_level,
        "severity": severity_label,
        "failure_modes": {
            "roof_collapse": roof_collapse,
            "facade_failure": facade_failure,
            "debris_field": debris_field,
        },
        "geojson_features": {
            "type": "FeatureCollection",
            "features": features,
        },
        "metrics": {
            "image_width": w,
            "image_height": h,
            "laplacian_variance": round(lap_var, 2),
            "estimated_damage_area_m2": round(
                damage_pixel_count * (pixel_resolution_m**2), 2
            ),
        },
        "model_version": "rubicon-drone-v1",
        "exec_ms": exec_ms,
    }
