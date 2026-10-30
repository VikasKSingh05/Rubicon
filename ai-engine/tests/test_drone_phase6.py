"""Unit and integration tests for Phase 6: Drone triage endpoint + cell linkage."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_engine.model.drone_serving import analyze_drone_image, FAILURE_MODES, DAMAGE_CLASSES
from ai_engine.serve import app


def _create_synthetic_drone_image(pattern: str = "normal", size: tuple[int, int] = (256, 256)) -> bytes:
    """Generate synthetic drone images representing different disaster conditions."""
    h, w = size
    if pattern == "intact":
        # Smooth building roof with minimal high-frequency noise
        img = np.ones((h, w, 3), dtype=np.uint8) * 180
        # Draw clean geometric building roof edges
        cv2.rectangle(img, (40, 40), (w - 40, h - 40), (120, 120, 120), -1)
        cv2.rectangle(img, (40, 40), (w - 40, h - 40), (60, 60, 60), 2)
    elif pattern == "roof_collapse":
        # Roof with dark central puncture and jagged collapsed hole
        img = np.ones((h, w, 3), dtype=np.uint8) * 160
        cv2.rectangle(img, (40, 40), (w - 40, h - 40), (130, 130, 130), -1)
        # Jagged central void
        pts = np.array([[80, 80], [180, 90], [170, 170], [90, 160]], np.int32)
        cv2.fillPoly(img, [pts], (20, 20, 20))
        # High contrast debris around hole
        noise = np.random.randint(0, 150, (60, 60, 3), dtype=np.uint8)
        img[90:150, 90:150] = noise
    elif pattern == "severe_debris":
        # Heavy rubble and destroyed debris field
        img = np.random.randint(20, 230, (h, w, 3), dtype=np.uint8)
        # Add high-contrast irregular cracks
        for _ in range(15):
            pt1 = (np.random.randint(0, w), np.random.randint(0, h))
            pt2 = (np.random.randint(0, w), np.random.randint(0, h))
            cv2.line(img, pt1, pt2, (10, 10, 10), 3)
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)

    ok, buf = cv2.imencode(".png", img)
    assert ok, "Failed to encode synthetic test image"
    return buf.tobytes()


def test_drone_serving_intact_image():
    """Intact structure returns low damage percentage and 'none' or 'minor' level."""
    img_bytes = _create_synthetic_drone_image(pattern="intact")
    result = analyze_drone_image(
        image_bytes=img_bytes,
        parent_cell_id="8928308280fffff",
        lat=29.7604,
        lon=-95.3698,
    )

    assert result["image_sha256"]
    assert result["parent_cell_id"] == "8928308280fffff"
    assert result["damage_percentage"] < 30.0
    assert result["damage_level"] in ("none", "minor")
    assert "roof_collapse" in result["failure_modes"]
    assert "facade_failure" in result["failure_modes"]
    assert "debris_field" in result["failure_modes"]
    assert result["model_version"] == "rubicon-drone-v1"
    assert result["exec_ms"] >= 0.0


def test_drone_serving_roof_collapse_detection():
    """Roof collapse pattern triggers roof_collapse failure mode."""
    img_bytes = _create_synthetic_drone_image(pattern="roof_collapse")
    result = analyze_drone_image(
        image_bytes=img_bytes,
        parent_cell_id="89283082813ffff",
        lat=29.7610,
        lon=-95.3680,
    )

    assert result["parent_cell_id"] == "89283082813ffff"
    assert result["failure_modes"]["roof_collapse"]["detected"] is True
    assert result["failure_modes"]["roof_collapse"]["confidence"] > 0.1
    assert result["damage_percentage"] > 5.0
    assert result["severity"] in ("YELLOW", "ORANGE", "RED")
    assert result["geojson_features"]["type"] == "FeatureCollection"


def test_drone_serving_debris_field_detection():
    """Severe debris field triggers debris_field mode and high damage severity."""
    img_bytes = _create_synthetic_drone_image(pattern="severe_debris")
    result = analyze_drone_image(
        image_bytes=img_bytes,
        parent_cell_id="8928308280fffff",
    )

    assert result["failure_modes"]["debris_field"]["detected"] is True
    assert result["damage_percentage"] > 25.0
    assert result["severity"] in ("ORANGE", "RED")
    assert result["damage_level"] in ("major", "destroyed")
    assert len(result["geojson_features"]["features"]) >= 1


def test_drone_serving_h3_cell_resolution():
    """When parent_cell_id is omitted, lat/lon dynamically resolves H3 cell index."""
    img_bytes = _create_synthetic_drone_image(pattern="intact")
    result = analyze_drone_image(
        image_bytes=img_bytes,
        parent_cell_id=None,
        lat=29.7604,
        lon=-95.3698,
    )
    # H3 should auto-resolve resolution 9 index for Houston coords
    assert result["parent_cell_id"] is not None
    assert len(result["parent_cell_id"]) == 15  # standard 15-char H3 index hex string


def test_drone_endpoint_fastapi_contract():
    """FastAPI POST /infer/drone accepts multipart upload and validates response schema."""
    client = TestClient(app)
    img_bytes = _create_synthetic_drone_image(pattern="roof_collapse")

    response = client.post(
        "/infer/drone",
        files={"image": ("drone_site_01.png", io.BytesIO(img_bytes), "image/png")},
        data={"parent_cell_id": "8928308280fffff", "lat": 29.7604, "lon": -95.3698},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert "image_sha256" in data
    assert data["parent_cell_id"] == "8928308280fffff"
    assert "damage_percentage" in data
    assert data["damage_level"] in DAMAGE_CLASSES
    assert data["severity"] in ("GREEN", "YELLOW", "ORANGE", "RED")
    assert "failure_modes" in data
    assert "roof_collapse" in data["failure_modes"]
    assert "facade_failure" in data["failure_modes"]
    assert "debris_field" in data["failure_modes"]
    assert data["geojson_features"]["type"] == "FeatureCollection"
    assert data["model_version"] == "rubicon-drone-v1"


def test_drone_endpoint_invalid_extension():
    """FastAPI POST /infer/drone rejects non-image extension with 400."""
    client = TestClient(app)
    response = client.post(
        "/infer/drone",
        files={"image": ("report.pdf", io.BytesIO(b"%PDF-1.4 test"), "application/pdf")},
    )
    assert response.status_code == 400
    assert "must be PNG/JPEG/GeoTIFF" in response.json()["error"]
