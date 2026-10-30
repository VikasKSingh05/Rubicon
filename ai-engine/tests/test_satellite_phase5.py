"""Unit and integration tests for Phase 5: Siamese U-Net training + satellite inference."""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_image_pair(tmp_path_factory):
    """Generate a 256x256 synthetic pre/post image pair for inference tests."""
    tmp = tmp_path_factory.mktemp("sat_images")
    rng = np.random.default_rng(42)
    pre = (rng.integers(50, 200, (256, 256, 3), dtype=np.uint8))
    post = (rng.integers(50, 200, (256, 256, 3), dtype=np.uint8))
    pre_path = str(tmp / "pre.png")
    post_path = str(tmp / "post.png")
    cv2.imwrite(pre_path, cv2.cvtColor(pre, cv2.COLOR_RGB2BGR))
    cv2.imwrite(post_path, cv2.cvtColor(post, cv2.COLOR_RGB2BGR))
    return {"pre": pre_path, "post": post_path, "tmp": tmp}


# ── Training loop tests ────────────────────────────────────────────────────────

class TestSiameseUNetTrainingLoop:
    """Test the Siamese U-Net training loop with synthetic data."""

    def test_train_synthetic_completes(self, tmp_path):
        """Training on synthetic data completes without error and produces checkpoint."""
        from ai_engine.training.train import train

        result = train(
            synthetic=True,
            epochs=2,
            subset=6,
            output_dir=str(tmp_path / "models"),
        )

        # Checkpoint produced
        assert (tmp_path / "models" / "siamese_unet.pt").exists()
        assert (tmp_path / "models" / "siamese_metrics.json").exists()

        # Result structure
        assert "model_version" in result
        assert "best_val_xview2_score" in result
        assert "final_val" in result
        assert isinstance(result["best_val_xview2_score"], float)
        assert result["best_val_xview2_score"] >= 0.0

    def test_checkpoint_loadable(self, tmp_path):
        """Checkpoint produced by training is loadable by the serving module."""
        from ai_engine.training.train import train
        from ai_engine.model.sat_serving import load_siamese_runtime

        train(synthetic=True, epochs=1, subset=4, output_dir=str(tmp_path / "models"))
        runtime = load_siamese_runtime(str(tmp_path / "models"))

        assert runtime is not None
        assert "model" in runtime
        assert "checkpoint" in runtime

        model = runtime["model"]
        assert hasattr(model, "forward")

    def test_checkpoint_fields(self, tmp_path):
        """Checkpoint contains required fields for serving."""
        from ai_engine.training.train import train

        train(synthetic=True, epochs=1, subset=4, output_dir=str(tmp_path / "models"))

        cp = torch.load(str(tmp_path / "models" / "siamese_unet.pt"),
                       map_location="cpu", weights_only=True)
        assert "model_version" in cp
        assert "encoder_name" in cp
        assert "num_classes" in cp
        assert cp["num_classes"] == 5
        assert "state_dict" in cp


# ── H3 aggregation tests ──────────────────────────────────────────────────────

class TestH3Aggregator:
    """Test H3 cell aggregation from damage masks."""

    def test_aggregate_returns_cells_for_both_resolutions(self):
        """Aggregator produces cells at both res-9 and res-10."""
        from ai_engine.model.aggregator import aggregate_to_h3

        rng = np.random.default_rng(1)
        dmg_mask = rng.integers(0, 5, (256, 256), dtype=np.uint8)
        results = aggregate_to_h3(dmg_mask, resolutions=(9, 10))

        assert 9 in results and 10 in results
        assert len(results[9]) > 0
        assert len(results[10]) > 0

    def test_cell_stats_fields(self):
        """CellStats have required fields with correct types."""
        from ai_engine.model.aggregator import aggregate_to_h3

        dmg_mask = np.ones((128, 128), dtype=np.uint8) * 3  # all major-damage
        results = aggregate_to_h3(dmg_mask, resolutions=(9,))
        cells = results[9]

        assert len(cells) > 0
        cell = cells[0]
        assert isinstance(cell.h3_index, str) and len(cell.h3_index) > 0
        assert cell.resolution == 9
        assert cell.severity in ("RED", "ORANGE", "YELLOW", "GREEN", "GREY")
        assert isinstance(cell.class_counts, list) and len(cell.class_counts) == 5
        assert 0.0 <= cell.building_pixel_fraction <= 1.0

    def test_severity_mapping_all_destroyed(self):
        """All-destroyed mask → all cells marked RED."""
        from ai_engine.model.aggregator import aggregate_to_h3

        dmg_mask = np.full((128, 128), 4, dtype=np.uint8)  # 4 = destroyed
        results = aggregate_to_h3(dmg_mask, resolutions=(9,))
        non_grey = [c for c in results[9] if c.severity != "GREY"]

        assert len(non_grey) > 0
        assert all(c.severity == "RED" for c in non_grey)

    def test_cells_to_geojson_structure(self):
        """cells_to_geojson produces valid GeoJSON FeatureCollection."""
        from ai_engine.model.aggregator import aggregate_to_h3, cells_to_geojson

        dmg_mask = np.ones((128, 128), dtype=np.uint8) * 2  # minor-damage
        results = aggregate_to_h3(dmg_mask, resolutions=(9,))
        fc = cells_to_geojson(results[9])

        assert fc["type"] == "FeatureCollection"
        assert isinstance(fc["features"], list)
        if fc["features"]:
            f = fc["features"][0]
            assert f["type"] == "Feature"
            assert "geometry" in f
            assert "properties" in f
            props = f["properties"]
            assert "h3_index" in props
            assert "severity" in props
            assert "dominant_class" in props


# ── Satellite serving tests ───────────────────────────────────────────────────

class TestSatServingModule:
    """Test the sat_serving module directly (no trained checkpoint required)."""

    def test_load_absent_checkpoint_raises_filenotfounderror(self, tmp_path):
        """load_siamese_runtime raises FileNotFoundError when checkpoint absent."""
        from ai_engine.model.sat_serving import load_siamese_runtime
        with pytest.raises(FileNotFoundError):
            load_siamese_runtime(str(tmp_path))

    def test_infer_satellite_with_model(self, synthetic_image_pair, tmp_path):
        """infer_satellite produces the correct contract body shape."""
        from ai_engine.training.train import train
        from ai_engine.model.sat_serving import load_siamese_runtime, infer_satellite

        # Train 1 epoch to get a checkpoint
        model_dir = str(tmp_path / "models")
        train(synthetic=True, epochs=1, subset=4, output_dir=model_dir)
        runtime = load_siamese_runtime(model_dir)

        result = infer_satellite(
            pre_path=synthetic_image_pair["pre"],
            post_path=synthetic_image_pair["post"],
            model=runtime["model"],
            checkpoint=runtime["checkpoint"],
            device=torch.device("cpu"),
            crop_size=128,
        )

        assert "pre_sha256" in result
        assert "post_sha256" in result
        assert "buildings_geojson" in result
        assert "cells_geojson_r9" in result
        assert "cells_geojson_r10" in result
        assert "model_version" in result
        assert "exec_ms" in result

        assert result["buildings_geojson"]["type"] == "FeatureCollection"
        assert result["cells_geojson_r9"]["type"] == "FeatureCollection"
        assert result["cells_geojson_r10"]["type"] == "FeatureCollection"

        # SHA-256s are valid hex strings
        assert len(result["pre_sha256"]) == 64
        assert len(result["post_sha256"]) == 64


# ── FastAPI /infer/satellite endpoint tests ───────────────────────────────────

class TestSatelliteEndpoint:
    """Test the FastAPI /infer/satellite endpoint in stub mode (no checkpoint)."""

    @pytest.fixture(scope="class")
    def client(self):
        from ai_engine.serve import app
        return TestClient(app)

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_satellite_stub_mode_returns_contract(self, client, synthetic_image_pair):
        """Without a checkpoint, /infer/satellite returns a contract-compliant stub."""
        pre_path = synthetic_image_pair["pre"]
        post_path = synthetic_image_pair["post"]

        with open(pre_path, "rb") as pf, open(post_path, "rb") as qf:
            resp = client.post(
                "/infer/satellite",
                files={
                    "pre": ("pre.png", pf, "image/png"),
                    "post": ("post.png", qf, "image/png"),
                },
            )

        # Either 200 (stub) or 503 (corrupt model), never 500
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            body = resp.json()
            assert "pre_sha256" in body
            assert "post_sha256" in body
            assert "buildings_geojson" in body
            assert "model_version" in body

    def test_bad_extension_rejected(self, client, synthetic_image_pair, tmp_path):
        """Non-image file extension returns 400."""
        txt_path = str(tmp_path / "bad.txt")
        Path(txt_path).write_bytes(b"not an image")
        pre_path = synthetic_image_pair["pre"]

        with open(pre_path, "rb") as pf, open(txt_path, "rb") as tf:
            resp = client.post(
                "/infer/satellite",
                files={
                    "pre": ("pre.png", pf, "image/png"),
                    "post": ("bad.txt", tf, "text/plain"),
                },
            )
        assert resp.status_code == 400
