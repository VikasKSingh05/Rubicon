"""Hardening tests: oversized uploads and corrupt-artifact degradation."""
import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.inference as inference
from src.inference import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_runtime(monkeypatch, tmp_path):
    """Reset cached runtime state and default to a *missing* model dir so tests
    never accidentally load a developer's local artifact."""
    monkeypatch.setenv("RUBICON_MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(inference, "_runtime", None)
    monkeypatch.setattr(inference, "_runtime_dir", None)
    monkeypatch.setattr(inference, "_runtime_error", None)
    monkeypatch.delenv("AI_ENGINE_MAX_MB", raising=False)


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_ENGINE_MAX_MB", "0")
    res = client.post(
        "/predict",
        files={"hsi": ("big.tif", b"0123456789")},
    )
    assert res.status_code == 413
    assert res.json() == {"error": "hsi too large"}


def test_corrupt_artifact_returns_clean_503(tmp_path, monkeypatch):
    model_dir = tmp_path / "corrupt"
    model_dir.mkdir()
    (model_dir / "model.pt").write_bytes(b"this is not a torch checkpoint")
    (model_dir / "pca.pkl").write_bytes(pickle.dumps("also not a pca"))
    monkeypatch.setenv("RUBICON_MODEL_DIR", str(model_dir))
    monkeypatch.setattr(inference, "_runtime", None)
    monkeypatch.setattr(inference, "_runtime_dir", None)
    monkeypatch.setattr(inference, "_runtime_error", None)

    res = client.post("/predict", files={"hsi": ("a.tif", b"fake-hsi")})
    assert res.status_code == 503
    assert res.json() == {"error": "model unavailable"}

    health = client.get("/health").json()
    assert health["model_loaded"] is False
    assert "model_error" in health


def test_missing_artifact_keeps_graceful_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBICON_MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(inference, "_runtime", None)
    monkeypatch.setattr(inference, "_runtime_dir", None)
    monkeypatch.setattr(inference, "_runtime_error", None)

    res = client.post("/predict", files={"hsi": ("a.tif", b"fake-hsi")})
    assert res.status_code == 200
    assert res.json()["model_version"] == "stub-v0"