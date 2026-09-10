"""Rubicon ai-engine FastAPI service.

Serves the Phase 3 model through the exact /predict contract (see
docs/api-contracts.md). Behavior:
  - POST /predict: optional multipart `hsi` (.tif/.tiff) and `lidar` (.tif/.las)
    files. With a trained artifact at RUBICON_MODEL_DIR (default ./data/models),
    runs real per-patch inference. Without files or an artifact, returns a stub
    that matches the contract so earlier phases keep working.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

from src.model.serving import predict_files, load_runtime

app = FastAPI(title="Rubicon AI Engine", version="1.0.0")

STUB = {
    "prediction": "Severe Collapse",
    "confidence": 0.91,
    "class_probs": {"None": 0.02, "Moderate": 0.07, "Severe Collapse": 0.91},
    "geojson_polygon": {
        "type": "Polygon",
        "coordinates": [
            [[-95.4, 29.8], [-95.3, 29.8], [-95.3, 29.9], [-95.4, 29.9], [-95.4, 29.8]]
        ],
    },
    "model_version": "stub-v0",
}

_runtime = None
_runtime_dir = None


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "models"


def get_runtime():
    """Load and cache the trained artifact; returns None when unavailable."""
    global _runtime, _runtime_dir
    model_dir = os.environ.get("RUBICON_MODEL_DIR") or str(_default_model_dir())
    if _runtime is not None and _runtime_dir == model_dir:
        return _runtime
    try:
        _runtime = load_runtime(model_dir)
        _runtime_dir = model_dir
        return _runtime
    except FileNotFoundError:
        _runtime, _runtime_dir = None, model_dir
        return None


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-engine", "model_loaded": get_runtime() is not None}


def _save_upload(field) -> str:
    suffix = Path(field.filename or "").suffix.lower() or ".tif"
    fd, path = tempfile.mkstemp(prefix="rubicon_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(field.file.read())
    return path


@app.post("/predict")
def predict(hsi: UploadFile | None = File(None), lidar: UploadFile | None = File(None)):
    """Run the fusion model on an uploaded scene, or return the stub.

    `hsi` may be a .tif/.tiff hyperspectral cube; `lidar` a .tif DSM or raw .las.
    When no files are provided (the backend's Phase 3 wiring path) or no artifact
    is deployed, the stub body is returned so earlier phases keep working.
    """
    runtime = get_runtime()
    if hsi is None or not (hsi and runtime):
        return STUB

    ext_ok = {"hsi": (".tif", ".tiff"), "lidar": (".tif", ".tiff", ".las", ".laz")}
    if Path(hsi.filename or "").suffix.lower() not in ext_ok["hsi"]:
        return {"error": "hsi must be .tif/.tiff"}, 400
    if lidar and Path(lidar.filename or "").suffix.lower() not in ext_ok["lidar"]:
        return {"error": "lidar must be .tif/.tiff/.las/.laz"}, 400

    hsi_path = _save_upload(hsi)
    lidar_path = _save_upload(lidar) if lidar else None
    try:
        model, pca, checkpoint = runtime
        body = predict_files(hsi_path, lidar_path, model, pca, checkpoint)
    finally:
        for p in (hsi_path, lidar_path):
            if p:
                p = Path(p)
                if p.exists():
                    p.unlink()
    return body