"""Rubicon ai-engine FastAPI service.

Serves the Phase 3 model through the exact /predict contract (see
docs/api-contracts.md). Behavior:
  - POST /predict: optional multipart `hsi` (.tif/.tiff) and `lidar` (.tif/.las/.laz)
    files. With a trained artifact at RUBICON_MODEL_DIR (default ./data/models),
    runs real per-patch inference. Without an artifact it returns a stub that
    matches the contract; with a corrupt/unloadable artifact it returns a clean
    503 so the backend's graceful-degrade stub is clearly visible.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import Request

from src.model.serving import predict_files, load_runtime

app = FastAPI(title="Rubicon AI Engine", version="1.0.0")

engine_logger = logging.getLogger("rubicon.engine")


@app.middleware("http")
async def _access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = round((time.perf_counter() - start) * 1000, 2)
    engine_logger.info(
        json.dumps(
            {
                "level": "info",
                "service": "ai-engine",
                "msg": "http",
                "requestId": request.headers.get("x-request-id") or str(uuid.uuid4()),
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "durationMs": dur_ms,
            }
        )
    )
    return response

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
_runtime_error = None  # set when the artifact exists but fails to load


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "models"


def _max_bytes() -> int:
    """Upload cap, in bytes. Mirrors the backend's 100MB limit by default and is
    overridable per-environment for tests/small deployments."""
    return int(os.environ.get("AI_ENGINE_MAX_MB", "100")) * 1024 * 1024


def get_runtime():
    """Load and cache the trained artifact; returns None when unavailable."""
    global _runtime, _runtime_dir, _runtime_error
    model_dir = os.environ.get("RUBICON_MODEL_DIR") or str(_default_model_dir())
    if _runtime is not None and _runtime_dir == model_dir:
        return _runtime
    _runtime_error = None
    try:
        _runtime = load_runtime(model_dir)
        _runtime_dir = model_dir
        return _runtime
    except FileNotFoundError:
        # No artifact deployed yet — graceful pre-model stub mode.
        _runtime, _runtime_dir = None, model_dir
        return None
    except Exception as exc:  # corrupt/partial artifact must not 500 /health
        _runtime, _runtime_dir, _runtime_error = None, model_dir, str(exc)
        return None


@app.get("/health")
def health():
    run = get_runtime()
    body = {"status": "ok", "service": "ai-engine", "model_loaded": run is not None}
    if _runtime_error:
        body["model_error"] = _runtime_error
    return body


def _read_limited(field) -> bytes | None:
    """Read a spooled upload up to the cap; returns None when it overruns."""
    limit = _max_bytes()
    chunks = []
    total = 0
    while True:
        chunk = field.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _save_upload(data: bytes, filename: str) -> str:
    suffix = Path(filename or "").suffix.lower() or ".tif"
    fd, path = tempfile.mkstemp(prefix="rubicon_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _validate_magic(filename: str, kind: str, data: bytes) -> str | None:
    """Reject files whose magic bytes don't match their extension. Added in
    Phase 7 so malformed bytes fail fast at 400 instead of hitting rasterio."""
    ext = Path(filename or "").suffix.lower()
    if ext in (".tif", ".tiff"):
        ok = data[:4] in (b"II*\x00", b"MM\x00*")
        if not ok:
            return "hsi file is not a valid TIFF" if kind == "hsi" else "lidar file is not a valid TIFF"
        return None
    if not data.startswith(b"LASF"):
        return "lidar file is not a valid LAS/LAZ"
    return None


@app.post("/predict")
def predict(hsi: UploadFile | None = File(None), lidar: UploadFile | None = File(None)):
    """Run the fusion model on an uploaded scene, or return the stub."""
    if hsi is None:
        return STUB

    ext_ok = {"hsi": (".tif", ".tiff"), "lidar": (".tif", ".tiff", ".las", ".laz")}
    if Path(hsi.filename or "").suffix.lower() not in ext_ok["hsi"]:
        return JSONResponse(status_code=400, content={"error": "hsi must be .tif/.tiff"})
    if lidar and Path(lidar.filename or "").suffix.lower() not in ext_ok["lidar"]:
        return JSONResponse(status_code=400, content={"error": "lidar must be .tif/.tiff/.las/.laz"})

    hsi_data = _read_limited(hsi)
    if hsi_data is None:
        return JSONResponse(status_code=413, content={"error": "hsi too large"})
    lidar_data = _read_limited(lidar) if lidar else None
    if lidar and lidar_data is None:
        return JSONResponse(status_code=413, content={"error": "lidar too large"})

    magic_error = _validate_magic(hsi.filename or "", "hsi", hsi_data)
    if magic_error:
        return JSONResponse(status_code=400, content={"error": magic_error})
    if lidar:
        magic_error = _validate_magic(lidar.filename or "", "lidar", lidar_data)
        if magic_error:
            return JSONResponse(status_code=400, content={"error": magic_error})

    runtime = get_runtime()
    if runtime is None:
        if _runtime_error:
            return JSONResponse(status_code=503, content={"error": "model unavailable"})
        return STUB

    hsi_path = _save_upload(hsi_data, hsi.filename or "scene.tif")
    lidar_path = _save_upload(lidar_data, lidar.filename or "scene.las") if lidar else None
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