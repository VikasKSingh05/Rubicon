"""Rubicon ai-engine satellite inference endpoints.

Adds to the existing FastAPI app (src/inference.py):
  POST /infer/satellite  — accepts pre + post image pair, returns GeoJSON damage maps

This module is imported by src/inference.py (or served standalone) and preserves
the legacy /predict endpoint unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from starlette.requests import Request

from ai_engine.model.sat_serving import load_siamese_runtime, infer_satellite
from ai_engine.model.drone_serving import analyze_drone_image

import torch

app = FastAPI(title="Rubicon AI Engine — Satellite Triage", version="2.0.0")

sat_logger = logging.getLogger("rubicon.sat-engine")

_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MAX_MB = int(os.environ.get("AI_ENGINE_MAX_MB", "200"))
_MAX_BYTES = _MAX_MB * 1024 * 1024

_sat_runtime: dict | None = None
_sat_runtime_dir: str | None = None
_sat_runtime_error: str | None = None


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "models" / "xbd"


def get_sat_runtime() -> dict | None:
    """Load and cache the Siamese U-Net checkpoint. Returns None if absent."""
    global _sat_runtime, _sat_runtime_dir, _sat_runtime_error
    model_dir = os.environ.get("RUBICON_MODEL_DIR") or str(_default_model_dir())
    if _sat_runtime is not None and _sat_runtime_dir == model_dir:
        return _sat_runtime
    _sat_runtime_error = None
    try:
        _sat_runtime = load_siamese_runtime(model_dir)
        _sat_runtime_dir = model_dir
        return _sat_runtime
    except FileNotFoundError:
        _sat_runtime, _sat_runtime_dir = None, model_dir
        return None
    except Exception as exc:
        _sat_runtime, _sat_runtime_dir = None, model_dir
        _sat_runtime_error = str(exc)
        return None


@app.middleware("http")
async def _access_log(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = round((time.perf_counter() - start) * 1000, 2)
    sat_logger.info(json.dumps({
        "level": "info",
        "service": "ai-engine-sat",
        "msg": "http",
        "requestId": request.headers.get("x-request-id") or str(uuid.uuid4()),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "durationMs": dur_ms,
    }))
    return response


@app.get("/health")
def health():
    runtime = get_sat_runtime()
    body = {
        "status": "ok",
        "service": "ai-engine-satellite",
        "model_loaded": runtime is not None,
    }
    if _sat_runtime_error:
        body["model_error"] = _sat_runtime_error
    return body


def _read_upload(field: UploadFile) -> bytes | None:
    """Read upload up to size cap; return None on overrun."""
    chunks, total = [], 0
    while True:
        chunk = field.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _save_tmp(data: bytes, filename: str) -> str:
    suffix = Path(filename or "").suffix.lower() or ".png"
    fd, path = tempfile.mkstemp(prefix="rubicon_sat_", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ── Synthetic stub for no-model mode ─────────────────────────────────────────

def _satellite_stub(pre_bytes: bytes, post_bytes: bytes) -> dict:
    import hashlib
    return {
        "pre_sha256": hashlib.sha256(pre_bytes).hexdigest(),
        "post_sha256": hashlib.sha256(post_bytes).hexdigest(),
        "buildings_geojson": {"type": "FeatureCollection", "features": []},
        "cells_geojson_r9": {"type": "FeatureCollection", "features": []},
        "cells_geojson_r10": {"type": "FeatureCollection", "features": []},
        "model_version": "stub-satellite-v0",
        "damage_classes": ["background", "no-damage", "minor-damage", "major-damage", "destroyed"],
        "exec_ms": 0.0,
    }


# ── /infer/satellite endpoint ─────────────────────────────────────────────────

@app.post("/infer/satellite")
def infer_satellite_endpoint(
    pre: UploadFile = File(...),
    post: UploadFile = File(...),
):
    """Accept pre + post optical image pair, return damage GeoJSON maps.

    Accepts: PNG, JPEG, GeoTIFF (.tif/.tiff)
    Returns: {pre_sha256, post_sha256, buildings_geojson, cells_geojson_r9,
              cells_geojson_r10, model_version, damage_classes, exec_ms}
    """
    # Validate extensions
    pre_ext = Path(pre.filename or "").suffix.lower()
    post_ext = Path(post.filename or "").suffix.lower()
    if pre_ext not in _ALLOWED_IMAGE_EXTS:
        return JSONResponse(status_code=400, content={"error": "pre image must be PNG/JPEG/GeoTIFF"})
    if post_ext not in _ALLOWED_IMAGE_EXTS:
        return JSONResponse(status_code=400, content={"error": "post image must be PNG/JPEG/GeoTIFF"})

    # Read uploads
    pre_data = _read_upload(pre)
    if pre_data is None:
        return JSONResponse(status_code=413, content={"error": f"pre image exceeds {_MAX_MB} MB"})
    post_data = _read_upload(post)
    if post_data is None:
        return JSONResponse(status_code=413, content={"error": f"post image exceeds {_MAX_MB} MB"})

    runtime = get_sat_runtime()
    if runtime is None:
        if _sat_runtime_error:
            return JSONResponse(status_code=503, content={"error": "satellite model unavailable"})
        # Pre-model stub mode: return synthetic empty response
        sat_logger.warning("satellite model not loaded, returning stub")
        return _satellite_stub(pre_data, post_data)

    # Save to temp files and run inference
    pre_path = _save_tmp(pre_data, pre.filename or "pre.png")
    post_path = _save_tmp(post_data, post.filename or "post.png")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = infer_satellite(
            pre_path=pre_path,
            post_path=post_path,
            model=runtime["model"],
            checkpoint=runtime["checkpoint"],
            device=device,
        )
    finally:
        for p in (pre_path, post_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    return result


# ── /infer/drone endpoint ─────────────────────────────────────────────────────

@app.post("/infer/drone")
def infer_drone_endpoint(
    image: UploadFile = File(...),
    parent_cell_id: str | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
):
    """Accept high-resolution drone image, return structural failure analysis.

    Accepts: PNG, JPEG, GeoTIFF (.tif/.tiff)
    Optional inputs: parent_cell_id (H3 index), lat, lon
    Returns: {image_sha256, parent_cell_id, coordinates, damage_percentage,
              damage_level, severity, failure_modes, geojson_features,
              metrics, model_version, exec_ms}
    """
    img_ext = Path(image.filename or "").suffix.lower()
    if img_ext not in _ALLOWED_IMAGE_EXTS:
        return JSONResponse(status_code=400, content={"error": "image must be PNG/JPEG/GeoTIFF"})

    img_data = _read_upload(image)
    if img_data is None:
        return JSONResponse(status_code=413, content={"error": f"image exceeds {_MAX_MB} MB"})

    try:
        result = analyze_drone_image(
            image_bytes=img_data,
            parent_cell_id=parent_cell_id,
            lat=lat,
            lon=lon,
        )
        return result
    except Exception as exc:
        sat_logger.error(f"Drone inference error: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Drone analysis failed: {str(exc)}"})
