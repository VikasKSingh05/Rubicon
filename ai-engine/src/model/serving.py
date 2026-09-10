"""Serve-side inference: uploaded HSI/LiDAR files -> /predict contract body.

Reuses the Phase 2 preprocessing pipeline and the trained checkpoint + fitted
PCA. Predictions are per-patch left probabilities aggregated over the scene.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..preprocessing.load_hsi import load_hsi, apply_pca_to_patches
from ..preprocessing.load_lidar import load_lidar_raster, rasterize_las
from ..preprocessing.align import align_rasters
from ..preprocessing.patchify import patch_dataset
from ..preprocessing.dataset import normalize_dem
from ..model.factory import create_model


HOUSTON_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [-95.4, 29.8], [-95.3, 29.8], [-95.3, 29.9], [-95.4, 29.9], [-95.4, 29.8],
    ]],
}


def load_runtime(model_dir):
    """Load (model, pca, checkpoint). Raises FileNotFoundError when absent."""
    import pickle

    cp_path = Path(model_dir) / "model.pt"
    pca_path = Path(model_dir) / "pca.pkl"
    # Prefer the safe weights_only loader; fall back for legacy artifacts that
    # embed non-tensor python objects (e.g. numpy arrays). If both fail the
    # error propagates to callers (get_runtime) which degrade cleanly.
    try:
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    pca = pickle.loads(pca_path.read_bytes())

    cfg = checkpoint["config"]
    cfg["data"]["pca_components"] = checkpoint.get("pca_components", 0)
    model = create_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, pca, checkpoint


def _geojson_from_patches(hsi_transform, hsi_crs, patch_results, patch_size: int):
    """Envelope polygon (lon/lat) covering patches of the predicted class."""
    try:
        import rasterio as rio
        from rasterio.warp import transform_xy

        rows, cols = [], []
        for (r0, c0), (probs, pred_cls) in patch_results:
            if pred_cls == 0:  # only geometry of predicted-class ("issue") patches
                continue
            rows.append(r0 + patch_size / 2)
            cols.append(c0 + patch_size / 2)
        if not rows:
            return None
        xs, ys = rio.transform.xy(hsi_transform, rows, cols)
        if hsi_crs and hsi_crs != rio.CRS.from_epsg(4326):
            lons, lats = transform_xy(hsi_crs, rio.CRS.from_epsg(4326), xs, ys)
        else:
            lons, lats = xs, ys
        lon0, lon1 = min(lons), max(lons)
        lat0, lat1 = min(lats), max(lats)
        if math.isnan(lon0):
            return None
        ring = [
            [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
        ]
        return {"type": "Polygon", "coordinates": [ring]}
    except Exception:
        return None


def predict_files(hsi_path, lidar_path, model, pca, checkpoint):
    """Run the model over an uploaded scene and return the /predict body."""
    cfg = checkpoint["config"]
    pl = cfg.get("pipeline", {})
    patch_size = int(pl.get("patch_size", 32))

    hsi, hsi_t, hsi_crs = load_hsi(hsi_path)
    lidar = Path(lidar_path)
    if lidar.suffix.lower() in (".las", ".laz"):
        from rasterio.transform import array_bounds

        left, bottom, right, top = array_bounds(hsi.shape[0], hsi.shape[1], hsi_t)
        res = (right - left) / hsi.shape[1]
        dem, dem_t, dem_crs = rasterize_las(lidar, resolution=res, bounds=(left, right, bottom, top))
    else:
        dem, dem_t, dem_crs = load_lidar_raster(lidar)
    dem_align, hsi_t, hsi_crs = align_rasters(hsi, hsi_t, hsi_crs, dem, dem_t, dem_crs)

    labels = np.zeros(hsi.shape[:2], dtype=np.int32)
    patches = patch_dataset(
        hsi, dem_align, labels,
        patch_size=patch_size, overlap=0.0, min_labeled=0.0,
    )

    hsi_patches = apply_pca_to_patches(patches["hsi"], pca)        # (N, p, p, k)
    hsi_patches = hsi_patches.transpose(0, 3, 1, 2).astype(np.float32)
    dem_norm = normalize_dem(patches["dem"][..., None], *checkpoint["dem_scale"])
    dem_norm = dem_norm.transpose(0, 3, 1, 2).astype(np.float32)

    class_names = checkpoint["class_names"]
    patch_results = []
    with torch.no_grad():
        for i in range(len(hsi_patches)):
            x_hsi = torch.from_numpy(hsi_patches[i : i + 1])
            x_dem = torch.from_numpy(dem_norm[i : i + 1])
            logits = model(x_hsi, x_dem)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            patch_results.append((tuple(patches["coords"][i]), (probs, int(probs.argmax()))))

    mean_probs = np.mean([p for _, (p, _) in patch_results], axis=0)
    pred_cls = int(mean_probs.argmax())
    confidence = float(mean_probs[pred_cls])

    polygon = _geojson_from_patches(hsi_t, hsi_crs, patch_results, patch_size)
    polygon = polygon or HOUSTON_POLYGON

    return {
        "prediction": class_names[pred_cls],
        "confidence": round(confidence, 4),
        "class_probs": {name: round(float(mean_probs[i]), 4) for i, name in enumerate(class_names)},
        "geojson_polygon": polygon,
        "model_version": checkpoint["model_version"],
    }