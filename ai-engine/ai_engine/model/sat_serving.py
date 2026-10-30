"""Siamese U-Net serving: optical image pair → damage maps + H3 GeoJSON.

Supports:
  - PNG / JPEG: loaded with OpenCV, no CRS info (synthetic-mode lat/lon grid)
  - GeoTIFF: loaded with rasterio, full geospatial metadata used for H3 aggregation

Returns the /infer/satellite contract body defined in docs/api-contracts.md.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .siamese_unet import SiameseUNet
from .aggregator import aggregate_to_h3, cells_to_geojson


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

DAMAGE_CLASSES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


def _sha256(path: str | Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_image_cv2(path: str | Path) -> tuple[np.ndarray, None, None]:
    """Load an image as RGB uint8 (H, W, 3). Returns (array, None, None)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, None, None


def _load_image_geotiff(path: str | Path) -> tuple[np.ndarray, Any, Any]:
    """Load a GeoTIFF as RGB uint8 (H, W, 3). Returns (array, transform, crs)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    with rasterio.open(path) as src:
        # Read up to 3 bands (or fewer); treat as RGB
        n = min(src.count, 3)
        bands = src.read(list(range(1, n + 1)))  # (n, H, W)
        transform = src.transform
        crs = src.crs

        # Normalize each band to uint8
        result = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
        for i in range(n):
            b = bands[i].astype(np.float32)
            lo, hi = np.percentile(b[np.isfinite(b)], (2, 98))
            if hi > lo:
                b = np.clip((b - lo) / (hi - lo), 0, 1) * 255
            else:
                b = np.zeros_like(b)
            result[:, :, i] = b.astype(np.uint8)

    return result, transform, crs


def _preprocess(rgb: np.ndarray, crop_size: int | None = None) -> torch.Tensor:
    """Normalize and convert to (1, 3, H, W) tensor."""
    if crop_size is not None:
        h, w = rgb.shape[:2]
        if h != crop_size or w != crop_size:
            rgb = cv2.resize(rgb, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    img = rgb.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()  # (1,3,H,W)
    return tensor


def load_siamese_runtime(model_dir: str | Path) -> dict:
    """Load the Siamese U-Net checkpoint. Raises FileNotFoundError when absent."""
    cp_path = Path(model_dir) / "siamese_unet.pt"
    if not cp_path.exists():
        raise FileNotFoundError(cp_path)
    cp = torch.load(str(cp_path), map_location="cpu", weights_only=True)
    model = SiameseUNet(
        encoder_name=cp.get("encoder_name", "resnet34"),
        encoder_weights=None,
        num_classes=cp.get("num_classes", 5),
    )
    model.load_state_dict(cp["state_dict"])
    model.eval()
    return {"model": model, "checkpoint": cp}


def _is_geotiff(path: str | Path) -> bool:
    ext = Path(path).suffix.lower()
    return ext in (".tif", ".tiff")


def infer_satellite(
    pre_path: str | Path,
    post_path: str | Path,
    model: SiameseUNet,
    checkpoint: dict,
    device: torch.device | None = None,
    crop_size: int = 512,
    resolutions: tuple[int, ...] = (9, 10),
) -> dict:
    """Run Siamese U-Net over a pre/post image pair and return the contract body.

    Returns:
        {
          pre_sha256, post_sha256,
          buildings_geojson: GeoJSON FeatureCollection (building polygons),
          cells_geojson_r9: GeoJSON FeatureCollection (H3 res-9 cells),
          cells_geojson_r10: GeoJSON FeatureCollection (H3 res-10 cells),
          model_version, damage_classes, exec_ms
        }
    """
    t0 = time.perf_counter()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load images
    loader = _load_image_geotiff if _is_geotiff(pre_path) else _load_image_cv2
    pre_rgb, pre_transform, pre_crs = loader(pre_path)
    post_rgb, _, _ = loader(post_path)

    # Compute SHA-256 of input files
    pre_sha256 = _sha256(pre_path)
    post_sha256 = _sha256(post_path)

    H_orig, W_orig = pre_rgb.shape[:2]

    # Preprocess
    pre_t = _preprocess(pre_rgb, crop_size).to(device)
    post_t = _preprocess(post_rgb, crop_size).to(device)

    if pre_t.is_cuda:
        pre_t = pre_t.to(memory_format=torch.channels_last)
        post_t = post_t.to(memory_format=torch.channels_last)
        model = model.to(device).to(memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # Inference
    with torch.no_grad():
        loc_logits, dmg_logits = model(pre_t, post_t)

    loc_prob = torch.sigmoid(loc_logits)[0, 0].cpu().numpy()   # (H, W)
    dmg_pred = dmg_logits[0].argmax(dim=0).cpu().numpy()        # (H, W), values 0-4
    dmg_conf = torch.softmax(dmg_logits[0], dim=0).max(dim=0).values.cpu().numpy()

    # Resize outputs back to original resolution (for pixel-accurate reporting)
    if (loc_prob.shape[0] != H_orig or loc_prob.shape[1] != W_orig):
        dmg_pred_full = cv2.resize(
            dmg_pred.astype(np.uint8), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST
        )
        conf_full = cv2.resize(dmg_conf.astype(np.float32), (W_orig, H_orig))
    else:
        dmg_pred_full = dmg_pred.astype(np.uint8)
        conf_full = dmg_conf.astype(np.float32)

    # Building polygon FeatureCollection (contour extraction from loc_mask)
    loc_bin = (loc_prob > 0.5).astype(np.uint8)
    if loc_bin.shape[:2] != (H_orig, W_orig):
        loc_bin = cv2.resize(loc_bin, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

    buildings_fc = _extract_building_geojson(loc_bin, dmg_pred_full, pre_transform, pre_crs)

    # H3 aggregation
    h3_results = aggregate_to_h3(
        dmg_mask=dmg_pred_full,
        conf_map=conf_full,
        transform=pre_transform,
        crs=pre_crs,
        resolutions=resolutions,
        patch_size=32,
    )

    exec_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "pre_sha256": pre_sha256,
        "post_sha256": post_sha256,
        "buildings_geojson": buildings_fc,
        "model_version": checkpoint.get("model_version", "siamese_unet-v1"),
        "damage_classes": DAMAGE_CLASSES,
        "exec_ms": exec_ms,
    }

    for res in resolutions:
        result[f"cells_geojson_r{res}"] = cells_to_geojson(h3_results[res])

    return result


def _extract_building_geojson(
    loc_bin: np.ndarray,
    dmg_mask: np.ndarray,
    transform,
    crs,
) -> dict:
    """Convert building localization mask to GeoJSON FeatureCollection."""
    features = []
    contours, _ = cv2.findContours(loc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 9:  # skip tiny noise
            continue
        cnt_squeezed = cnt.squeeze(1)  # (N, 2) pixel xy
        if len(cnt_squeezed) < 3:
            continue

        # Sample damage class inside the contour
        mask_bld = np.zeros(loc_bin.shape, dtype=np.uint8)
        cv2.drawContours(mask_bld, [cnt], -1, 1, -1)
        bld_pixels = dmg_mask[mask_bld > 0]
        counts = np.bincount(bld_pixels.flatten(), minlength=5)
        dom_cls = int(counts[1:].argmax()) + 1 if counts[1:].sum() > 0 else 0

        severity_map = {0: "GREY", 1: "GREEN", 2: "YELLOW", 3: "ORANGE", 4: "RED"}

        # Convert pixel coords to geographic coords
        latlons = []
        for px, py in cnt_squeezed:
            if transform is not None:
                import rasterio
                xs, ys = rasterio.transform.xy(transform, [py], [px])
                if crs and crs.to_epsg() != 4326:
                    from rasterio.warp import transform as warp_transform
                    lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
                    latlons.append([float(lons[0]), float(lats[0])])
                else:
                    latlons.append([float(xs[0]), float(ys[0])])
            else:
                # No geo reference: pixel coords as synthetic stand-in
                latlons.append([float(px), float(py)])

        if len(latlons) < 3:
            continue
        latlons.append(latlons[0])  # close ring

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [latlons]},
            "properties": {
                "dominant_class": dom_cls,
                "damage_label": DAMAGE_CLASSES[dom_cls],
                "severity": severity_map[dom_cls],
                "area_px": int(cv2.contourArea(cnt)),
            },
        })

    return {"type": "FeatureCollection", "features": features}
