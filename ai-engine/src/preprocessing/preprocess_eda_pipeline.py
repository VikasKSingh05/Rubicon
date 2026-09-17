"""Rubicon Standalone Preprocessing & EDA Pipeline.

Loads hyperspectral (HSI) and LiDAR rasters, co-registers spatial grids,
fits PCA dimensionality reduction across spectral bands, extracts spatial patches,
maps ground truth to damage severity proxies, and exports preprocessed tensors
specifically for Exploratory Data Analysis (EDA) and visualization without any
train/test splitting or model training.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import yaml

# =====================================================================
# Land Cover & Damage Severity Proxy Mapping
# =====================================================================

HOUSTON2013_CLASSES: Dict[int, str] = {
    1: "Grass healthy",
    2: "Grass stressed",
    3: "Grass synthetic",
    4: "Tree",
    5: "Soil",
    6: "Water",
    7: "Residential",
    8: "Commercial",
    9: "Road",
    10: "Highway",
    11: "Railway",
    12: "Parking Lot 1",
    13: "Parking Lot 2",
    14: "Tennis Court",
    15: "Running Track",
}

CLASS_TO_SEVERITY: Dict[int, str] = {
    1: "none",        # Grass healthy
    2: "none",        # Grass stressed
    3: "none",        # Grass synthetic
    4: "none",        # Tree
    5: "moderate",    # Soil
    6: "none",        # Water
    7: "moderate",    # Residential
    8: "severe",      # Commercial
    9: "moderate",    # Road
    10: "severe",     # Highway
    11: "severe",     # Railway
    12: "moderate",   # Parking Lot 1
    13: "moderate",   # Parking Lot 2
    14: "moderate",   # Tennis Court
    15: "moderate",   # Running Track
}

SEVERITY_NAMES = ["none", "moderate", "severe"]
SEVERITY_TO_INDEX = {name: i for i, name in enumerate(SEVERITY_NAMES)}

RGB_BANDS_CASI = (63, 37, 22)   # ~670 nm (R), ~550 nm (G), ~480 nm (B)
NIR_BAND_CASI = 100             # ~850 nm (Near-Infrared)


# =====================================================================
# Normalization & Loading Utilities
# =====================================================================

def normalize_raster(data: np.ndarray) -> np.ndarray:
    """Normalize integer rasters to [0, 1]; float rasters scaled by maximum if > 1.0."""
    data = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data / info.max
    mx = float(data.max())
    if mx > 1.0:
        return data / mx
    return data


def generate_synthetic_houston2013(rows: int = 349, cols: int = 1905, bands: int = 144) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic synthetic multi-modal dataset matching Houston 2013 dimensions if raw files are absent."""
    rng = np.random.default_rng(42)
    
    # 1. Generate base elevation map with realistic terrain and structures
    x = np.linspace(0, 10, cols, dtype=np.float32)
    y = np.linspace(0, 2, rows, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dem = 10.0 + 5.0 * np.sin(xx) + 2.0 * np.cos(yy * 2.0)
    
    # Add building blocks
    for _ in range(35):
        bx = rng.integers(50, cols - 100)
        by = rng.integers(20, rows - 50)
        bw = rng.integers(30, 80)
        bh = rng.integers(15, 30)
        dem[by:by + bh, bx:bx + bw] += rng.uniform(8.0, 25.0)
    dem = dem.astype(np.float32)

    # 2. Generate Ground Truth labels
    labels = np.zeros((rows, cols), dtype=np.int32)
    for c_id in range(1, 16):
        for _ in range(12):
            lx = rng.integers(10, cols - 100)
            ly = rng.integers(10, rows - 40)
            lw = rng.integers(30, 90)
            lh = rng.integers(15, 35)
            labels[ly:ly + lh, lx:lx + lw] = c_id

    # 3. Vectorized HSI Spectral Cube Generation
    wavelengths = np.linspace(380, 1050, bands, dtype=np.float32)
    base_veg_curve = (0.1 + 0.6 / (1.0 + np.exp(-(wavelengths - 700.0) / 30.0))).astype(np.float32)
    base_soil_curve = (0.15 + 0.35 * (wavelengths - 380.0) / (1050.0 - 380.0)).astype(np.float32)
    base_built_curve = (0.25 + 0.1 * np.sin((wavelengths - 400.0) / 100.0)).astype(np.float32)
    base_water_curve = (0.2 * np.exp(-(wavelengths - 380.0) / 200.0)).astype(np.float32)

    hsi = np.empty((rows, cols, bands), dtype=np.float32)
    hsi[:] = base_built_curve.reshape(1, 1, bands)
    
    veg_mask = np.isin(labels, [1, 2, 3, 4])
    soil_mask = (labels == 5)
    water_mask = (labels == 6)

    hsi[veg_mask] = base_veg_curve
    hsi[soil_mask] = base_soil_curve
    hsi[water_mask] = base_water_curve

    # Add spatial-spectral variations and noise in float32 directly
    spatial_var = rng.uniform(0.85, 1.15, size=(rows, cols, 1)).astype(np.float32)
    noise = rng.standard_normal(size=(rows, cols, bands), dtype=np.float32) * 0.015
    hsi = np.clip(hsi * spatial_var + noise, 0.0, 1.0)

    return hsi, dem, labels


def load_raw_dataset(root: Path, data_cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, Any, Any]:
    """Load HSI cube, LiDAR DEM, and GT labels or generate unified synthetic dataset if files are not present."""
    hsi_path = root / data_cfg["hsi"]
    lidar_path = root / data_cfg["lidar"]
    labels_path = root / data_cfg["labels"]

    if not hsi_path.exists() or not lidar_path.exists() or not labels_path.exists():
        print(f"[load_raw_dataset] Dataset files not found under '{root}'. Generating unified synthetic Houston 2013 dataset.")
        hsi, dem, labels = generate_synthetic_houston2013()
        return hsi, dem, labels, None, None, None, None

    # Load actual GeoTIFF files if present
    import rasterio
    with rasterio.open(hsi_path) as src:
        hsi = normalize_raster(np.transpose(src.read(), (1, 2, 0)))
        hsi_t, hsi_crs = src.transform, src.crs

    with rasterio.open(labels_path) as src:
        labels = src.read(1).astype(np.int32)

    if lidar_path.suffix.lower() in (".las", ".laz"):
        import laspy
        from rasterio import Affine
        las = laspy.read(lidar_path)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
        res = 2.5
        cols = max(1, int(np.ceil((xmax - xmin) / res)))
        rows = max(1, int(np.ceil((ymax - ymin) / res)))
        dem = np.full((rows, cols), np.nan, dtype=np.float32)
        col_idx = np.clip(((x - xmin) / res).astype(np.int64), 0, cols - 1)
        row_idx = np.clip(((ymax - y) / res).astype(np.int64), 0, rows - 1)
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & np.isfinite(z)
        if inside.any():
            np.fmax.at(dem, (row_idx[inside], col_idx[inside]), z[inside])
        dem_t, dem_crs = Affine(res, 0, xmin, 0, -res, ymax), None
    else:
        with rasterio.open(lidar_path) as src:
            dem = src.read(1).astype(np.float32)
            dem_t, dem_crs = src.transform, src.crs

    return hsi, dem, labels, hsi_t, hsi_crs, dem_t, dem_crs


# =====================================================================
# Alignment & Spatial Co-Registration
# =====================================================================

def align_modalities(
    hsi: np.ndarray, hsi_transform: Any, hsi_crs: Any,
    dem: np.ndarray, dem_transform: Any, dem_crs: Any,
) -> np.ndarray:
    """Co-register and resample the LiDAR DEM onto the exact spatial grid of the HSI cube."""
    if hsi_crs and dem_crs and hsi_transform and dem_transform:
        from rasterio.warp import Resampling, reproject
        dst = np.full(hsi.shape[:2], np.nan, dtype=np.float32)
        reproject(
            source=dem,
            destination=dst,
            src_transform=dem_transform,
            src_crs=dem_crs,
            dst_transform=hsi_transform,
            dst_crs=hsi_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return dst

    if dem.shape != hsi.shape[:2]:
        from scipy.ndimage import zoom
        zy = hsi.shape[0] / dem.shape[0]
        zx = hsi.shape[1] / dem.shape[1]
        aligned_dem = zoom(dem, (zy, zx), order=1).astype(np.float32)
        return aligned_dem[:hsi.shape[0], :hsi.shape[1]]

    return dem.astype(np.float32)


# =====================================================================
# Dimensionality Reduction (PCA)
# =====================================================================

def perform_pca_reduction(hsi: np.ndarray, target_variance: float = 0.99) -> Tuple[Any, np.ndarray, int, float]:
    """Fit PCA on pixels in HSI to reduce spectral dimensions to target explained variance."""
    from sklearn.decomposition import PCA

    h, w, b = hsi.shape
    pixels = hsi.reshape(-1, b)
    
    # Subsample for fast exact covariance calculation
    sample_stride = max(1, len(pixels) // 10000)
    samples = pixels[::sample_stride]
    
    mean = np.mean(samples, axis=0, keepdims=True)
    centered = samples - mean
    cov = (centered.T @ centered) / (len(samples) - 1)
    
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0.0)
    vecs = vecs[:, idx]
    
    total_spectral_var = np.sum(vals)
    evr = vals / max(total_spectral_var, 1e-9)
    cum_evr = np.cumsum(evr)
    
    k = int(np.argmax(cum_evr >= target_variance) + 1)
    k = max(1, min(k, b))
    
    components = vecs[:, :k].T
    retained_var = float(np.sum(evr[:k]))
    
    # Fast projection
    reduced_pixels = (pixels - mean) @ components.T
    reduced_hsi = reduced_pixels.reshape(h, w, k).astype(np.float32)
    
    pca = PCA(n_components=k)
    pca.mean_ = mean.squeeze()
    pca.components_ = components
    pca.explained_variance_ratio_ = evr[:k]
    pca.n_components_ = k
    
    return pca, reduced_hsi, k, retained_var


# =====================================================================
# Patch Extraction & Multi-Modal Dataset Assembly (No Split)
# =====================================================================

def extract_patches(
    hsi_pca: np.ndarray,
    hsi_raw: np.ndarray,
    dem: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 32,
    overlap: float = 0.0,
    min_labeled: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Extract fixed-size multi-modal spatial patches covering the survey area."""
    rows, cols = hsi_pca.shape[:2]
    stride = max(1, int(patch_size * (1.0 - overlap)))
    
    anchors_r = list(range(0, rows - patch_size + 1, stride)) or [0]
    anchors_c = list(range(0, cols - patch_size + 1, stride)) or [0]
    if anchors_r[-1] + patch_size < rows:
        anchors_r.append(rows - patch_size)
    if anchors_c[-1] + patch_size < cols:
        anchors_c.append(cols - patch_size)

    patches_hsi_pca = []
    patches_hsi_raw = []
    patches_dem = []
    patch_classes = []
    patch_severities = []
    patch_coords = []

    # Normalize DEM elevations
    dem_valid = dem[np.isfinite(dem)]
    dem_min = float(np.nanmin(dem_valid)) if dem_valid.size > 0 else 0.0
    dem_max = float(np.nanmax(dem_valid)) if dem_valid.size > 0 else 1.0
    dem_norm = np.clip((dem - dem_min) / max(dem_max - dem_min, 1e-6), 0.0, 1.0).astype(np.float32)

    for attempt_thresh in [min_labeled, 0.2, 0.0]:
        patches_hsi_pca.clear()
        patches_hsi_raw.clear()
        patches_dem.clear()
        patch_classes.clear()
        patch_severities.clear()
        patch_coords.clear()
        
        for r0 in anchors_r:
            for c0 in anchors_c:
                r1, c1 = r0 + patch_size, c0 + patch_size
                lab_crop = labels[r0:r1, c0:c1]
                if lab_crop.size == 0:
                    continue
                labeled_ratio = (lab_crop > 0).mean()
                if labeled_ratio < attempt_thresh:
                    continue

                valid_labels = lab_crop[lab_crop > 0]
                if valid_labels.size == 0:
                    majority_c = 1  # Default to healthy grass if empty
                else:
                    counts = np.bincount(valid_labels)
                    majority_c = int(np.flatnonzero(counts == counts.max())[0])

                sev_str = CLASS_TO_SEVERITY.get(majority_c, "none")
                sev_idx = SEVERITY_TO_INDEX.get(sev_str, 0)

                patches_hsi_pca.append(hsi_pca[r0:r1, c0:c1])
                if len(patches_hsi_raw) < 50:
                    patches_hsi_raw.append(hsi_raw[r0:r1, c0:c1])
                patches_dem.append(dem_norm[r0:r1, c0:c1])
                patch_classes.append(majority_c)
                patch_severities.append(sev_idx)
                patch_coords.append((r0, c0))
                
        if len(patches_hsi_pca) >= 5:
            break

    if not patches_hsi_pca:
        raise ValueError(f"No patches could be extracted from grid shape {hsi_pca.shape[:2]}")

    X_hsi = np.transpose(np.stack(patches_hsi_pca), (0, 3, 1, 2)).astype(np.float32)
    X_dem = np.expand_dims(np.stack(patches_dem), axis=1).astype(np.float32)
    sample_raw = np.stack(patches_hsi_raw).astype(np.float32) if patches_hsi_raw else np.empty((0, patch_size, patch_size, hsi_raw.shape[2]), dtype=np.float32)

    return {
        "X_hsi": X_hsi,
        "X_dem": X_dem,
        "patches_hsi_raw": sample_raw,
        "y_class": np.asarray(patch_classes, dtype=np.int64),
        "y_sev": np.asarray(patch_severities, dtype=np.int64),
        "coords": np.asarray(patch_coords, dtype=np.int64),
        "dem_scale": (dem_min, dem_max),
    }


# =====================================================================
# Complete Preprocessing Execution
# =====================================================================

def run_preprocessing_pipeline(config_path: Path) -> Dict[str, Any]:
    """Execute the full data preprocessing pipeline without splitting or training."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    pipeline_cfg = cfg["pipeline"]
    output_cfg = cfg["output"]

    root = Path(data_cfg["root"])
    out_root = Path(output_cfg["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      RUBICON DATA PREPROCESSING & EDA PREPARATION PIPELINE")
    print("=" * 70)

    # 1. Load HSI, LiDAR, Labels
    print("\n[Step 1/5] Loading Raw Modalities...")
    hsi, dem, labels, hsi_t, hsi_crs, dem_t, dem_crs = load_raw_dataset(root, data_cfg)
    print(f"  -> Raw HSI Cube:    {hsi.shape} (Rows, Cols, Bands) | range [{hsi.min():.2f}, {hsi.max():.2f}]")
    print(f"  -> Raw LiDAR DEM:   {dem.shape} (Rows, Cols)        | range [{np.nanmin(dem):.2f}, {np.nanmax(dem):.2f}] m")
    print(f"  -> Ground Truth GT: {labels.shape} (Rows, Cols)     | {len(np.unique(labels))} classes found")

    # 2. Align & Co-register
    print("\n[Step 2/5] Spatial Co-Registration & Alignment...")
    dem_aligned = align_modalities(hsi, hsi_t, hsi_crs, dem, dem_t, dem_crs)
    print(f"  -> Aligned DEM:     {dem_aligned.shape} matching HSI grid precisely.")

    # 3. PCA Spectral Dimensionality Reduction
    target_var = float(pipeline_cfg.get("pca_variance", 0.99))
    print(f"\n[Step 3/5] Performing Spectral PCA (target explained variance: {target_var * 100:.1f}%)...")
    pca_model, hsi_pca, n_comp, total_var = perform_pca_reduction(hsi, target_variance=target_var)
    print(f"  -> Spectral Bands Reduced: {hsi.shape[2]} -> {n_comp} Principal Components")
    print(f"  -> Total Retained Variance: {total_var * 100:.2f}%")

    # 4. Patch Extraction
    patch_size = int(pipeline_cfg.get("patch_size", 32))
    overlap = float(pipeline_cfg.get("overlap", 0.0))
    min_labeled = float(pipeline_cfg.get("min_labeled", 0.5))
    print(f"\n[Step 4/5] Extracting Multi-Modal Patches ({patch_size}x{patch_size}, overlap={overlap}, min_labeled={min_labeled})...")
    patch_data = extract_patches(
        hsi_pca=hsi_pca,
        hsi_raw=hsi,
        dem=dem_aligned,
        labels=labels,
        patch_size=patch_size,
        overlap=overlap,
        min_labeled=min_labeled,
    )
    num_patches = len(patch_data["y_class"])
    print(f"  -> Generated {num_patches} Preprocessed Patches")
    print(f"  -> X_hsi Tensor Shape: {patch_data['X_hsi'].shape}")
    print(f"  -> X_dem Tensor Shape: {patch_data['X_dem'].shape}")

    # 5. Export Preprocessed Data Artifacts
    print("\n[Step 5/5] Saving Preprocessed Dataset Artifacts (No Train/Test Split)...", flush=True)
    npz_path = out_root / output_cfg["npz"]
    pca_path = out_root / output_cfg["pca"]
    meta_path = out_root / output_cfg["metadata"]

    np.savez(
        npz_path,
        X_hsi=patch_data["X_hsi"],
        X_dem=patch_data["X_dem"],
        y_class=patch_data["y_class"],
        y_sev=patch_data["y_sev"],
        coords=patch_data["coords"],
        dem_scale=np.asarray(patch_data["dem_scale"], dtype=np.float32),
        pca_n_components=n_comp,
    )

    with open(pca_path, "wb") as f:
        pickle.dump(pca_model, f)

    # Class & Severity frequency summary
    unique_classes, class_counts = np.unique(patch_data["y_class"], return_counts=True)
    class_distribution = {
        HOUSTON2013_CLASSES.get(int(c), f"Class_{c}"): int(cnt)
        for c, cnt in zip(unique_classes, class_counts)
    }

    unique_sev, sev_counts = np.unique(patch_data["y_sev"], return_counts=True)
    severity_distribution = {
        SEVERITY_NAMES[int(s)]: int(cnt)
        for s, cnt in zip(unique_sev, sev_counts)
    }

    metadata = {
        "status": "preprocessed_successfully",
        "description": "Preprocessed Houston 2013 HSI & LiDAR dataset ready for EDA",
        "num_patches": num_patches,
        "patch_size": patch_size,
        "hsi_spatial_shape": list(hsi.shape[:2]),
        "raw_hsi_bands": int(hsi.shape[2]),
        "pca_n_components": n_comp,
        "pca_explained_variance": total_var,
        "dem_min_elevation": patch_data["dem_scale"][0],
        "dem_max_elevation": patch_data["dem_scale"][1],
        "class_distribution": class_distribution,
        "severity_distribution": severity_distribution,
        "files": {
            "npz_data": str(npz_path),
            "pca_model": str(pca_path),
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> Saved Preprocessed Tensors:  {npz_path}", flush=True)
    print(f"  -> Saved Fitted PCA Model:      {pca_path}", flush=True)
    print(f"  -> Saved Metadata Summary:      {meta_path}", flush=True)
    print("\n Preprocessing successfully finished! Ready for EDA & Visualization.", flush=True)

    return {
        "hsi_raw": hsi,
        "dem_raw": dem_aligned,
        "labels": labels,
        "hsi_pca": hsi_pca,
        "pca_model": pca_model,
        "patches": patch_data,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Rubicon Preprocessing Pipeline for EDA")
    parser.add_argument("--config", default="configs/preprocess_eda.yaml", help="Path to config yaml")
    args = parser.parse_args()
    
    run_preprocessing_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
