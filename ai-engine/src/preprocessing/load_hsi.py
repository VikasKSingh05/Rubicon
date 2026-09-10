"""Load Houston 2013 hyperspectral cube (.tiff) and reduce it with PCA."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize(data: np.ndarray) -> np.ndarray:
    """Normalize integer rasters to [0, 1]; floats scaled by their max if needed."""
    data = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data / info.max
    mx = float(data.max())
    if mx > 1.0:
        return data / mx
    return data


def load_hsi(path):
    """Load a hyperspectral .tiff cube.

    Returns
    -------
    (data, transform, crs)
        data is (height, width, bands) float32 in [0, 1]; transform/crs are the
        rasterio affine transform and CRS (or None).
    """
    import rasterio

    with rasterio.open(path) as src:
        bands = src.read()
        transform = src.transform
        crs = src.crs
    data = np.transpose(bands, (1, 2, 0))
    return _normalize(data), transform, crs


def load_labels(path):
    """Load an integer label/ground-truth raster. Returns (labels, transform, crs)."""
    import rasterio

    with rasterio.open(path) as src:
        labels = src.read(1)
        transform = src.transform
        crs = src.crs
    return labels.astype(np.int32), transform, crs


def fit_pca(x: np.ndarray, variance: float = 0.99):
    """Fit PCA on (n_samples, n_features) to retain `variance` of explained variance.

    Returns (pca, reduced) where reduced shape is (n_samples, k). The fitted PCA
    should be pickled and reused at inference time.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=variance)
    reduced = pca.fit_transform(x)
    return pca, reduced


def transform_pca(x: np.ndarray, pca):
    """Apply an already-fitted PCA to (n_samples, n_features)."""
    return pca.transform(x)


def build_pca_from_flat(pixels: np.ndarray, variance: float) -> "tuple":
    """Fit PCA on a (N, B) pixel stack; returns (pca, reduced, k)."""
    pca, reduced = fit_pca(pixels, variance)
    return pca, reduced, int(pca.n_components_)


def apply_pca_to_patches(patches: np.ndarray, pca) -> np.ndarray:
    """PCA-reduce a batch of (N, H, W, B) HSI patches in place of the band axis.

    Returns (N, H, W, k) float32.
    """
    n, h, w, b = patches.shape
    flat = patches.reshape(n * h * w, b)
    reduced = pca.transform(flat).astype(np.float32)
    return reduced.reshape(n, h, w, reduced.shape[1])


def example_paths(root) -> dict:
    """Convenience for notebooks: returns absolute example file paths under root."""
    return {
        "hsi": Path(root) / "Houston_2013_HSI.tif",
        "lidar": Path(root) / "Houston_2013_DSM.tif",
        "labels": Path(root) / "Houston_2013_GT.tif",
    }