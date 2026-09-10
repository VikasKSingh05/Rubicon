"""Load LiDAR for Houston 2013: raw .las rasterization or a pre-rasterized DSM.

The GRSS 2013 distribution ships a LiDAR-derived DSM raster; raw .las point
clouds are rasterized on the fly when provided (laspy path required).
"""
from __future__ import annotations

import numpy as np

DEFAULT_RESOLUTION = 2.5  # meters per cell (Houston HSI ground sample distance)


def load_lidar_raster(path):
    """Load a pre-rasterized LiDAR height grid (.tif). Returns (dem, transform, crs)."""
    import rasterio

    with rasterio.open(path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
    dem[np.isnan(dem)] = np.nan
    return dem, transform, crs


def rasterize_las(path, resolution: float = DEFAULT_RESOLUTION, bounds=None):
    """Rasterize a .las point cloud into a 2D height DEM (max-z per cell).

    Parameters
    ----------
    path : PathLike
        Input .las / .laz file.
    resolution : float
        Output cell size (meters). 2.5 matches the Houston HSI GSD.
    bounds : (xmin, xmax, ymin, ymax) or None
        Clipping bounds; defaults to the full point cloud extent.

    Returns
    -------
    (dem, transform, None)
        dem is (rows, cols) float32 of NaN-backed heights; transform is the
        rasterio Affine mapping grid -> world coordinates. CRS is None because
        laspy does not always expose it; datasets coordinate the grids later.
    """
    import laspy
    from rasterio import Affine

    las = laspy.read(path)
    x = np.asarray(las.x)
    y = np.asarray(las.y)
    z = np.asarray(las.z)

    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
    else:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())

    cols = max(1, int(np.ceil((xmax - xmin) / resolution)))
    rows = max(1, int(np.ceil((ymax - ymin) / resolution)))

    dem = np.full((rows, cols), np.nan, dtype=np.float32)
    col_idx = np.clip(((x - xmin) / resolution).astype(np.int64), 0, cols - 1)
    row_idx = np.clip(((ymax - y) / resolution).astype(np.int64), 0, rows - 1)
    inside = (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & np.isfinite(z)
    )
    if inside.any():
        # fmax ignores the NaN seed (unlike maximum) so first-return z survives
        np.fmax.at(dem, (row_idx[inside], col_idx[inside]), z[inside])

    transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    return dem, transform, None