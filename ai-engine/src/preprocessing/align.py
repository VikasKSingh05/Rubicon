"""Co-register a LiDAR DEM onto the HSI grid using shared georeferencing.

Houston 2013 ships HSI and LiDAR/DSM in the same UTM zone, but grid resolutions
can differ. When both rasters carry a CRS, the DEM is warped (reprojected) onto
the exact HSI grid via rasterio.warp.reproject. When georeferencing is missing,
both rasters are assumed to share the same pixel grid (sanity-checked).
"""
from __future__ import annotations

import numpy as np

_NODATA = np.nan


def align_rasters(hsi, hsi_transform, hsi_crs, dem, dem_transform, dem_crs, nodata=_NODATA):
    """Reproject `dem` onto the HSI grid.

    Returns (dem_aligned, hsi_transform, hsi_crs): the DEM reshaped to the HSI
    grid height/width, plus the grid metadata reused downstream.
    """
    import rasterio as rio
    from rasterio.warp import reproject, Resampling

    if hsi_crs and dem_crs:
        dst = np.full(hsi.shape[:2], nodata, dtype=np.float32)
        reproject(
            source=dem,
            destination=dst,
            src_transform=dem_transform,
            src_crs=dem_crs,
            dst_transform=hsi_transform,
            dst_crs=hsi_crs,
            src_nodata=nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
        return dst, hsi_transform, hsi_crs

    if dem.shape != hsi.shape[:2]:
        raise ValueError(
            f"no CRS available and grids differ: DEM {dem.shape} vs HSI {hsi.shape[:2]}"
        )
    return dem.astype(np.float32), hsi_transform, hsi_crs


def assert_aligned(hsi, dem, mask=None):
    """Assert the DEM grid matches the HSI grid (used for tests and sanity checks)."""
    if dem.shape != hsi.shape[:2]:
        raise ValueError(f"misaligned grids: DEM {dem.shape} vs HSI {hsi.shape[:2]}")
    valid = np.isfinite(dem)
    if valid.any():
        dem_vals = dem[valid]
        if dem_vals.min() < -10000 or dem_vals.max() > 100000:
            raise ValueError("suspicious DEM height range")