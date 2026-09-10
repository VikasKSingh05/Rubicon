"""Phase 2 preprocessing tests: PCA, LiDAR rasterization, alignment, patchify,
and a full synthetic build_dataset run that exercises the 70/15/15 split."""
import numpy as np
import pytest

from src.preprocessing.load_hsi import fit_pca, apply_pca_to_patches
from src.preprocessing.load_lidar import rasterize_las
from src.preprocessing.align import align_rasters
from src.preprocessing.patchify import patch_dataset, majority_label, sliding_windows
from src.preprocessing.dataset import build_dataset, save_dataset


def test_fit_pca_retains_target_variance():
    rng = np.random.default_rng(0)
    # structured data: 8 latent factors + noise -> fast variance saturation
    latent = rng.normal(size=(1000, 8))
    x = latent @ rng.normal(size=(8, 30)) + rng.normal(0, 0.1, size=(1000, 30))
    pca, reduced = fit_pca(x, 0.99)
    assert pca.explained_variance_ratio_.sum() >= 0.99 - 1e-6
    assert reduced.shape[1] <= 8
    assert reduced.shape[0] == 1000


def test_apply_pca_reduces_band_axis():
    rng = np.random.default_rng(1)
    patches = rng.normal(size=(8, 4, 5, 20))
    pca, _ = fit_pca(patches.reshape(-1, 20), 0.99)
    out = apply_pca_to_patches(patches, pca)
    assert out.shape == (8, 4, 5, pca.n_components_)


def test_rasterize_las_grid(tmp_path):
    import laspy

    path = tmp_path / "scene.las"
    las = laspy.create(point_format=laspy.PointFormat(2))
    las.x = np.array([0.0, 5.0, 10.0, 25.0])
    las.y = np.array([0.0, 0.0, 0.0, 0.0])
    las.z = np.array([1.0, 2.0, 3.0, 9.0])
    las.write(path)

    dem, transform, crs = rasterize_las(path, resolution=5, bounds=(0, 30, 0, 10))
    assert crs is None
    assert dem.shape == (2, 6)
    # y=0 lands in the bottom row (row 1); x=25 -> col 5
    assert dem[1, 5] == pytest.approx(9.0)
    assert dem[1, 2] == pytest.approx(3.0)  # (10, 0) -> col 2
    assert dem[1, 1] == pytest.approx(2.0)  # (5, 0) -> col 1
    assert np.isnan(dem[0, :]).all()        # top row covers y in [5, 10)
    assert transform.a == 5.0
    assert transform.e == -5.0
    assert transform.f == 10.0


def test_align_different_grids_warp():
    import rasterio as rio
    from rasterio.transform import Affine

    hsi = np.zeros((10, 12))
    dem = np.full((5, 6), 5.0)
    t_hsi = Affine(1.0, 0, 0, 0, -1.0, 10)
    t_dem = Affine(2.0, 0, 0, 0, -2.0, 10)
    crs = rio.CRS.from_epsg(4326)

    aligned, t, c = align_rasters(hsi, t_hsi, crs, dem, t_dem, crs)
    assert aligned.shape == (10, 12)
    assert np.allclose(aligned, 5.0)


def test_align_fallback_raises_on_grid_mismatch():
    dem = np.zeros((5, 6))
    hsi = np.zeros((10, 12))
    with pytest.raises(ValueError):
        align_rasters(hsi, None, None, dem, None, None)


def test_sliding_windows_counts():
    assert sum(1 for _ in sliding_windows((64, 64), 32, 0.0)) == 4
    # overlapping: more windows
    assert sum(1 for _ in sliding_windows((64, 64), 32, 0.5)) == 9
    # uneven grid anchors the bottom-right corner
    windows = list(sliding_windows((70, 70), 32, 0.0))
    assert (70 - 32, 70 - 32, 70, 70) in windows


def test_majority_label():
    labels = np.array([[3, 3, 0], [3, 1, 0], [0, 0, 0]])
    assert majority_label(labels) == 3


def test_patch_dataset_counts_and_labels():
    rng = np.random.default_rng(2)
    hsi = rng.normal(size=(64, 64, 8))
    dem = rng.normal(size=(64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[:32, :32] = 1
    labels[32:, 32:] = 8

    patches = patch_dataset(hsi, dem, labels, patch_size=32, overlap=0.0, min_labeled=0.5)
    assert patches["hsi"].shape == (2, 32, 32, 8)
    assert patches["dem"].shape == (2, 32, 32)
    assert set(patches["y_class"].tolist()) == {1, 8}


def test_patch_dataset_filters_background_only():
    labels = np.zeros((32, 32), dtype=np.int32)  # no labels anywhere
    with pytest.raises(ValueError):
        patch_dataset(np.zeros((32, 32, 3)), np.zeros((32, 32)), labels)


def _synthetic_scene():
    """128x128 scene, 4x 64x64 pure-class blocks, 20 HSI bands.

    Quadrants (TL/TR/BL/BR) give 4 none, 8 moderate, 4 severe patches so the
    stratified 70/15/15 split always sees every severity in the train set.
    """
    rng = np.random.default_rng(7)
    h = w = 128
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:64, :64] = 1   # grass healthy  -> none
    labels[:64, 64:] = 7   # residential    -> moderate
    labels[64:, :64] = 5   # soil           -> moderate
    labels[64:, 64:] = 8   # commercial     -> severe
    b = 20
    base = rng.normal(size=(15, b)) * 30000 + 30000
    x = np.zeros((h, w, b))
    for c in range(1, 16):
        m = labels == c
        if m.any():
            x[m] = base[c - 1]
    x = np.clip(x, 0, 65535).astype("uint16")
    dem = np.full((h, w), 10.0, dtype=np.float32)
    dem[labels == 1] = 12.0
    dem[labels == 8] = 7.5
    return x, dem, labels


def _write_geotiffs(tmp_path, x, dem, labels):
    import rasterio as rio
    from rasterio.transform import Affine

    t = Affine(2.5, 0, 0, 0, -2.5, 0)
    crs = "EPSG:32615"

    with rio.open(
        tmp_path / "hsi.tif", "w", driver="GTiff",
        height=x.shape[0], width=x.shape[1], count=x.shape[2],
        dtype="uint16", crs=crs, transform=t,
    ) as dst:
        dst.write(np.moveaxis(x, 2, 0))
    for name, arr, dtype in (
        ("dem.tif", dem, "float32"),
        ("gt.tif", labels, "int16"),
    ):
        with rio.open(
            tmp_path / name, "w", driver="GTiff",
            height=arr.shape[0], width=arr.shape[1], count=1,
            dtype=dtype, crs=crs, transform=t,
        ) as dst:
            dst.write(arr, 1)


def test_build_dataset_end_to_end(tmp_path):
    x, dem, labels = _synthetic_scene()
    _write_geotiffs(tmp_path, x, dem, labels)

    cfg = {
        "data": {
            "root": str(tmp_path),
            "hsi": "hsi.tif",
            "lidar": "dem.tif",
            "lidar_resolution": 2.5,
            "labels": "gt.tif",
        },
        "pipeline": {"patch_size": 32, "overlap": 0.0, "min_labeled": 0.5, "pca_variance": 0.99},
        "split": {"train": 0.7, "val": 0.15, "test": 0.15, "random_state": 42},
        "output": {"root": str(tmp_path / "out"), "npz": "ds.npz", "pca": "pca.pkl"},
    }

    result = build_dataset(cfg)
    n = result["X_hsi"].shape[0]
    k = result["pca"].n_components_
    assert n == 16  # 96/32 x 96/32 quadrants -> 4x4 patches
    assert result["X_hsi"].shape == (16, k, 32, 32)
    assert result["X_dem"].shape == (16, 1, 32, 32)
    assert set(result["y_sev"].tolist()) == {0, 1, 2}

    tr = ~result["val_mask"] & ~result["test_mask"]
    assert int(tr.sum()) + int(result["val_mask"].sum()) + int(result["test_mask"].sum()) == n
    assert not (result["val_mask"] & result["test_mask"]).any()
    assert set(result["y_sev"][tr].tolist()) == {0, 1, 2}  # stratified presence in train

    path = save_dataset(cfg, result)
    assert path.exists()
    with np.load(path) as ds:
        assert ds["X_hsi"].shape == (16, k, 32, 32)
        assert ds["X_dem"].shape == (16, 1, 32, 32)
        assert ds["val_mask"].sum() + ds["test_mask"].sum() == 16 - int(tr.sum())
    assert (tmp_path / "out" / "metadata.json").exists()