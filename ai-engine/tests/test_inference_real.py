"""Integration: POST multipart HSI + LiDAR to /predict -> real model response.

Builds a full prepared scene, trains a tiny model on it, then serves it through
the FastAPI app (mirroring the live deployment flow).
"""
import tempfile

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

import src.inference as inference_app


def _tiny_npz(tmp_path, k):
    rng = np.random.default_rng(3)
    n = 16
    y = rng.integers(0, 3, size=n)
    X_hsi = (
        rng.normal(size=(n, k, 32, 32))
        + (y - 1)[:, None, None, None] * 0.7
    ).astype(np.float32)
    X_dem = rng.normal(size=(n, 1, 32, 32)).astype(np.float32)
    val = rng.choice(n, size=4, replace=False)
    test = rng.choice(np.setdiff1d(np.arange(n), val), size=4, replace=False)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    val_mask[val] = True
    test_mask[test] = True
    path = tmp_path / "data.npz"
    np.savez_compressed(
        path, X_hsi=X_hsi, X_dem=X_dem, y_sev=y,
        val_mask=val_mask, test_mask=test_mask,
        pca_n_components=k, dem_scale=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    return path


def _write_tiff(path, arr, count, dtype):
    import rasterio as rio
    from rasterio.transform import Affine

    t = Affine(2.5, 0, 0, 0, -2.5, 0)
    with rio.open(
        path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
        count=count, dtype=dtype, crs="EPSG:32615", transform=t,
    ) as dst:
        if count == 1:
            dst.write(arr, 1)
        else:
            dst.write(arr)  # band-first (bands, height, width)


def _scene_files(tmp_path):
    rng = np.random.default_rng(7)
    h = w = 40
    b = 20
    x = rng.normal(size=(b, h, w)).astype("float32")
    x = ((x - x.min()) / (x.max() - x.min()) * 50000 + 10000).astype("uint16")
    dem = np.full((h, w), 10.0, dtype="float32")
    dem[:20, :20] = 13.0
    hsi_path = tmp_path / "scene_hsi.tif"
    dem_path = tmp_path / "scene_dem.tif"
    _write_tiff(hsi_path, x, b, "uint16")
    _write_tiff(dem_path, dem, 1, "float32")
    return hsi_path, dem_path


@pytest.fixture
def trained_artifact(tmp_path):
    from sklearn.decomposition import PCA
    from src.training.train import main

    hsi_path, dem_path = _scene_files(tmp_path)

    pca = PCA(n_components=0.99)  # fit on the 20-band scene space like Phase 2 does
    pca.fit(np.random.default_rng(0).normal(size=(1024, 20)))
    k = int(pca.n_components_)
    npz_path = _tiny_npz(tmp_path, k=k)

    pca_path = tmp_path / "pca.pkl"
    with open(pca_path, "wb") as fh:
        import pickle

        pickle.dump(pca, fh)

    cfg = {
        "model": {
            "kind": "baseline_cnn",
            "hsi": {"stem_channels": 4, "token_grid": 8, "d_model": 8,
                    "d_state": 4, "d_conv": 3, "expand": 2, "n_blocks": 1,
                    "n_heads": 2, "ff_scale": 2, "dropout": 0.0},
            "dem": {"channels": [4, 8], "dropout": 0.0},
            "head": {"hidden": 8},
        },
        "data": {"npz": str(npz_path), "pca": str(pca_path)},
        "pipeline": {"patch_size": 32},
        "train": {"batch_size": 8, "epochs": 2, "lr": 1e-3, "weight_decay": 1e-4,
                  "seed": 0, "class_weights": False},
        "output": {"dir": str(tmp_path / "models"), "checkpoint": "model.pt",
                   "metrics": "metrics.json"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(cfg_path)]) == 0
    return tmp_path / "models", hsi_path, dem_path


def test_predict_with_real_model(trained_artifact, monkeypatch):
    model_dir, hsi_path, dem_path = trained_artifact
    monkeypatch.setenv("RUBICON_MODEL_DIR", str(model_dir))
    inference_app._runtime = None

    client = TestClient(inference_app.app)
    assert client.get("/health").json()["model_loaded"] is True

    files = {
        "hsi": ("scene_hsi.tif", open(hsi_path, "rb"), "image/tiff"),
        "lidar": ("scene_dem.tif", open(dem_path, "rb"), "image/tiff"),
    }
    res = client.post("/predict", files=files)
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {
        "prediction", "confidence", "class_probs", "geojson_polygon", "model_version",
    }
    assert body["prediction"] in ["None", "Moderate", "Severe Collapse"]
    assert set(body["class_probs"].keys()) == {"None", "Moderate", "Severe Collapse"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["geojson_polygon"]["type"] == "Polygon"
    assert body["model_version"] == "baseline_cnn-v1"


def test_predict_stub_when_no_model(monkeypatch):
    monkeypatch.setenv("RUBICON_MODEL_DIR", tempfile.mkdtemp())
    inference_app._runtime = None
    client = TestClient(inference_app.app)
    res = client.post("/predict")
    assert res.status_code == 200
    body = res.json()
    assert body["model_version"] == "stub-v0"
    assert body["prediction"] == "Severe Collapse"