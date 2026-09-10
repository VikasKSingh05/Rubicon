"""Training smoke test: end-to-end CLI run on a tiny synthetic npz."""
import json
import pickle

import numpy as np
import pytest
import yaml
from sklearn.decomposition import PCA

from src.training.train import main


def _cfg(tmp_path, npz_path, pca_path):
    return {
        "model": {
            "kind": "baseline_cnn",  # fastest to train in CI
            "hsi": {
                "stem_channels": 4, "token_grid": 8, "d_model": 8,
                "d_state": 4, "d_conv": 3, "expand": 2, "n_blocks": 1,
                "n_heads": 2, "ff_scale": 2, "dropout": 0.0,
            },
            "dem": {"channels": [4, 8], "dropout": 0.0},
            "head": {"hidden": 8},
        },
        "data": {"npz": str(npz_path), "pca": str(pca_path)},
        "pipeline": {"patch_size": 32},
        "train": {
            "batch_size": 8, "epochs": 2, "lr": 1e-3, "weight_decay": 1e-4,
            "seed": 0, "class_weights": False,
        },
        "output": {"dir": str(tmp_path / "models"), "checkpoint": "model.pt",
                   "metrics": "metrics.json"},
    }


def test_train_cli_end_to_end(tmp_path):
    rng = np.random.default_rng(3)
    n = 24
    k = 4
    y = rng.integers(0, 3, size=n)
    X_hsi = rng.normal(size=(n, k, 32, 32)).astype(np.float32)
    X_dem = rng.normal(size=(n, 1, 32, 32)).astype(np.float32)
    # give classes separable structure so a couple of epochs make progress
    for i in range(n):
        X_hsi[i] = X_hsi[i] + (y[i] - 1) * 0.6

    val = rng.choice(n, size=5, replace=False)
    test = rng.choice(np.setdiff1d(np.arange(n), val), size=5, replace=False)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    val_mask[val] = True
    test_mask[test] = True

    npz_path = tmp_path / "data.npz"
    np.savez_compressed(
        npz_path, X_hsi=X_hsi, X_dem=X_dem, y_sev=y,
        val_mask=val_mask, test_mask=test_mask,
        pca_n_components=k, dem_scale=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    pca = PCA(n_components=k)
    pca.fit(rng.normal(size=(1024, k)))
    pca_path = tmp_path / "pca.pkl"
    with open(pca_path, "wb") as fh:
        pickle.dump(pca, fh)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path, npz_path, pca_path)))

    assert main(["--config", str(cfg_path)]) == 0

    out = tmp_path / "models"
    assert (out / "model.pt").exists()
    assert (out / "metrics.json").exists()
    assert (out / "pca.pkl").exists()  # PCA copied beside the model

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["model_version"] == "baseline_cnn-v1"
    assert 0.0 <= metrics["best_val_accuracy"] <= 1.0
    assert set(metrics["test"].keys()) == {
        "accuracy", "macro_f1", "per_class_f1", "counts",
    }


def test_train_errors_on_missing_npz(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "nope.npz", "")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError):
        main(["--config", str(cfg_path)])