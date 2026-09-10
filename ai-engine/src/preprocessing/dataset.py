"""End-to-end Houston 2013 preprocessing: load -> align -> patchify -> split.

Run non-interactively (Phase 3 training and CI call this, not notebook cells):

    python -m src.preprocessing.dataset --config configs/phase2.yaml

The pipeline is leak-free: PCA and DEM normalization statistics are fitted on
the TRAIN split only. Outputs are saved as a single .npz plus a pickled PCA for
reuse at inference time.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .align import align_rasters, assert_aligned
from .load_hsi import apply_pca_to_patches, fit_pca, load_hsi, load_labels
from .load_lidar import load_lidar_raster, rasterize_las
from .patchify import patch_dataset
from . import CLASS_TO_SEVERITY, SEVERITY_TO_INDEX


def map_severity(y_class: np.ndarray) -> np.ndarray:
    """Map land-cover class ids -> severity index. Drops label 0 (background).

    Returns (y_sev, y_class_filtered) aligned to the kept patches.
    """
    keep = y_class > 0
    yc = y_class[keep]
    mapped = [CLASS_TO_SEVERITY.get(int(c)) for c in yc.tolist()]
    unk = [i for i, s in enumerate(mapped) if s is None]
    if unk:
        raise ValueError(f"unmapped classes in labels: {set(yc[unk].tolist())}")
    y_sev = np.asarray([SEVERITY_TO_INDEX[s] for s in mapped], dtype=np.int64)
    return y_sev, yc


def stratified_split(y_sev: np.ndarray, ratios, random_state: int):
    """Return (train_idx, val_idx, test_idx) stratified on severity.

    ratios = (train, val, test). The train/rest split is always stratified; the
    val/test sub-split falls back to a shuffled split when a stratum has fewer
    than 2 samples in the remainder (only possible for tiny datasets).
    """
    rng = np.random.default_rng(random_state)
    train_r, val_r, test_r = ratios
    idx = np.arange(len(y_sev))
    tr, rest, _, _ = train_test_split(
        idx, y_sev,
        test_size=1.0 - train_r, stratify=y_sev, random_state=random_state,
    )
    rest_y = y_sev[rest]
    _, counts = np.unique(rest_y, return_counts=True)
    if counts.min() >= 2:
        rel_val = val_r / (val_r + test_r)
        vl, te = train_test_split(
            rest, rest_y, test_size=1.0 - rel_val,
            stratify=rest_y, random_state=random_state,
        )
        return tr, vl, te
    rng.shuffle(rest)
    n_val = int(round(val_r / (val_r + test_r) * len(rest)))
    return tr, rest[:n_val], rest[n_val:]


def normalize_dem(dem_patches: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """Clamp + min-max scale DEM patches to [0, 1]."""
    out = np.clip(dem_patches, mn, mx)
    out = (out - mn) / max(mx - mn, 1e-9)
    return out.astype(np.float32)


def assemble_split(
    patches: dict, train_idx, val_idx, test_idx,
    pca, dem_mn, dem_mx,
) -> dict:
    """Reorder patches into train|val|test and build channels-first tensors."""
    order = np.concatenate([train_idx, val_idx, test_idx])
    hsi = apply_pca_to_patches(patches["hsi"][order], pca)          # (N, p, p, k)
    dem_raw = patches["dem"][order][..., None]                       # (N, p, p, 1)
    dem = normalize_dem(dem_raw, dem_mn, dem_mx)
    n = order.size
    p = hsi.shape[1]

    X_hsi = np.transpose(hsi, (0, 3, 1, 2)).astype(np.float32)      # (N, k, p, p)
    X_dem = np.transpose(dem, (0, 3, 1, 2)).astype(np.float32)      # (N, 1, p, p)
    y_sev = patches["sev"][order]
    y_class = patches["y_class"][order]

    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    val_mask[len(train_idx): len(train_idx) + len(val_idx)] = True
    test_mask[len(train_idx) + len(val_idx):] = True

    return {
        "X_hsi": X_hsi,
        "X_dem": X_dem,
        "y_sev": y_sev,
        "y_class": y_class,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "coords": patches["coords"][order],
    }


def build_dataset(cfg: dict) -> dict:
    """Run the whole pipeline for the config; returns assembled arrays + metadata."""
    data_cfg, pl, split = cfg["data"], cfg["pipeline"], cfg["split"]
    root = Path(data_cfg["root"])

    hsi, hsi_t, hsi_crs = load_hsi(root / data_cfg["hsi"])
    labels, _, _ = load_labels(root / data_cfg["labels"])

    lidar_path = root / data_cfg["lidar"]
    if lidar_path.suffix.lower() in (".las", ".laz"):
        from rasterio.transform import array_bounds

        left, bottom, right, top = array_bounds(hsi.shape[0], hsi.shape[1], hsi_t)
        res = (right - left) / hsi.shape[1]
        dem, dem_t, dem_crs = rasterize_las(
            lidar_path, resolution=res, bounds=(left, right, bottom, top)
        )
    else:
        dem, dem_t, dem_crs = load_lidar_raster(lidar_path)

    dem_align, hsi_t, hsi_crs = align_rasters(
        hsi, hsi_t, hsi_crs, dem, dem_t, dem_crs
    )
    assert_aligned(hsi, dem_align)

    patches = patch_dataset(
        hsi,
        dem_align,
        labels,
        patch_size=pl["patch_size"],
        overlap=pl["overlap"],
        min_labeled=pl["min_labeled"],
    )
    patches["sev"], patches["y_class"] = map_severity(patches["y_class"])

    tr, vl, te = stratified_split(
        patches["sev"],
        (split["train"], split["val"], split["test"]),
        split["random_state"],
    )

    # --- leak-free statistics: fit ONLY on the training split ---
    tr_pix = patches["hsi"][tr].reshape(-1, patches["hsi"].shape[-1])
    pca, _ = fit_pca(tr_pix, pl["pca_variance"])

    dem_tr_valid = patches["dem"][tr][np.isfinite(patches["dem"][tr])]
    dem_mn, dem_mx = float(np.nanmin(dem_tr_valid)), float(np.nanmax(dem_tr_valid))

    result = assemble_split(patches, tr, vl, te, pca, dem_mn, dem_mx)
    result["pca"] = pca
    result["dem_scale"] = (dem_mn, dem_mx)
    return result


def save_dataset(cfg: dict, result: dict) -> Path:
    """Persist the assembled splits and fitted PCA; returns the npz path."""
    out = Path(cfg["output"]["root"])
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / cfg["output"]["npz"]
    keys = ["X_hsi", "X_dem", "y_sev", "y_class", "val_mask", "test_mask", "coords"]
    np.savez_compressed(
        npz_path,
        **{k: result[k] for k in keys if k in result},
        pca_n_components=int(result["pca"].n_components_),
        dem_scale=np.asarray(result["dem_scale"], dtype=np.float32),
    )
    with open(out / cfg["output"]["pca"], "wb") as fh:
        pickle.dump(result["pca"], fh)

    def n(mask):
        return int(mask.sum())

    meta = {
        "npz": str(npz_path),
        "pca": cfg["output"]["pca"],
        "pca_variance": cfg["pipeline"]["pca_variance"],
        "pca_n_components": int(result["pca"].n_components_),
        "pca_explained_variance": float(result["pca"].explained_variance_ratio_.sum()),
        "dem_scale": list(result["dem_scale"]),
        "patch_size": cfg["pipeline"]["patch_size"],
        "split": {
            "train": n(~result["val_mask"] & ~result["test_mask"]),
            "val": n(result["val_mask"]),
            "test": n(result["test_mask"]),
        },
        "severity_counts_train": {
            s: int((result["y_sev"][~result["val_mask"] & ~result["test_mask"]] == i).sum())
            for i, s in enumerate(SEVERITY_TO_INDEX)
        },
    }
    with open(out / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return npz_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/phase2.yaml")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    result = build_dataset(cfg)
    path = save_dataset(cfg, result)
    print(f"[preprocessing] wrote {path}")
    print(
        f"[preprocessing] {n} patches | HSI bands after PCA: "
        f"{int(result['pca'].n_components_)} | explained var: "
        f"{result['pca'].explained_variance_ratio_.sum():.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())