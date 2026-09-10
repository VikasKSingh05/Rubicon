# Rubicon ai-engine

Staged plan:
- Phase 0: stub returning the exact `/predict` contract shape.
- Phase 2: preprocessing pipeline (HSI PCA, LiDAR DEM, alignment, patchify).
- Phase 3: real Mamba–Transformer fusion model swapped in with zero contract change.

## Phase 2 — Houston 2013 preprocessing

```
pip install -r requirements.txt
python scripts/download_houston2013.py          # fetch HSI + DSM + GT into data/houston2013/
python -m src.preprocessing.dataset --config configs/phase2.yaml
jupyter lab notebooks/eda.ipynb                 # exploratory checks
python -m pytest
```

Pipeline (`src/preprocessing/`):

| module | role |
|---|---|
| `load_hsi.py` | read the 144-band cube, normalize, PCA to ~99% variance |
| `load_lidar.py` | rasterize raw `.las` (laspy) or read a pre-rasterized DSM |
| `align.py` | reproject the DEM onto the HSI grid |
| `patchify.py` | sliding-window tiling into 32x32 patches, majority-label |
| `dataset.py` | orchestrator: leak-free PCA + DEM stats (train only), 70/15/15 stratified split, saves `.npz` + pickled PCA for inference |

Datasets are gitignored (`data/`). Damage classes are a land-cover proxy —
see `src/preprocessing/__init__.py`.

## Phase 3 — Mamba-Transformer fusion model

```
python -m src.training.train --config configs/phase3.yaml   # reads the Phase 2 npz
python -m pytest
```

| module | role |
|---|---|
| `src/model/mamba.py` | pure-PyTorch selective SSM (CPU-portable Mamba layer) |
| `src/model/fusion.py` | HSI branch (conv stem -> Mamba+Transformer tokens) + LiDAR CNN branch, late fusion head |
| `src/model/baseline.py` | fallback CNN baseline (report comparison / constrained environments) |
| `src/model/factory.py` | `kind: mamba_transformer \| baseline_cnn` from config |
| `src/training/train.py` | CLI: train on the npz, write `data/models/{model.pt,metrics.json,pca.pkl}` |
| `src/model/serving.py` | uploaded HSI/LiDAR files -> aligned patches -> contract response |

`data/models/` is gitignored and mounted into the ai-engine container
(`docker-compose.yml`), so a trained artifact is picked up at startup:
`/health` reports `model_loaded`. Without files or an artifact, `/predict`
returns the original stub body so earlier phases keep working.

The Mamba-Transformer selection is a per-scene aggregation: per-patch
severity probabilities are averaged into `class_probs`, `prediction` is the
argmax, `confidence` the mean probability of that class, and `geojson_polygon`
is the envelope of affected (non-"None") patches georeferenced to lon/lat.

## Inference

`src/inference.py` serves `/predict`. With a trained artifact at
`RUBICON_MODEL_DIR` (default `./data/models`), it accepts multipart `hsi`
(.tif/.tiff) and `lidar` (.tif/.las) files and runs the real fusion model;
otherwise it falls back to the Phase 0 stub. The contract shape never changes.