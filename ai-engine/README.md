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

## Inference

`src/inference.py` serves the `/predict` stub. Phase 3 replaces `PREDICT_STUB`
with the real fusion model; the contract shape does not change.