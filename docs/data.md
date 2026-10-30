# Rubicon Data Documentation

## 1. Primary Dataset: xBD (xView2)

**xBD** is the premier large-scale satellite imagery dataset for building damage assessment. It features high-resolution (0.5m GSD), $1024 \times 1024$ optical satellite image pairs (pre-disaster and post-disaster) across diverse natural disasters worldwide (hurricanes, earthquakes, wildfires, floods, volcanic eruptions, and tsunamis).

### Damage Classification Scale
In accordance with the Joint Damage Scale (xView2):
- **0: Background** (no building footprint)
- **1: No Damage** (undamaged building; also incorporates `"un-classified"` buildings per specification)
- **2: Minor Damage** (partially affected roofs, superficial wall damage, minor debris)
- **3: Major Damage** (significant structural failure, partial roof/wall collapse)
- **4: Destroyed** (total collapse, foundation-only remnants)

> **Note on "un-classified":** Pre-disaster annotations and certain ambiguous post-disaster buildings are marked `"un-classified"` in xBD labels. These are treated as **no-damage (class 1)** during target mask generation to avoid false damage signals.

---

## 2. Dataset Acquisition & Setup

### Manual Download Steps (xView2 / xBD)
1. Navigate to [xView2.org](https://www.xview2.org/) and register for a free account.
2. Download the dataset archives:
   - `train.tar.gz`
   - `hold.tar.gz`
   - `test.tar.gz`
3. Extract the archives into the project data directory:
   ```text
   Rubicon/
   └── data/
       └── xbd/
           ├── train/
           │   ├── images/   # *.png (pre and post)
           │   ├── labels/   # *.json (pre and post)
           │   └── targets/  # optional xView2 pre-rendered masks
           ├── hold/
           │   ├── images/
           │   └── labels/
           └── test/
               ├── images/
               └── labels/
   ```
4. Configure the dataset root directory (if located outside `./data/xbd`):
   ```bash
   export XBD_ROOT="/path/to/xbd"       # Linux / macOS
   $env:XBD_ROOT="D:\data\xbd"         # Windows PowerShell
   ```
   If the directory does not exist, all data pipeline loaders fail immediately with an explicit, actionable error message.

---

## 3. Disaster Event Splits

To prevent spatial autocorrelation and test real generalization, tiles are **never split randomly**. Splits are partitioned strictly by disaster event (parsed from filename prefixes, e.g., `hurricane-florence_00000006` belongs to event `hurricane-florence`).

- At least 2 full disaster events are held out for validation (default: `mexico-earthquake`, `palu-tsunami`).
- Event splits and tile manifests are deterministically generated and saved to `data/splits.json`.

---

## 4. Synthetic Fallback Mode (`--synthetic`)

For continuous integration (CI) and local environments without the multi-gigabyte xBD dataset:
- Run with `--synthetic` flag:
  ```bash
  python -m ai_engine.data.check --synthetic
  python train.py --synthetic --epochs 1
  ```
- Generates procedural building geometries, synthetic pre/post textures, and simulated damage polygons.
- Outputs are visibly labeled with `"mode": "synthetic"` in all API responses and UI headers.

---

## 5. Mask Caching & Preprocessing

- Raw xBD polygon WKTs are parsed and rasterized once into uint8 PNG masks and stored in `data/xbd_cache/`:
  - `<tile_id>_loc.png`: Binary building localization mask (`{0, 1}`)
  - `<tile_id>_dmg.png`: Multi-class damage mask (`{0, 1, 2, 3, 4}`)
- Caching eliminates CPU rasterization bottlenecks during PyTorch DataLoader iterations.

---

## 6. Windows Geospatial Library Installation Notes

`rasterio`, `geopandas`, and `shapely` are pre-compiled for Python 3.11–3.13 on Windows x86_64:
```bash
pip install rasterio geopandas shapely
```
If your system encounters C++ runtime or GDAL compilation errors:
```bash
conda create -n rubicon python=3.11
conda activate rubicon
conda install -c conda-forge rasterio geopandas gdal shapely
pip install -r ai-engine/requirements.txt
```
