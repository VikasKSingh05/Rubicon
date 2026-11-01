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

## 3. Verified Local Inventory

The pipeline was validated against the real xBD archives on disk (`XBD_ROOT=./data/xbd`):

| Split | Pre/post pairs | Images | Labels | Targets | Events |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `train/` | 2,799 | 5,598 | 5,598 | 5,598 | 10 |
| `hold/` | 933 | 1,866 | 1,866 | 1,866 | 10 |

Per-event pair counts:

| Event | `train/` | `hold/` |
| :--- | ---: | ---: |
| `guatemala-volcano` | 18 | 5 |
| `hurricane-florence` | 319 | 119 |
| `hurricane-harvey` | 319 | 95 |
| `hurricane-matthew` | 238 | 94 |
| `hurricane-michael` | 343 | 109 |
| `mexico-earthquake` | 121 | 34 |
| `midwest-flooding` | 279 | 86 |
| `palu-tsunami` | 113 | 41 |
| `santa-rosa-wildfire` | 226 | 77 |
| `socal-fire` | 823 | 273 |

`data/` is gitignored — the dataset is never committed.

---

## 4. Disaster Event Splits

To prevent spatial autocorrelation and test real generalization, tiles are **never split randomly**. Splits are partitioned strictly by disaster event (parsed from filename prefixes, e.g., `hurricane-florence_00000006` belongs to event `hurricane-florence`).

- Primary source is `data/xbd/train/` (auto-selected when present; `hold/` is used only if `train/` is missing).
- Default validation holds out two whole events: **`mexico-earthquake`** and **`palu-tsunami`**.
- Manifest is written to `data/splits_train.json` (or `data/splits_<split>.json`), with `metadata.split_mode="event"` and `metadata.event_disjoint=true`.
- `load_or_create_splits()` refuses to reuse a manifest that fails `validate_splits()` (non-event mode, tile overlap, or wrong source split) and regenerates it.

### Actual split used for training/validation

| | Events | Tiles |
| :--- | :--- | ---: |
| **Train** | `guatemala-volcano`, `hurricane-florence`, `hurricane-harvey`, `hurricane-matthew`, `hurricane-michael`, `midwest-flooding`, `santa-rosa-wildfire`, `socal-fire` | 2,565 |
| **Validation** | `mexico-earthquake`, `palu-tsunami` | 234 (8.36%) |

**Event overlap: none.**

> **Honest limitation:** 234 validation tiles across 2 events is on the small side (the checker warns below 100 tiles, so it does not warn here, but 2 events still give a coarse estimate). Treat per-event variance as large and prefer the Kaggle 16 GB profile for the full run. Report metrics per event, not just the aggregate.

### `hold/` is not an event-disjoint test set

`hold/` contains the **same 10 event names** as `train/` (different tiles). It is a tile-level holdout, useful for qualitative spot checks and sanity checks, but it is **not** a generalization metric for unseen disasters. The primary metric is the event-disjoint validation split above.

---

## 5. Mask Parity with Shipped Targets

`targets/` PNGs are compared against masks rasterized from the JSON labels by `python -m ai_engine.data.check`. Coordinates are **rounded** (not truncated) to the nearest pixel, which matches the xBD rasterizer.

| Check | Tiles | Localization agreement | Damage agreement |
| :--- | ---: | ---: | ---: |
| Event-disjoint split (`train/`) | 12 | 99.9994% | 99.9447% |
| Secondary `hold/` | 6 | 99.6569% | 99.6439% |

Per-class IoU on the 12-tile check: `background=0.9994`, `no-damage=0.9795`, `minor-damage=1.0000`, `major-damage=1.0000`, `destroyed=1.0000`.

Class frequencies (200 tiles): background 77.51% of all pixels; among building pixels: `no-damage` 95.52%, `minor-damage` 0.37%, `major-damage` 0.94%, `destroyed` 3.18%. This is the imbalance the focal/Lovász loss and the severe-damage weighted sampler are designed for.

Regenerate all of the above plus qualitative overlays:

```bash
$env:XBD_ROOT="./data/xbd"          # PowerShell
python -m ai_engine.data.check --xbd-root ./data/xbd --crosscheck 12 --samples 5
```

Overlays are written to `docs/samples/` (gitignored; regenerate on demand rather than committing ~10 MB of PNGs).

---

## 6. Synthetic Fallback Mode (`--synthetic`)

For continuous integration (CI) and local environments without the multi-gigabyte xBD dataset:
- Run with `--synthetic` flag:
  ```bash
  python -m ai_engine.data.check --synthetic
  python train.py --synthetic --epochs 1
  ```
- Generates procedural building geometries, synthetic pre/post textures, and simulated damage polygons.
- Outputs are visibly labeled with `"mode": "synthetic"` in all API responses and UI headers.
- Synthetic data has no `targets/` PNGs, so the target cross-check is skipped and the split carries a "validation is noisy" warning (18 tiles).

---

## 7. Mask Caching & Preprocessing

- Localization masks come from the **pre-disaster** label; damage classes come from the **post-disaster** label. Damage pixels are force-localized so the two masks stay geometrically consistent.
- Raw WKT polygons are rasterized once into uint8 PNG masks stored in `data/xbd_cache/` (override with `XBD_CACHE_DIR`):
  - `<tile_id>_loc.png`: Binary building localization mask (`{0, 1}`)
  - `<tile_id>_dmg.png`: Multi-class damage mask (`{0, 1, 2, 3, 4}`)
- Cache entries are invalidated automatically when either source JSON is newer (`MASK_CACHE_VERSION` is bumped when rasterization changes).
- Normalization statistics are fitted on a deterministic sample of **training** tiles only (`compute_norm_stats`), never on validation.

---

## 8. Windows Geospatial Library Installation Notes


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
