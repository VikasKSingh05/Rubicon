# Rubicon System Audit & Architecture Baseline

**Date:** 2026-10-27  
**Author:** Senior ML + Full-Stack Engineer  
**Status:** Completed (Phase 1 Acceptance)

---

## 1. Executive Summary

Rubicon is an end-to-end disaster damage assessment platform designed to provide rapid situational awareness to emergency responders. This audit reviews the pre-existing monorepo, evaluates its current services and test suites, verifies hardware and environment constraints, and sets the roadmap for replacing the legacy HSI/LiDAR Mamba-Transformer model with a modern, high-accuracy **Siamese U-Net** on optical satellite imagery (**xBD / xView2**) and drone inspection imagery, while preserving existing authentication, rate limiting, IPFS (Pinata CIDv0), and Polygon Amoy on-chain proof logic (**PramaanLedger**).

---

## 2. Monorepo Structure & Entrypoints

| Service | Directory | Tech Stack | Entrypoint | Ports | Test Framework |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI Engine** | `ai-engine/` | Python 3.13, PyTorch, FastAPI, SMP, Rasterio | `src/inference.py` / `uvicorn` | 8000 | `pytest` (31 tests passing) |
| **Backend** | `backend/` | Node.js (ESM), Express, MongoDB (Mongoose), Ethers.js | `src/index.js` (app: `src/app.js`) | 4000 | Node test runner (`node --test`, 51 tests passing) |
| **Agent Service** | `agent-service/` | Node.js (ESM), Express, LLM Providers (OpenRouter/Anthropic/OpenAI) | `src/index.js` | 8001 | Node test runner (`node --test`, 21 tests passing) |
| **Frontend** | `frontend/` | React 18, Vite, TailwindCSS, Leaflet | `src/main.jsx` | 3000 | `vitest` (30 tests passing) + Vite build |
| **Contracts** | `contracts/` | JSON Schemas & Solidity (`PramaanLedger.sol`) | `contracts/PramaanLedger.sol` | — | Hardhat / contract mirrors |
| **Orchestration**| Root | Docker Compose | `docker-compose.yml` | — | `docker compose config` verified |

---

## 3. Hardware & Environment Profile

Environment auto-detection executed on the local deployment machine:
- **Operating System:** Windows 11 (AMD64)
- **Node Runtime:** Node.js `v22.23.2`, npm `12.0.2`
- **Python Runtime:** Python `3.13.15`
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU
  - **VRAM:** 6,144 MiB (6 GB)
  - **NVIDIA Driver:** 581.86 (CUDA 13.0 driver level)
- **Docker & Compose:** Docker Engine 29.7.2, Docker Compose v5.4.0 (validated with `docker compose config`)

### Hardware Profile Configuration Strategy
1. **Local Profile (6 GB RTX 3050):**
   - ResNet34 encoder (fallback to ResNet18 upon memory pressure)
   - $512 \times 512$ tile crops, batch size 2, gradient accumulation 4 (effective batch size 8)
   - Automatic Mixed Precision (`torch.cuda.amp.autocast(fp16)`) and `channels_last` memory layout
   - Automatic OOM recovery: catches `torch.cuda.OutOfMemoryError`, clears CUDA cache, dynamically reduces crop size to $384 \times 384$ then $256 \times 256$, logs warnings, and resumes training.
   - Safe DataLoader defaults: `num_workers=2` on Windows to avoid spawn/fork bottlenecks.
2. **Kaggle Profile ($\ge 15$ GB VRAM, P100/T4):**
   - ResNet34 encoder, $512 \times 512$ crops, batch size 8, gradient accumulation 1.
3. **CPU Fallback Profile (No GPU):**
   - ResNet18 encoder, $256 \times 256$ crops, batch size 2, subset mode with clear console warnings.

---

## 4. API Contracts & Verification Status

Inter-service contracts documented in `docs/api-contracts.md` and mirrored in `contracts/`:
1. `contracts/predict.response.json`:
   - Returns `{ prediction, confidence, class_probs, geojson_polygon, model_version }`.
   - Backward-compatible endpoint maintained at `POST /predict`.
2. `contracts/upload.response.json`:
   - Returns `{ assessmentId, state, prediction, confidence, class_probs, geojson_polygon, model_version, createdAt, links }`.
3. `contracts/assessment.schema.json`:
   - State machine: `uploaded -> analyzing -> analyzed -> chain_pending -> chain_logged` (or `error`).
   - Fields: `_id`, `user`, `state`, `filename`, `prediction`, `confidence`, `class_probs`, `geojson_polygon`, `model_version`, `hsiCid`, `lidarCid`, `txHash`, `chainVerified`, `timestamps`.
4. `contracts/agent-query.response.json`:
   - Returns `{ answer, tool_calls, createdAt }`, grounded strictly in backend tool executions.

### New API Enhancements for Satellite & Drone Triage (Phase 5 & 6)
To support the two-stage satellite + drone workflow without breaking existing clients:
- **`POST /infer/satellite`**: Accepts `pre` and `post` optical images (GeoTIFF/PNG/JPG with optional BBox/CRS). Outputs GeoJSON FeatureCollections for building polygons and H3 hexagonal cells (resolutions 9–10) with severity classifications (`RED`, `ORANGE`, `YELLOW`, `GREEN`), SHA-256 hashes, and execution metrics.
- **`POST /infer/drone`**: Accepts high-resolution drone imagery and optional `parent_cell_id`. Outputs detailed damage segmentation masks, damage area percentages, and cell linkage.
- **`POST /assessments`**: Accepts satellite pairs, dispatches async jobs, computes input/output SHA-256 hashes, pins evidence JSON to IPFS (Pinata CIDv0), logs on Polygon Amoy (`PramaanLedger`), and stores records in MongoDB/PostGIS.
- **`GET /assessments/:id/cells`**: Returns H3 cells with GeoJSON geometry and optional BBox filtering.
- **`GET /assessments/:id/verify`**: Cryptographically re-checks input/output hashes against IPFS content and on-chain event logs.

---

## 5. Dependency & Windows Environment Management

In `ai-engine/requirements.txt`, all core dependencies have been pinned:
- `torch>=2.6.0`, `torchvision>=0.21.0`
- `segmentation-models-pytorch==0.6.0`, `albumentations==2.0.8`, `opencv-python-headless==4.13.0.92`
- `rasterio==1.5.1`, `shapely==2.2.0`, `geopandas==1.2.0`, `h3==4.5.0`
- `fastapi==0.115.0`, `uvicorn[standard]==0.30.6`, `pydantic==2.12.5`, `tensorboard==2.21.0`, `pytest==8.3.2`

### Windows Installation Notes
- `rasterio` and `geopandas` now provide binary wheels via PyPI for Python 3.11–3.13 on Windows x86_64.
- If native compilation issues occur on a fresh Windows system, install using Conda:
  ```bash
  conda create -n rubicon python=3.11
  conda activate rubicon
  conda install -c conda-forge rasterio geopandas gdal shapely
  pip install -r ai-engine/requirements.txt
  ```

---

## 6. Automation Scripts Created

Under `scripts/`, unified cross-platform scripts were added:
- `scripts/setup.ps1` & `scripts/setup.sh`: Installs dependencies across all four workspaces.
- `scripts/train.ps1` & `scripts/train.sh`: Dispatches model training jobs.
- `scripts/eval.ps1` & `scripts/eval.sh`: Dispatches model evaluation and metric generation.
- `scripts/serve.ps1` & `scripts/serve.sh`: Runs the full stack via Docker Compose or locally (`--local`).
- `scripts/test.ps1` & `scripts/test.sh`: Runs all test suites sequentially with non-zero exit on failure.

---

## 7. Baseline Test Run Results (Phase 1 Baseline)

All 133 initial tests across the monorepo pass:
- **`ai-engine`**: 31 passed (`pytest`)
- **`backend`**: 51 passed (`node --test`)
- **`agent-service`**: 21 passed (`node --test`)
- **`frontend`**: 30 passed (`vitest`), production bundle builds cleanly (`vite build` in 5.4s)

**Total:** 133/133 tests green. Phase 1 acceptance criteria met.

---

## 8. Phases 2–7 Implementation & Verification Summary

### Phase 2: HSI & LiDAR Preprocessing Pipeline
- Implemented robust multi-modal reading and alignment for HSI (GeoTIFF) and LiDAR (`.las`/`.laz` via `laspy` with fallback parsing).
- Houston 2013 dataset pipeline with spatial cropping, elevation normalization, and synthetic fallback generators.

### Phase 3: Mamba-Transformer Fusion Model
- Dual-encoder fusion architecture combining selective state-space layers (Mamba) for spectral feature modeling and spatial attention transformers for LiDAR point-cloud heights.
- Hardware-adaptive training loop with fp16 AMP, gradient accumulation, and OOM recovery.

### Phase 4: Siamese U-Net Architecture & xBD Pipeline
- Siamese dual-branch ResNet34/ResNet18 U-Net for optical satellite disaster triage (`xBD` / `xView2`).
- Joint loss function: Binary Cross-Entropy + Dice loss for building localization, Focal + Lovász softmax for 4-class damage classification (none, minor, major, destroyed).

### Phase 5: Satellite Inference & H3 Hexagonal Grid Aggregation
- Implemented `ai_engine/training/train.py` and `ai_engine/serve.py` exposing `POST /infer/satellite`.
- `ai_engine/model/aggregator.py` converts per-pixel building segmentations into H3 hexagonal cells (resolutions 9 & 10) with severity ratings (`RED`, `ORANGE`, `YELLOW`, `GREEN`).
- Dockerized satellite inference service (`Dockerfile.satellite`) added to `docker-compose.yml` on port 8002.
- Backend proxy route `POST /satellite` wired into `backend/src/app.js`.

### Phase 6: Stage 2 Drone Triage & Cell Linkage
- High-resolution drone inspection module (`ai_engine/model/drone_serving.py`) detecting specific structural failure modes:
  - `roof_collapse`: Dark voids and structural puncture analysis.
  - `facade_failure`: Vertical edge energy and shear analysis.
  - `debris_field`: High-frequency rubble and Laplacian scatter analysis.
- Exposes `POST /infer/drone` on AI engine and `POST /drone` on backend with parent H3 cell linkage and assessment attachment.
- Added `GET /assessments/:id/cells` and `GET /assessments/:id/verify`.

### Phase 7: Interactive Frontend Visualization & Drone Inspector
- Mapbox/Leaflet-based `MapView.jsx` enhanced to render H3 hexagonal cells with severity-based coloring, building metrics, and damage ratios.
- Created `DroneInspectorModal.jsx` allowing responders to upload drone imagery, inspect detected failure modes with confidence bars, and visualize damage zones.
- Production bundle verified with `npm run build` (built cleanly in 5.2s).

---

## 9. Final Test Suite Results (All Phases Complete)

All **168** tests across the monorepo pass cleanly:
- **`ai-engine`**: 55 passed (`pytest` in 64.9s)
- **`backend`**: 59 passed (`node --test` in 7.7s)
- **`agent-service`**: 21 passed (`node --test` in 0.9s)
- **`frontend`**: 33 passed (`vitest` in 9.3s), production build verified (`vite build` in 5.2s)

**Grand Total:** 168/168 tests green (100% pass rate). All phases verified and operational.

---

## 10. Addendum: Real xBD Data Verification (2026-11-01)

The xBD pipeline was re-validated against the real archives in `data/xbd` (not synthetic):

- `train/`: 2,799 pre/post pairs, 10 events; `hold/`: 933 pairs, 10 events. `data/` is gitignored and nothing under it is tracked.
- Event-disjoint split from `train/`: 8 train events / 2,565 tiles, 2 validation events (`mexico-earthquake`, `palu-tsunami`) / 234 tiles (8.36%). Zero event overlap.
- Mask parity vs shipped `targets/` on 12 tiles: localization 99.9994%, damage 99.9447%. Hold split: 99.6569% / 99.6439% on 6 tiles.
- `hold/` shares event names with `train/`; it is a tile-level holdout only, not an unseen-disaster metric.
- Parser fix: WKT coordinates are rounded rather than truncated (damage agreement rose from ~99.58% to ~99.94–99.98%); localization is taken from the pre-disaster label, damage from the post-disaster label.
- `ai-engine` test suite after the fixes: **67 passed** (55 previous + 7 targeted split/parser/cache tests + 5 checkpoint-contract tests).

### Checkpoint and training plumbing (Phase 3)

- Training, evaluation, and serving now share `ai_engine.model.checkpoint`:
  `state_dict` + `config` + `norm_mean`/`norm_std` + `damage_classes` +
  `state_dict_hash`, loaded with `torch.load(..., weights_only=True)`.
  Legacy `model_state_dict` checkpoints are upgraded transparently.
- Serving now normalises inputs with the statistics stored in the checkpoint
  instead of hard-coded ImageNet values.
- OOM recovery retries the current batch after reducing the crop size instead
  of restarting the epoch.
- Verified real-data CPU smoke run (5 train tiles, 1 epoch): train loss 1.9209,
  val loss 1.6607, localization F1 0.5574, xView2 0.1672. Evaluation on 2
  validation tiles: xView2 0.1719, localization F1 0.5731, damage F1 0.0000.
  **These numbers only validate the pipeline; they are not model quality.**
- Resume from `checkpoint_last.pt` was exercised successfully (epoch 2 ran from
  the saved optimizer/scheduler state).

### Known environment blocker (honest status)

The local Python environment currently has `torch 2.10.0+cpu` and `torch.cuda.is_available() == False` despite the RTX 3050 being present. Therefore **no GPU training metrics exist yet**. A CUDA-enabled wheel must be installed (or training run on Kaggle) before any accuracy number can be claimed. CPU-only smoke runs are used to validate the training/eval plumbing, and are labelled as such.

