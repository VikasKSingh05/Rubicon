# Architecture & Delivery Decisions

Decisions are recorded here so future work does not re-litigate them. Each
entry states the decision, the reason, and the consequence.

## 1. Additive migration, not a rewrite

- **Decision:** Extend the existing monorepo in place. Keep the legacy
  `POST /predict` and `POST /upload` routes working while adding
  `/infer/satellite`, `/infer/drone`, and `/assessments`.
- **Reason:** Existing auth, rate limiting, IPFS/Pinata, and Polygon Amoy proof
  logic already work and are covered by tests.
- **Consequence:** Some legacy HSI/LiDAR code remains (`ai-engine/src/`); it is
  not on the xBD path and must not be deleted without a migration plan.

## 2. MongoDB + `2dsphere`, not PostGIS

- **Decision:** Store assessment geometry in MongoDB with `2dsphere` indexes.
- **Reason:** MongoDB is already the backend datastore; PostGIS would add a new
  service and break the existing test setup.
- **Consequence:** Spatial queries are limited to GeoJSON point/polygon
  containment; H3 cell filtering is done in application code.

## 3. Leaflet stays

- **Decision:** Keep Leaflet for map rendering.
- **Reason:** The existing `MapView.jsx` already renders scans and H3 cells.
- **Consequence:** No MapLibre migration; vector tiles are not used.

## 4. In-process async queue

- **Decision:** Run assessment jobs in-process with MongoDB status documents.
- **Reason:** Avoids Redis/Celery infrastructure for a single-node deployment.
- **Consequence:** Jobs are lost on process restart; `state=error` is written on
  failure and the client can retry.

## 5. Event-disjoint splits, never random tiles

- **Decision:** Split xBD by whole disaster event. Primary source is
  `data/xbd/train/`; validation holds out `mexico-earthquake` and
  `palu-tsunami`.
- **Reason:** Random tile splits leak spatially correlated imagery and inflate
  metrics.
- **Consequence:** `hold/` shares event names with `train/` and is only a
  tile-level qualitative holdout, not an unseen-disaster metric.

## 6. One canonical Python package

- **Decision:** `ai-engine/ai_engine/` is the only implementation. The root
  `ai_engine/__init__.py` extends `__path__` so root scripts import the
  canonical package.
- **Reason:** Two copies had already drifted (parser and split logic differed).
- **Consequence:** Do not add code under a root `ai_engine/` subpackage; edit
  the canonical tree.

## 7. Checkpoint contract v2

- **Decision:** Checkpoints are plain dicts loadable with
  `torch.load(..., weights_only=True)` and contain `state_dict`, `config`,
  `norm_mean`/`norm_std`, `damage_classes`, `model_version`, and
  `state_dict_hash`. Legacy `model_state_dict` checkpoints are upgraded on load.
- **Reason:** Serving previously assumed ImageNet normalisation while training
  could use dataset statistics, silently degrading accuracy; unsafe pickle
  loading was also a risk.
- **Consequence:** Training, evaluation, and serving all go through
  `ai_engine.model.checkpoint`. Optimizer/scheduler/scaler state is only stored
  in `checkpoint_last.pt`, keeping best/serving checkpoints small.

## 8. Synthetic fallback is always labelled

- **Decision:** `--synthetic` data and no-checkpoint inference are clearly
  marked (`data_mode`, `mode`, UI banner) instead of silently pretending to be
  real.
- **Reason:** Prevents fabricated metrics and misleading demos.
- **Consequence:** CI can run without xBD, but results are not reported as
  model quality.

## 9. Honest hardware reporting

- **Decision:** Report the actual detected device and profile. Do not claim GPU
  results when training ran on CPU.
- **Reason:** The local machine initially had a CPU-only PyTorch wheel even
  though an RTX 3050 was present.
- **Consequence:** `docs/audit.md` records the blocker; a CUDA wheel install or
  Kaggle run is required before quoting real accuracy numbers.
