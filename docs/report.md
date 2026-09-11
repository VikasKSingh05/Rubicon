# Rubicon — Phase 7 Report

**Autonomous Multimodal Disaster Assessment with Immutable Proof System**

This report documents what the system does, what is real vs. simulated under which conditions,
how to reproduce every layer, and the known limits ahead of any production claim.

---

## 1. Architecture recap

```
User ──► Frontend (React/Vite, localhost:3000)
          │  REST (JWT)                    │  chat (token relay)
          ▼                                ▼
        Backend (Express, :4000) ──► Mongo Atlas (assessments)
          │  POST /predict (multipart)
          ▼
      ai-engine (FastAPI, :8000) — Mamba–Transformer fusion + PCA
          │
          ▼
   Backend proof pipeline
     ├─ ipfs.js       content CIDv0 (local sha2-256) + Pinata pin (optional)
     └─ chainService  PramaanLedger log on Polygon Amoy (optional)

Agent-service (:8001) — tool-grounded chat (LLM or deterministic fallback)
```

## 2. Real vs simulated matrix

| Layer                       | Real                                                                                          | Falls back to (clearly marked)                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Upload validation**       | multer allow-list (`.tif/.tiff/.las/.laz`), 100 MB cap, extension check                        | —                                                                                    |
| **AI inference**            | Backend forwards the uploaded buffers to `/predict`; per-patch fusion model inference          | `stub-v0` randomized body (backend verbose-warns) when engine unreachable/non-2xx     |
| **IPFS addressing**         | CIDv0 computed locally from the actual file bytes (deterministic sha2-256 + base58btc)         | Pinata pin hashing: skipped when `PINATA_JWT` unset (`pinned:false`, CID still real) |
| **Pinata pin**              | Pinned upload via `pinFileToIPFS`                                                              | Offline mode — content hash only                                                      |
| **On-chain log**            | Real `logAssessment` tx on Amoy when RPC+key+contract configured                              | Deterministic fake txHash (`chainVerified:false`, state stays `analyzed`)             |
| **On-chain verify**         | `AssessmentLogged` event lookup on `/chain/verify/:cid`                                       | DB-backed status                                                                    |
| **Chat agent**              | Anthropic / OpenAI / OpenRouter tool-calling loop                                             | Deterministic tool-grounded fallback (answers strictly from backend data)            |
| **Auth**                    | bcrypt(10); short-lived access JWT (default 15m, browser memory only) + rotating httpOnly refresh cookie (7d) | — |

Every degradation is intentional and surfaced in the UI via `model_version`, `chainVerified`,
and state.

## 3. Reproduce each suite

```bash
# Backend (Node 22; in-memory Mongo)
(cd backend && npm install && npm test)

# Agent-service
(cd agent-service && npm install && npm test)

# Frontend
(cd frontend && npm install && npm test && npm run build)

# ai-engine (Python 3.13 + torch CPU)
(cd ai-engine && pip install -r requirements.txt && python -m pytest)

# Contracts (compile check; no chain needed)
(cd contracts && npm install && npm run compile)
```

CI runs all of the above plus a `contracts` compile job (`.github/workflows/ci.yml`).

### End-to-end local smoke
```bash
docker compose up --build
curl -X POST http://localhost:8000/predict -F hsi=@ai-engine/data/houston2013/Houston_2013_HSI.tif \
     -F lidar=@ai-engine/data/houston2013/Houston_2013_DSM.tif
# → model_version: "mamba_transformer-v1" (requires trained artifact + data dir mounted)
```
The artifact (`ai-engine/data/models/model.pt`) and the demo GeoTIFFs are gitignored; see the
training path below.

## 4. Training reproducibility

- Pipeline: Houston 2013 HSI + LiDAR preprocessing (`ai-engine/src/preprocessing/`) →
  land-cover → severity proxy labels → patch dataset.
- Train: `cd ai-engine && python -m src.training.train --config configs/phase3.yaml`
  (GPU optional; CPU works on the small synthetic set).
- Artifact: `ai-engine/data/models/model.pt` + `pca.pkl` + `data/models/metrics.json`.

## 5. Manual deployment steps (not runnable in CI)

1. **IPFS pinning real:** set `PINATA_JWT` (free at app.pinata.cloud). CIDs are real without it.
2. **Contract deploy (one-time):** `cd contracts && npm install && npm run deploy`
   (env: `AMOY_RPC_URL`, `DEPLOYER_PRIVATE_KEY`, `CHAIN_ID=80002`). Fund the testnet-only
   wallet with faucet POL first. Prints `CONTRACT_ADDRESS`; set it in `.env`.
3. Real chain logging activates automatically once RPC + key + address are present.

## 6. Known limits & security posture

**Limits**
- Model quality is near-baseline (reported test accuracy ≈ 0.33, macro-F1 ≈ 0.25) on a tiny
  11-train/2-val/3-test patch set. This is a **reference implementation**, not operational.
- `/chain/verify` event scan now begins from a persisted `ChainIndex` cursor instead of
  block 0 (Phase 7). On a very low-traffic testnet the scan reaches depth 0 fast; a
  production ledger should index per season for near-instant lookups.
- Originals are spooled to a per-request temp dir (magic-byte validated TIFF/LASF) and
  persisted under `STORAGE_DIR/<assessmentId>/`; they are downloadable owner-scoped via
  `GET /files/:id/:kind`. Multi-hundred-MB files are still read fully into memory once
  during inference — a streaming proxy to the engine is a later optimization.
- JWT secret is required in production (boot fails on a missing/dev default); access
  tokens are short-lived (default 15m) and held in browser **memory only**. Refresh uses
  opaque 256-bit tokens stored sha-256'd in Mongo (token blinding), delivered as
  httpOnly/SameSite=Strict/Path=/auth cookies, rotating on every use and revocable.
- `get_runtime` now uses only `weights_only=True` torch loading plus a `RestrictedUnpickler`
  allow-list for `pca.pkl` (no unsafe fallback); the mounted model directory is operator
  controlled, and corrupt artifacts return a clean 503 instead of executing pickles.

**Posture**
- `.env` is gitignored; `.env.example` documents every key; `NODE_ENV=production`
  enforces secret checks on boot. Never commit real keys.
- Passwords bcrypt-hashed (cost 10, `select:false`); assessment and file queries are
  user-scoped; `change-password` revokes every other session.
- Graceful-degrade everywhere: engine unreachable → stub; no chain keys → simulated proof;
  no LLM key → deterministic agent. All covered by tests.
- Hardening applied and tested (Phase 7): rate limiting on auth/upload (429), CSP +
  restricted CORS (no wildcard), cookie refresh rotation, short-lived access tokens,
  magic-byte upload validation, path-traversal-safe file downloads, production secret
  gating, Prometheus `/metrics`, and per-request `requestId` logs. Remaining known items
  are self-inflicted reference-implementation limits (small dataset, near-baseline model)
  rather than security gaps.

## 7. Deliberate trade-offs

- **Pinata gateway URL** was removed from docs/env (unused) to stop doc drift.
- **`model_version` canonical string is `mamba_transformer-v1`** (docs/mirror updated accordingly).
- **`/assessments` uses limit/offset pagination** (fine at ≤200 records; cursor is overkill here).
- **Errors** never leak internals: 500s return `internal server error` and log server-side.