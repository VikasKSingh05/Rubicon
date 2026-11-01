# Rubicon — Phase 7 Report

**xBD Satellite Disaster Assessment with an Immutable Proof System**

This report documents what the system does, what is real vs. simulated under which conditions,
how to reproduce every layer, and the known limits ahead of any production claim.
The repository's earlier Houston HSI/LiDAR Mamba-Transformer experiment remains as a separate
legacy path; it is not the primary xBD satellite model.

---

## 1. Architecture recap

```
User ──► Frontend (React/Vite, localhost:3000)
          │ REST (JWT)                        │ Chat
          ▼                                   ▼
        Backend (Express, :4000) ─────► Agent service (:8001)
          │
          ├─ POST /upload (pre + post) ─► xBD Satellite Engine (:8002)
          │                               Siamese U-Net
          ├─ MongoDB (assessment records)
          └─ Optional IPFS + Polygon Amoy proof

Legacy: POST /upload (hsi + lidar) ─► HSI/LiDAR engine (:8000)
```

## 2. Real vs simulated matrix

| Layer                       | Real                                                                                          | Falls back to (clearly marked)                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **xBD upload validation**   | Pre/post PNG/JPEG/TIFF extension and magic-byte validation                                   | —                                                                                    |
| **xBD AI inference**        | Backend forwards pre/post imagery to `/infer/satellite`; Siamese U-Net if `siamese_unet.pt` is available | `stub-satellite-v0` when the checkpoint is absent or the service is unreachable |
| **Legacy HSI/LiDAR**        | Separate `/predict` and Houston 2013 model path                                               | Legacy `stub-v0`; does not represent xBD results                                     |
| **IPFS addressing**         | CIDv0 computed locally from the actual file bytes (deterministic sha2-256 + base58btc)         | Pinata pin hashing: skipped when `PINATA_JWT` unset (`pinned:false`, CID still real) |
| **Pinata pin**              | Pinned upload via `pinFileToIPFS`                                                              | Offline mode — content hash only                                                      |
| **On-chain log**            | Real `logAssessment` tx on Amoy when RPC+key+contract configured                              | Deterministic fake txHash (`chainVerified:false`, state stays `analyzed`)             |
| **On-chain verify**         | `AssessmentLogged` event lookup on `/chain/verify/:cid`                                       | DB-backed status                                                                    |
| **Chat agent**              | Anthropic / OpenAI / OpenRouter tool-calling loop                                             | Deterministic tool-grounded fallback (answers strictly from backend data)            |
| **Auth**                    | bcrypt(10); short-lived access JWT (default 15m, browser memory only) + rotating httpOnly refresh cookie (7d) | — |

Every degradation is intentional and surfaced in the UI via `model_version`, `chainVerified`,
and state.

The model version in each response identifies whether the request used the trained model or a
stub. A successful upload alone does not demonstrate trained-model inference.

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
```
The satellite service health endpoint is `http://localhost:8002/health`. It reports
`model_loaded: true` only when the xBD Siamese U-Net serving checkpoint has been trained and
placed in the mounted xBD model directory. Without it, the upload flow uses a stub.

## 4. Training reproducibility

- Dataset: xBD labeled pre/post optical image pairs under `train/{images,labels}`.
  The trainer makes a disaster-event-disjoint validation split from `train/`;
  the optional `hold/` split is evaluated separately because it shares event names.
- Train and evaluate: follow [`training.md`](training.md) from PowerShell. Outputs
  include `training_metrics.json`, `checkpoint_best.pt`, and independent validation/hold
  reports and overlays.
- Keep xBD artifacts under `ai-engine/data/models/xbd/`. The `model.pt` and `metrics.json`
  directly under `ai-engine/data/models/` belong to a separate HSI/LiDAR experiment and
  must not be reported as xBD metrics.

## 5. Manual deployment steps (not runnable in CI)

1. **IPFS pinning real:** set `PINATA_JWT` (free at app.pinata.cloud). CIDs are real without it.
2. **Contract deploy (one-time):** `cd contracts && npm install && npm run deploy`
   (env: `AMOY_RPC_URL`, `DEPLOYER_PRIVATE_KEY`, `CHAIN_ID=80002`). Fund the testnet-only
   wallet with faucet POL first. Prints `CONTRACT_ADDRESS`; set it in `.env`.
3. Real chain logging activates automatically once RPC + key + address are present.

## 6. Known limits & security posture

**Limits**
- The small accuracy/macro-F1 metrics artifact under `ai-engine/data/models/` identifies
  `mamba_transformer-v1`; it is not an xBD Siamese U-Net evaluation. Use only a report
  generated from the xBD `checkpoint_best.pt` to discuss xBD performance, and include the
  evaluated split and tile/event counts.
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
- The legacy HSI/LiDAR `get_runtime` uses `weights_only=True` torch loading plus a `RestrictedUnpickler`
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
- **Model versions are path-specific**: xBD checkpoints use a `siamese_unet_<encoder>_v1`
  identifier; `mamba_transformer-v1` identifies the separate legacy HSI/LiDAR path.
- **`/assessments` uses limit/offset pagination** (fine at ≤200 records; cursor is overkill here).
- **Errors** never leak internals: 500s return `internal server error` and log server-side.