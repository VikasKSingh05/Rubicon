# Rubicon — Operations & "What Is Still Dummy?" Checklist

This document is for anyone who receives the repo and wants to run it honestly.
It separates **what already works for real** from **what is still simulated or
placeholder**, and lists the exact steps that remain before the stack is
production-meaningful.

---

## 1. Real vs. dummy matrix

| Layer | Status | Evidence |
| --- | --- | --- |
| Authentication (register / login / refresh rotation / logout / change-password) | **Real** | bcrypt(10), 15m access JWT in browser memory, rotating httpOnly refresh cookie, server-side session hashes in Mongo. `POST /auth/change-password` revokes all other sessions. |
| Upload validation | **Real** | Magic-byte checks (TIFF `II*\0`/`MM\0*`, LAS `LASF`), multipart to temp dir then persisted to `STORAGE_DIR/<assessmentId>/`, owner-scoped downloads `GET /files/:id/:kind`. |
| Rate limiting / CSP / helmet / secret-gating | **Real** | `/auth` 10/15min, `/upload` 20/15min → 429; production boot refuses a missing/default `JWT_SECRET`. |
| MongoDB persistence | **Real** | Atlas (or bundled `mongodb` container via `MONGO_URI`). Assessments, refresh sessions, `ChainIndex` cursor. |
| IPFS content addressing | **Real** (local CIDs always); **pinning live only if `PINATA_JWT` set** | CIDs are computed locally regardless; without the JWT nothing is pinned to a public gateway. |
| AI-model inference | **DUMMY-ish** | The shipped artifact is the **smoke-training** checkpoint: 16 labeled patches, 5 epochs, `mamba_transformer-v1`, test accuracy **0.333** ≈ chance (`ai-engine/data/models/metrics.json`). Works end-to-end but has no predictive value. |
| Backend `stub-v0` fallback | **Dummy by design** | If the engine is unreachable / 503s / returns a malformed body, the backend returns a **randomized** class + a random ~1.5 km² polygon around Houston (`backend/src/services/aiService.js`), tagged `model_version: "stub-v0"`. |
| agent-service | **Real when a key is set** | LLM tool-grounded agent (Anthropic / OpenAI / OpenRouter via `LLM_PROVIDER`); deterministic fallback agent otherwise. Answers strictly from tool results. |
| On-chain proof (Amoy/PramaanLedger) | **DUMMY until you deploy** | `DEPLOYER_PRIVATE_KEY` and `CONTRACT_ADDRESS` are empty → `chainService` writes a **deterministic fake txHash** (`simulated: true`, `chainVerified: false`, state stays `analyzed`). Real `AssessmentLogged` txs only after contract deploy. |
| Frontend URL wiring | **Local-only by default** | `VITE_API_URL`/`VITE_AGENT_URL` are build-time; unset → `http://localhost:4000` / `:8001` baked into `dist`. |
| Observability | **Real** | `/health` per service, `GET /metrics` (Prometheus: `rubicon_uploads_total`, `rubicon_proofs_total`, `rubicon_errors_total`), requestId JSON logs. |

**How to tell you are looking at dummy data:** every assessment's
`model_version` is `"stub-v0"`, and every `txHash` is fake (`chainVerified: false`)
until section 3 below is done.

---

## 2. What you must do (in order)

### 2.1 Secrets
```bash
openssl rand -hex 32        # → JWT_SECRET (the current local .env value is only 6 chars — replace it)
```
- Set `PINATA_JWT` if you want real pinning (optional; CIDs work without it).
- Never commit `.env` (`git log` is your audit trail if you ever doubt this).

### 2.2 Full-dataset model (make `stub-v0` disappear)
See §3 for the exact reproduction. Until this is done, **every prediction in the
system is random**.

### 2.3 Real on-chain proof
1. Create a fresh testnet-only wallet, fund with Amoy POL (`docs/phase0-setup.md` §3).
2. `cd contracts && npm install && npm run deploy` (or set `RUBICON_CONTRACT_OUT`)
   → prints the ledger address.
3. Set `CONTRACT_ADDRESS` and `DEPLOYER_PRIVATE_KEY` in `.env`.
4. Restart the backend. New uploads now write real `AssessmentLogged` txs
   (state `chain_logged`, real `txHash`).

### 2.4 Public URLs
- Set `CORS_ORIGINS` and `VITE_API_URL` / `VITE_AGENT_URL` to the public scheme+host.
- `docker compose build frontend` picks them up via build args
  (`docker-compose.yml` → `frontend` build). Rebuild all:
  ```bash
  docker compose build && docker compose up -d && docker compose ps
  ```

---

## 3. Full-dataset training reproduction (documented — no runner)

Preprocessing + training live in `ai-engine` and are fully local/deterministic.
The shipped `data/models/*` are the **smoke** artifact; regenerate from the real
Houston 2013 (GRSS DFC) scene to get a real model.

```bash
cd ai-engine
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Download the Houston 2013 scene (see scripts/download_houston2013.py for mirrors)
python scripts/download_houston2013.py            # → data/houston2013/{HSI,DSM,GT}.tif

# 2. Preprocess → npz + fitted PCA
python -m src.preprocessing.dataset --config configs/phase2.yaml
#    → data/processed/houston2013_p32.npz, data/processed/pca_p32.pkl

# 3. Train (edit configs/phase3.yaml for epochs/architecture)
python -m src.training.train --config configs/phase3.yaml
#    → data/models/{model.pt, metrics.json, pca.pkl}
```

**Validation gate — do not ship as "real" until this holds:**
`data/models/metrics.json` should show a test accuracy and per-class F1 well above
~0.33 (the current smoke model ≈ chance) evaluated on the held-out split
(`test_mask` from `phase2.yaml`, 15% / stratified, seed 42).

Deploying the trained artifact:
- `ai-engine` serves whatever is in `RUBICON_MODEL_DIR`
  (container default `/app/data/models` ← `./ai-engine/data` volume).
- `docker compose restart ai-engine`; `GET :8000/health` must report `model_loaded: true`.
- Backend drops the `stub-v0` fallback automatically once `/predict` returns a
  valid body with `model_version: "mamba_transformer-v1"`.

> The shipped smoke artifact is **intentionally not overfit**: it uses the tiny
> `data/processed/houston2013_p32.npz` (16 patches) so the pipeline can be tested
> end-to-end on a laptop. It proves plumbing, not prediction.

---

## 4. Runtime operations summary

| Concern | Command / pointer |
| --- | --- |
| Start / stop | `docker compose up -d` / `docker compose down` (built-in healthchecks on every service) |
| Logs | `docker compose logs -f backend` (`requestId` in every line) |
| Backups | `mongodump` archive + `STORAGE_DIR` snapshot (see `docs/deploy.md` §5) |
| Key rotation | `docs/deploy.md` §6 (JWT rotation without logging everyone out; revoke-all via clearing `refreshtokens`) |
| Monitoring | `GET /metrics` on backend (Prometheus), `/health` per service |
| Training re-run | §3 above; rebuild only the `ai-engine` image or restart the container after replacing the mounted artifact |

---

## 5. Known limits (deliberate)

- "Prediction" = single class on the whole scene + envelope polygon — not a
  per-building GIS product. Demanded by the smoke scope.
- `.las` inputs are rasterized against the HSI grid at inference time (no raw
  LAS ML features).
- Refresh sessions live in Mongo (convenient revoke/rotate; not a dedicated
  token store).
- `stub-v0` results are randomness — **never cite them**.

---

_Related:_ `docs/report.md` (what the suites prove), `docs/deploy.md`
(deployment), `docs/api-contracts.md` (wire formats), `docs/phase0-setup.md`
(per-teammate keys).