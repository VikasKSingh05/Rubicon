# Deploying Rubicon

This document covers production deployment of the Rubicon stack: services,
required secrets, TLS, backups, and key rotation.

## Topology

```
                    ┌─────────────┐
Browser (TLS)  ───► │  nginx/Caddy │──► frontend:3000 (static, served by node)
                    └─────────────┘──► backend:4000   (REST + auth)
                                      ├─ mongodb:27017 (state + refresh sessions + chain cursor)
                                      ├─ ai-engine:8000 (FastAPI inference)
                                      └─ agent-service:8001 (tool-grounded LLM agent)
```

Every service runs in Docker via `docker compose`. Mongo data and original
uploaded artifacts live on named/`STORAGE_DIR` volumes.

## 1. Preflight checks

- `git clone` the repo to the host, `cd` in, and copy `.env.example` to `.env`.
- Generate a strong secret: `openssl rand -hex 32` → `JWT_SECRET`.
- Set `NODE_ENV=production` **after** `JWT_SECRET` is set.
- Set `CORS_ORIGINS` to the exact origin(s) the browser will use (e.g. `https://rubicon.example.com`). There is **no wildcard** — requests from any other origin are rejected by the backend and agent-service.
- Decide upload limits and storage: `MAX_UPLOAD_MB`, `STORAGE_DIR` (backend volume).

## 2. Required secrets (no default in production)

| Variable | Purpose | Who needs it |
| --- | --- | --- |
| `JWT_SECRET` | signs access tokens | backend |
| `PINATA_JWT` | pins CIDs to IPFS (optional — CIDs are computed locally without it) | backend |
| `AMOY_RPC_URL` / `DEPLOYER_PRIVATE_KEY` / `CONTRACT_ADDRESS` | Polygon Amoy on-chain proof | backend |
| `LLM_PROVIDER` + its key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`) | conversational agent | agent-service |

If `AMOY_RPC_URL`/key/contract are all unset the backend runs a deterministic
**simulated** on-chain record (state stays `analyzed`). Set all three to write
real `AssessmentLogged` transactions. `PINATA_JWT` is not required for CIDs.

The backend refuses to boot in `NODE_ENV=production` when `JWT_SECRET` is
missing or set to the development default, so a misconfigured clone fails fast.

## 3. Deploy

```bash
docker compose build
docker compose up -d
docker compose ps          # all healthy
```

- The backend validates magic bytes (TIFF `II*\0`/`MM\0*`, LAS `LASF`) and
  persists originals under `STORAGE_DIR/<assessmentId>/`.
- Auth uses a short-lived access token (default 15m, `ACCESS_TOKEN_TTL`) held in
  browser memory and an `httpOnly; SameSite=Strict; Path=/auth` refresh cookie
  that rotates on every use. Sessions are revocable server-side.
- Rate limits on `/auth` (10/15min) and `/upload` (20/15min) are on by default;
  `RATE_LIMIT_DISABLED=1` is for development only.

## 4. TLS / reverse proxy

Put the public frontend and API behind a TLS-terminating reverse proxy:

```nginx
# nginx
server {
  listen 443 ssl;
  server_name rubicon.example.com;

  location / {
    proxy_pass http://127.0.0.1:3000;
    proxy_set_header Host $host;
  }
  location /api/ {
    proxy_pass http://127.0.0.1:4000;   # strip /api if you prefer
  }
}
# MUST echo Origin (and encode no wildcard), since cookies + JSON are used
```

Cookie `SameSite=Strict` + HTTPS means browsers only send the refresh cookie
back on same-site (top-level) requests over TLS.

## 5. Backups

- **Mongo**: snapshot regularly, e.g. `docker compose exec mongodb mongodump --archive --db rubicon | gzip > dump.gz`; restore with `mongorestore`.
  This also backs up refresh sessions (so logout/rotation survive restore) and
  the `ChainIndex` scan cursor used by `verifyOnChain`.
- **Originals**: `STORAGE_DIR` holds the uploaded HSI/LiDAR evidence; back it up
  with the Mongo dump so a restore is complete.
- **On-chain**: Amoy is a public ledger — CIDs and predictions remain readable
  from `AssessmentLogged` events even if a database is lost.

## 6. Key rotation

1. Generate a new `JWT_SECRET`, update `.env`, restart the backend
   (`docker compose up -d backend`). Existing access tokens die on restart age;
   refresh cookies remain valid because sessions are DB-backed (only the hash is
   stored, token blinding).
2. To force everyone to re-authenticate: clear the `refreshtokens` collection
   (or rely on the 7-day TTL / expireAfterSeconds index).
3. Rotate an exposed LLM/Pinata key the same way: change `.env`, restart the
   affected service. Rotate `DEPLOYER_PRIVATE_KEY` only if it was ever on a
   personal/known wallet — testnet keys are disposable.

## 7. Monitoring

- `GET /health` per service (used by compose healthchecks).
- `GET /metrics` on the backend: Prometheus text format with
  `rubicon_uploads_total`, `rubicon_proofs_total`, and `rubicon_errors_total`
  counters plus `process_uptime_seconds`. This is open on the backend port;
  block it at the proxy in production.
- Structured JSON request logs with `requestId` (backend, agent-service,
  ai-engine) — thread them through `X-Request-Id` for debugging.

## 8. First-run manual steps

These require an external account/wallet and are intentionally not automated:

1. Create a Pinata API key and set `PINATA_JWT` (CID compute/pinning).
2. Deploy the ledger to Amoy: `cd contracts && npm install && npm run deploy`
   with a funded testnet wallet; put the printed address in `CONTRACT_ADDRESS`.
3. Set `LLM_PROVIDER` to your provider and its key.