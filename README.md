# Rubicon

**Autonomous Multimodal Disaster Assessment with Immutable Proof System**

Rubicon assesses disaster damage from **hyperspectral** (HSI) and **LiDAR** imagery using a
Mamba–Transformer fusion model, then pins the evidence to IPFS and logs an immutable record
on the **Polygon Amoy testnet** — all explorable through a map-based dashboard and a
tool-grounded AI assistant.

> ⚠️ **Project status: Phase 5** — the stub vertical slice is live (register/login, upload,
> map, detail view, AI assistant), the real Mamba–Transformer fusion model runs behind
> `/predict`, and the agent answers from live backend data via tools. Uploads are
> content-addressed as real IPFS CIDv0 values (Pinata pinning when configured) and logged
> to the PramaanLedger on Polygon Amoy when the chain layer is configured; without
> `PINATA_JWT`/Amoy keys the proof steps run in a deterministic simulated mode.

## Monorepo layout

```
Rubicon/
├── ai-engine/       # FastAPI inference service (stub → Phase 2/3 real model)
├── agent-service/   # tool-grounded chat assistant (Phase 4: LLM + fallback tools)
├── backend/         # Express API: upload, assessments, auth (stub → Phase 1+)
├── frontend/        # map-based dashboard (stub → Phase 1)
├── contracts/       # JSON API-contract artifacts + Solidity (contracts/contracts/)
├── docs/            # api-contracts.md, phase0-setup.md
├── .env.example     # env template (Appendix B + OPENAI)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Quickstart

```bash
# 1. Configure environment
cp .env.example .env          # then fill in your keys (see docs/phase0-setup.md)

# 2. Run the full stack (Mongo + services)
docker compose up --build

# 3. Open the app
# Frontend: http://localhost:3000  — register an account, then upload a scan

# Health checks
curl http://localhost:8000/health   # ai-engine
curl http://localhost:8001/health   # agent-service
curl http://localhost:4000/health   # backend
curl http://localhost:3000/health   # frontend

# Ask the tool-grounded agent (no LLM key needed — falls back to
# a deterministic agent that answers strictly from the backend API)
curl -X POST http://localhost:8001/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"how many assessments do we have?"}'
```

## Contracts

All inter-service shapes are captured in [docs/api-contracts.md](docs/api-contracts.md) and
mirrored as JSON under [`contracts/`](contracts/). Build against these exactly.

## Local dev tests

```bash
# Node services
(cd backend && npm test)          # auth, upload, assessments (uses in-memory Mongo)
(cd agent-service && npm test)
(cd frontend && npm test && npm run build)

# Python service
(cd ai-engine && pip install -r requirements.txt && python -m pytest)
```

## Roadmap

| Phase | Focus |
| ----- | ----- |
| 0 | Environment, repo, contracts-first setup |
| 1 | Stub vertical slice (upload → fake AI → map) + auth |
| 2 | Real data exploration + preprocessing pipeline |
| 3 | Real DL model (Mamba–Transformer fusion) |
| 4 | Agentic AI layer (tool-grounded chat) |
| 5 | IPFS + blockchain layer (Amoy) |
| 6 | Integration, testing, polish, report |
