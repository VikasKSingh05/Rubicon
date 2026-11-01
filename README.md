# Rubicon

**xBD Satellite Disaster Assessment with an Immutable Proof System**

Rubicon's primary computer-vision prototype assesses building damage from paired
pre- and post-disaster optical satellite images in the **xBD (xView2) dataset**.
Its satellite model is a dual-task Siamese U-Net with a shared ResNet encoder,
building-localization and damage-classification heads. Assessments flow through
the backend to the map dashboard, with optional IPFS pinning and Polygon Amoy
proofs, and a tool-grounded AI assistant.

The repository also retains a separate HSI/LiDAR inference and preprocessing
path for the Houston 2013 experiment. That legacy path is not the xBD model or
the primary satellite training workflow; model artifacts and results must not
be mixed.

> ⚠️ **Project status: Phase 7** — hardened vertical slice is live end-to-end
> (register/login, upload, map, detail view, AI assistant, original-file
> downloads, chain re-verification, password change). Sessions use short-lived
> access tokens (browser memory only) + rotating httpOnly refresh cookies;
> auth/upload are rate-limited and uploads are magic-byte validated before
> inference. Uploads are content-addressed as IPFS CIDv0 values (Pinata pinning
> when configured) and the PramaanLedger is logged on Polygon Amoy when the
> chain layer is configured. Without engine/`PINATA_JWT`/Amoy keys each layer
> degrades to a clearly marked fallback so the app remains demonstrable.
> The agent answers from live backend data via tools (LLM-backed with a key,
> deterministic tool-grounded fallback otherwise). Production deployment
> guidance lives in [docs/deploy.md](docs/deploy.md).

## Monorepo layout

```
Rubicon/
├── ai-engine/       # xBD Siamese U-Net training, evaluation, and satellite inference
│                    # plus separate legacy HSI/LiDAR experiment
├── agent-service/   # tool-grounded chat assistant (LLM + deterministic fallback)
├── backend/         # Express API: upload → real inference → proof pipeline, auth
├── frontend/        # map-based dashboard
├── contracts/       # JSON API-contract artifacts + Solidity (contracts/contracts/)
├── docs/            # api-contracts.md, phase0-setup.md, report.md
├── .env.example     # env template (IPFS, blockchain, agent providers)
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Train the xBD model locally

Follow [docs/training.md](docs/training.md) for the full data-check, training,
validation/hold evaluation, TensorBoard, and service-integration workflow.
Quick outline from PowerShell:

```powershell
Set-Location C:\Users\LENOVO\Desktop\Rubicon\ai-engine
$XbdRoot = "D:\datasets\xbd" # contains train\ and hold\ directories
python -m ai_engine.data.check --xbd-root $XbdRoot --samples 5 --crosscheck 5
python train.py --xbd-root $XbdRoot --config auto --epochs 30 --subset 1.0 `
  --output-dir data/models/xbd --log-dir runs/xbd-mentor-demo
python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split val --output-dir data/models/xbd/eval-val
```

The primary score is event-disjoint validation from `train/`; `hold/` is
evaluated separately as a supplemental tile-level check. These commands do not
use synthetic data. A successful smoke test is not evidence of real-world
accuracy, and no training-quality claims should be made until the full real-data
run and evaluation have completed.

## Quickstart

```bash
# 1. Configure environment
cp .env.example .env          # then fill in your keys (see docs/phase0-setup.md)

# 2. Run the full stack (Mongo + services)
docker compose up --build

# 3. Open the app
# Frontend: http://localhost:3000  — register an account, then upload a scan

# Health checks
curl http://localhost:8002/health   # xBD Siamese U-Net satellite engine
curl http://localhost:8000/health   # separate legacy HSI/LiDAR engine
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
| 3 | xBD optical satellite model (Siamese U-Net); HSI/LiDAR is a separate legacy experiment |
| 4 | Agentic AI layer (tool-grounded chat) |
| 5 | IPFS + blockchain layer (Amoy) |
| 6          | Integration, testing, polish, report |
| 7          | Hardening: session security, rate limiting, artifact persistence, ops docs |
