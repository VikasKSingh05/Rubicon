# Rubicon Operations Guide

This guide describes the current xBD satellite workflow and the conditions
under which the prototype uses real inference versus a clearly marked fallback.

## 1. Runtime status checks

```powershell
Invoke-RestMethod http://localhost:8002/health # xBD + drone AI engine
Invoke-RestMethod http://localhost:4000/health # backend and MongoDB
Invoke-RestMethod http://localhost:8001/health # agent service
Invoke-RestMethod http://localhost:3000/health # frontend
```

The AI-engine health response must report `"model_loaded": true` before an
upload can be presented as trained-model inference. Its checkpoint is loaded
from `RUBICON_MODEL_DIR`, defaulting to `ai-engine/data/models/xbd`. If the
checkpoint is absent, the engine returns `stub-satellite-v0`. If the engine is
unreachable, the backend logs the cause and returns the same stub model label.
Neither fallback is a real prediction.

## 2. Dataset and training

The only supported satellite training pipeline uses xBD pre/post images and
building annotations. Keep the event-disjoint validation split from `train/`;
do not use `hold/` for the primary score. The full setup, commands, artifacts,
and measured v3 results are in [training.md](training.md) and
[xbd-v3-training-report.md](xbd-v3-training-report.md).

Model checkpoints, xBD data, evaluation overlays, and TensorBoard runs are
local artifacts. They are intentionally excluded from Git. Back up trained
checkpoints separately if the deployment depends on them.

## 3. Upload and persistence

- `POST /upload` requires one `pre` and one `post` PNG, JPEG, or TIFF image.
- The backend validates file extensions and image magic bytes, stages uploads
  in a temporary directory, then persists originals under
  `STORAGE_DIR/<assessmentId>/`.
- Owner-scoped downloads use `GET /files/:id/pre` and
  `GET /files/:id/post`.
- Old stored assessment records retain read compatibility; new HSI/LiDAR
  multipart uploads are not accepted.

## 4. Proof and integrations

- IPFS CIDs are computed locally from uploaded file bytes. Set `PINATA_JWT` to
  pin them with Pinata.
- Polygon Amoy writes require `AMOY_RPC_URL`, `DEPLOYER_PRIVATE_KEY`, and
  `CONTRACT_ADDRESS`. Without all three, the backend records a simulated
  transaction and reports `chainVerified: false`.
- The existing deployed ledger uses positional CID arguments. The application
  supplies the pre/post image CIDs in those positions without changing the
  deployed contract interface.
- The agent uses configured LLM credentials when available and otherwise
  answers through its deterministic backend-tool fallback.

## 5. Runtime commands

```powershell
# Start the full stack
docker compose up --build

# Follow service logs
docker compose logs -f ai-engine backend

# Run the end-to-end smoke suite against a running stack
npm run e2e
```

See [deploy.md](deploy.md) for production secrets, backups, TLS, and deployment
configuration.
