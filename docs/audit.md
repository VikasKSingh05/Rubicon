# Rubicon Architecture and Implementation Audit

## Current system

Rubicon's satellite workflow uses paired pre- and post-disaster optical imagery
and the xBD (xView2) dual-task Siamese U-Net. One shared encoder processes both
images; a U-Net decoder predicts building localization and four building-damage
classes. The backend persists the original pair and assessment, then optionally
pins the image CIDs and records them on Polygon Amoy.

The same FastAPI service exposes high-resolution drone triage for Stage 2. This
is a separate workflow from the xBD satellite model and is not an HSI/LiDAR
pipeline.

## Service boundaries

| Service | Entrypoint | Responsibility |
| --- | --- | --- |
| Frontend | `frontend/src/main.jsx` | Authenticated map dashboard, paired-image upload, assessment detail, and drone-triage UI |
| Backend | `backend/src/app.js` | Authentication, image validation/storage, inference proxy, assessment records, proof integration |
| AI engine | `ai-engine/ai_engine/serve.py` | xBD satellite inference and drone triage |
| Agent service | `agent-service/src/index.js` | Tool-grounded assessment and proof queries |
| Contracts | `contracts/contracts/PramaanLedger.sol` | Immutable evidence-CID and prediction records |

The AI model's training, evaluation, and serving implementations live under the
single `ai-engine/ai_engine/` Python package. Training uses the event-disjoint
validation split from `train/`; `hold/` remains a separate supplemental check.
The model, data, and run outputs are local artifacts, not source-controlled
files.

## Upload and proof flow

1. The dashboard sends `pre` and `post` PNG, JPEG, or TIFF images to
   `POST /upload`.
2. The backend checks image extensions and magic bytes, persists the pair, and
   forwards it to `POST /infer/satellite` on the AI engine.
3. The returned model version distinguishes a trained checkpoint from the
   `stub-satellite-v0` no-checkpoint or unavailable-engine fallback.
4. The backend stores the result and content-addresses each original as the
   `preCid` and `postCid`. It uses the existing positional CID interface when
   writing to the deployed ledger.

Historical assessment documents may still contain old storage/CID field names.
The backend and dashboard retain read compatibility for those records; new
uploads only accept and write pre/post satellite imagery. No HSI/LiDAR model or
Mamba-Transformer training path is part of the current application.

## Validation

Run the relevant suites from their service directories:

```powershell
Set-Location backend
npm test

Set-Location ..\frontend
npm test
npm run build

Set-Location ..\ai-engine
python -m pytest
```

The root CI workflow runs backend, agent, frontend, Python-engine, and contract
checks. Docker Compose builds the single AI-engine image from
`ai-engine/Dockerfile`.

For the actual xBD v3 training configuration and measured validation results,
see [the training report](xbd-v3-training-report.md) and
[the training guide](training.md).
