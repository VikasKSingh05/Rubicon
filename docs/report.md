# Rubicon Prototype Report

Rubicon is a disaster-assessment prototype whose satellite model operates on
paired pre- and post-disaster optical imagery from xBD (xView2). Its AI engine
uses a dual-task Siamese U-Net with a building-localization head and a
four-class damage head. The app also provides a separate Stage 2 drone-triage
workflow.

## Application flow

```text
Dashboard -> POST /upload (pre + post images)
          -> backend validation and evidence storage
          -> AI engine POST /infer/satellite
          -> assessment detail, map, and optional IPFS/Polygon proof
```

The backend and frontend communicate with the same xBD contract. New uploads
are pre/post satellite image pairs; historical MongoDB records and the
deployed ledger's existing CID argument positions remain compatible.

## Honest model status

An upload completing does not by itself prove trained inference occurred.
Check the AI engine's `/health` response for `model_loaded: true`, then confirm
the assessment's `model_version` names the trained checkpoint rather than
`stub-satellite-v0`. The stub is for demonstrating application flow only.

## Training and results

Use [training.md](training.md) for reproducible xBD data checks, training,
evaluation, checkpoint loading, and validation rules. The measured v3 run,
hyperparameters, per-class results, and confusion matrix are in
[xbd-v3-training-report.md](xbd-v3-training-report.md).

## Run and validate

```powershell
docker compose up --build
```

```powershell
Set-Location backend; npm test
Set-Location ..\frontend; npm test; npm run build
Set-Location ..\agent-service; npm test
Set-Location ..\ai-engine; python -m pytest
Set-Location ..\contracts; npm run compile
```

Optional external integrations are IPFS pinning (Pinata), Polygon Amoy, and a
configured LLM provider. The prototype remains usable without them and reports
simulated proof or deterministic agent behavior where applicable.
