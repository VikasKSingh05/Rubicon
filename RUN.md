# Run Rubicon

Rubicon uses one xBD Siamese U-Net AI engine for paired satellite-image
assessment and the Stage 2 drone-triage endpoint. There is no separate legacy
model service.

## Docker Compose

1. Copy `.env.example` to `.env` and set `JWT_SECRET`. Optional integrations
   (Pinata, Polygon Amoy, and an LLM provider) can be configured later.
2. From the repository root:

   ```powershell
   docker compose up --build
   ```

3. Open `http://localhost:3000`, register, and upload a pre-disaster and
   post-disaster PNG, JPEG, or TIFF image.
4. Confirm the AI service at `http://localhost:8002/health`. For trained
   inference, it must report `"model_loaded": true`; the checkpoint directory
   is `ai-engine/data/models/xbd`.

| Service | URL |
| --- | --- |
| Dashboard | `http://localhost:3000` |
| Backend | `http://localhost:4000` |
| xBD + drone AI engine | `http://localhost:8002` |
| Agent service | `http://localhost:8001` |

If the xBD checkpoint is absent or the inference service is unavailable, the
prototype identifies the result as `stub-satellite-v0`. Do not present that
fallback as trained-model output.

## Native Windows services

Install the dependencies listed in each service's package manifest and
`ai-engine/requirements.txt`, then start MongoDB. From the repository root:

```powershell
.\scripts\serve.ps1 -Local
```

This opens the backend, frontend, agent service, and one AI-engine process.
Set `SAT_ENGINE_URL=http://localhost:8002` in `.env` if the backend is started
separately.

## Training and evaluation

See [docs/training.md](docs/training.md) for the xBD directory layout, local
training/evaluation commands, validation split policy, and model artifact
locations. The measured real-data run is documented in
[docs/xbd-v3-training-report.md](docs/xbd-v3-training-report.md).

## Tests

```powershell
Set-Location backend; npm test
Set-Location ..\frontend; npm test; npm run build
Set-Location ..\agent-service; npm test
Set-Location ..\ai-engine; python -m pytest
Set-Location ..\contracts; npm run compile
```
