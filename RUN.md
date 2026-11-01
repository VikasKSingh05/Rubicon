# Rubicon — Complete Run & Deployment Guide

This guide explains **how Rubicon works**, how to run it **both via automated scripts and manually in terminal windows**, and **how to answer questions and handle errors smoothly during an interview or live demo**.

---

## 🏗️ Architecture Overview

Rubicon's application stack has four primary HTTP services, a separate legacy
engine, and the database:

| Service | Technology | Port | Purpose |
|---|---|---|---|
| **Frontend** | React 18 + Vite + Leaflet | `3000` | Map-based assessment dashboard, dual-temporal upload modal, AI chat panel |
| **Backend** | Node.js + Express + Mongoose | `4000` | REST API, JWT auth, assessment storage, IPFS/blockchain pipeline |
| **Agent Service** | Node.js + Express (Tool-Grounded) | `8001` | Conversational assistant querying live backend data |
| **Satellite AI Engine** | FastAPI + PyTorch + Siamese U-Net | `8002` | xBD pre/post optical satellite inference; uses a stub if the checkpoint is absent |
| **Legacy HSI/LiDAR Engine** | FastAPI + PyTorch | `8000` | Separate Houston 2013 HSI/LiDAR experiment; not the xBD satellite model |
| **Database** | MongoDB | `27017` | Persistent data storage |

> **Graceful Degradation Design**: If the trained xBD checkpoint or inference service is unavailable, the upload workflow returns an explicitly versioned stub response. This supports application demos, but is not trained-model inference.

---

## ⚡ Method 1: Running with Automated Scripts

### 1. Local Run (Native Processes)
Open PowerShell in the `Rubicon/` directory:
```powershell
.\scripts\serve.ps1 -Local
```
*What this does:*
- Spawns 5 dedicated PowerShell windows (legacy engine, xBD satellite engine, Backend, Agent Service, Frontend).
- Automatically configures execution policy and starts each service.

### 2. Docker Compose Run (All-in-One Container Stack)
If Docker Desktop is running:
```powershell
.\scripts\serve.ps1
# Or to rebuild:
.\scripts\serve.ps1 -Build
```

---

## 🎤 Method 2: Manual Terminal Run (Interview / Live Demo Step-by-Step)

If an interviewer or teammate clones the repository to run from scratch in separate terminals, follow these exact steps:

### Step 0: Clone the Repository
```powershell
git clone -b Vanshika https://github.com/VikasKSingh05/Rubicon.git
cd Rubicon
```

### Pre-requisites Check
Make sure MongoDB is running:
```powershell
Get-Service -Name MongoDB
```
*(If stopped, start it with `Start-Service MongoDB` in an admin terminal, or `sudo systemctl start mongod` on Linux).*

### Step 1: Environment File Configuration
Ensure `.env` exists in the `Rubicon/` root directory:
```powershell
# In Rubicon/
Copy-Item .env.example .env    # (or cp .env.example .env on Linux/macOS)
```
Key settings in `.env`:
- `MONGO_URI=mongodb://localhost:27017/rubicon`
- `JWT_SECRET=rubicon_secret_key_dev_mode_32chars_long`
- `CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000`
- `AI_ENGINE_URL=http://localhost:8000` (legacy HSI/LiDAR engine)
- `SAT_ENGINE_URL=http://localhost:8002` (xBD Siamese U-Net engine)
- `BACKEND_URL=http://localhost:4000`

---

### Step 2: Terminal 1 — Start the Backend
Open a terminal in `Rubicon/backend`:
```powershell
cd backend
npm.cmd install            # (or npm install on Linux/macOS)
node scripts/seed_xbd.js   # Seeds the 10 global xBD disaster scenes into MongoDB
npm.cmd start              # (or npm start)
```
**Expected Output:**
```json
{"level":"info","service":"backend","msg":"listening","port":4000}
```
*Verify in browser or curl:* `http://localhost:4000/health` → `{"status":"ok","service":"backend","mongo":"connected"}`

---

### Step 3: Terminal 2 — Start the Agent Service
Open a second terminal in `Rubicon/agent-service`:
```powershell
cd agent-service
npm.cmd install    # (First time only)
npm.cmd start
```
**Expected Output:**
```json
{"level":"info","service":"agent-service","msg":"listening","port":8001}
```
*Verify in browser or curl:* `http://localhost:8001/health` → `{"status":"ok","service":"agent-service","llm":"fallback"}`

---

### Step 4: Terminal 3 — Start the Frontend
Open a third terminal in `Rubicon/frontend`:
```powershell
cd frontend
npm.cmd install    # (First time only)
npm.cmd run dev
```
**Expected Output:**
```
  VITE v5.x.x  ready in ... ms
  ➜  Local:   http://localhost:3000/
```

---

### Step 5: Terminal 4 — xBD Satellite AI Engine
To run the xBD Siamese U-Net inference service natively, use a trained checkpoint
in `ai-engine\data\models\xbd\siamese_unet.pt`:
```powershell
Set-Location Rubicon\ai-engine
$env:RUBICON_MODEL_DIR = (Resolve-Path .\data\models\xbd).Path
python -m uvicorn ai_engine.serve:app --host 127.0.0.1 --port 8002
```
Verify `http://localhost:8002/health` reports `"model_loaded": true`. If the
checkpoint is absent, the service returns a stub response; that is useful for
testing the app workflow, not for demonstrating model inference.

The separate HSI/LiDAR experiment can still be started on port 8000 with
`python -m uvicorn src.inference:app --host 127.0.0.1 --port 8000` from
`Rubicon\ai-engine`. It is not required for the xBD satellite workflow.

> **Interview Talking Point:** *"If the satellite model is missing or the
> satellite service is unreachable, the backend uses its documented stub
> response so the surrounding upload workflow can still be demonstrated."*

---

## 🧪 Running the Full Test Suite

To demonstrate unit and integration test coverage:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```
Or run individually:
- **Backend (59 tests):** `cd backend && npm.cmd test`
- **Agent Service (21 tests):** `cd agent-service && npm.cmd test`
- **Frontend (33 tests):** `cd frontend && npm.cmd test`
- **AI Engine (xBD data pipeline):** `cd ai-engine && pytest tests/test_xbd_data.py`

---

## 🔬 xBD Exploratory Data Analysis (EDA) & Hyperparameter Tuning

### 1. Generate Synthetic xBD Dataset (for fast local dev / CI)
```powershell
cd ai-engine
python -c "from ai_engine.data.synthetic import generate_synthetic_dataset; generate_synthetic_dataset(output_dir='data/synthetic_xbd', tiles_per_event=3)"
```

### 2. Train and Evaluate on the Real xBD Split
```powershell
Set-Location Rubicon\ai-engine
$XbdRoot = "D:\datasets\xbd" # parent of train\ and hold\
python -m ai_engine.data.check --xbd-root $XbdRoot --samples 5 --crosscheck 5
python train.py --xbd-root $XbdRoot --epochs 30 --config auto --subset 1.0 `
  --output-dir data/models/xbd --log-dir runs/xbd-mentor-demo
python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split val --output-dir data/models/xbd/eval-val
python eval.py --xbd-root $XbdRoot --checkpoint data/models/xbd/checkpoint_best.pt `
  --split hold --output-dir data/models/xbd/eval-hold
```
Show `data/models/xbd/training_metrics.json`, the separate validation and hold
`metrics.json` files, and prediction overlays. `val` is the primary
event-disjoint score; `hold` shares event names with `train` and is a
supplemental tile-level result. See `docs/training.md` for full interpretation
and service integration instructions.

### 3. Run Exploratory Data Analysis (EDA)
Inspect xBD class distribution, polygon pixel areas, disaster event breakdown, and computed inverse-frequency loss weights:
```powershell
cd ai-engine
# Fast synthetic run
python scripts/eda_xbd.py --synthetic

# Or run against full downloaded xBD directory:
python scripts/eda_xbd.py --xbd-root $XbdRoot --save-json data/eda_summary.json
```
Interactive Jupyter Notebook:
```powershell
cd ai-engine
jupyter lab notebooks/eda.ipynb
```

### 4. Run Hyperparameter Tuning
The script samples xBD model settings and runs only a few batches per trial; it is for candidate screening, not full training or final performance evidence:
```powershell
cd ai-engine
# Fast synthetic pipeline check (5 trials; not quality evidence)
python scripts/tune_hyperparams.py --synthetic --trials 5 --dry-run

# Short real-xBD candidate search (each trial is only a few batches):
python scripts/tune_hyperparams.py --xbd-root $XbdRoot --trials 20 `
  --output runs\xbd-hyperparam-results.json
```
The tuning script has no `--epochs`, `--data-dir`, or `--device` options and
its brief trials are for candidate screening only, not a substitute for full
training and validation.

### 5. Interactive GIS Heatmap on Frontend
The Frontend dashboard (`http://localhost:3000`) includes a Leaflet GIS layer for xBD disaster hotspots:
1. Open **`http://localhost:3000`**.
2. Look at the top-right GIS Layer Control box on the map.
3. Toggle the **"xBD Trained Data"** checkbox to overlay concentric radial damage intensity heatmaps (Santa Rosa Fire, Hurricane Michael, Mexico City Earthquake, Volcán de Fuego, Pinery Bushfire, Midwest Floods).
4. Use the filter chips (**All Layers**, **Damage Heatmap**, **H3 Grid**, **Footprints**) to isolate disaster density gradients.

---

## 🛠️ Common Errors & How to Resolve Them

### 1. "Failed to fetch" on Register / Login
- **Root Cause:** The browser frontend makes `fetch` requests with `credentials: 'include'` (for secure httpOnly refresh cookies). If the backend CORS does not explicitly respond with `Access-Control-Allow-Credentials: true` or the origin does not match, the browser blocks the response.
- **Fix Applied:** In `backend/src/app.js` and `agent-service/src/index.js`, CORS is configured with `credentials: true` and includes both `http://localhost:3000` and `http://127.0.0.1:3000`.

### 2. "File npm.ps1 cannot be loaded because running scripts is disabled"
- **Root Cause:** Windows PowerShell default execution policy blocks unsigned `.ps1` wrapper scripts.
- **Fix:** Use `npm.cmd` instead of `npm`, or prefix with `cmd.exe /c npm ...`, or run PowerShell with `-ExecutionPolicy Bypass`.

### 3. Docker "command not found" on Windows
- **Root Cause:** Docker Desktop binaries are installed in `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin` and might not be active in the current shell session.
- **Fix:** Add the path to `$env:Path` in PowerShell:
  ```powershell
  $env:Path = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:Path"
  ```

### 4. Port already in use (e.g. 4000, 8001, 3000)
- Check what process is using the port:
  ```powershell
  Get-NetTCPConnection -LocalPort 4000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
  ```
- Terminate the process if needed:
  ```powershell
  Stop-Process -Id <PID> -Force
  ```

---

## 🚀 Live Demo Walkthrough for Interviewers

1. Open **`http://localhost:3000`**.
2. Click **Register** → Create an account with any email (`demo@rubicon.dev`) and password (`Password123`).
3. Click **+ Upload** on the top bar to open the **xBD Satellite Assessment** modal.
4. Drag in paired pre-disaster and post-disaster optical satellite scans (PNG, JPG, or TIFF).
5. Watch the dashboard plot the damage polygon on the Leaflet map with computed xBD damage classification (`No Damage`, `Minor Damage`, `Major Damage`, or `Destroyed`) and class confidence distribution.
6. Click the polygon to inspect the cryptographic IPFS CIDs and Polygon Amoy transaction verification.
7. Open the **AI Assistant** chat panel on the right and ask:
   - *"How many assessments do we have?"*
   - *"Show me severe damage zones."*
   The assistant uses tool-grounded function calling against the live backend database to answer accurately.
