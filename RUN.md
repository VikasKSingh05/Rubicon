# Rubicon — Complete Run & Deployment Guide

This guide explains **how Rubicon works**, how to run it **both via automated scripts and manually in terminal windows**, and **how to answer questions and handle errors smoothly during an interview or live demo**.

---

## 🏗️ Architecture Overview

Rubicon consists of 4 core services communicating via HTTP:

| Service | Technology | Port | Purpose |
|---|---|---|---|
| **Frontend** | React 18 + Vite + Leaflet | `3000` | Map-based assessment dashboard, dual-temporal upload modal, AI chat panel |
| **Backend** | Node.js + Express + Mongoose | `4000` | REST API, JWT auth, assessment storage, IPFS/blockchain pipeline |
| **Agent Service** | Node.js + Express (Tool-Grounded) | `8001` | Conversational assistant querying live backend data |
| **AI Engine** | FastAPI + PyTorch + Siamese U-Net | `8000` | xBD pre/post optical satellite change detection (falls back to deterministic stub if offline) |
| **Database** | MongoDB | `27017` | Persistent data storage |

> **Graceful Degradation Design**: Without GPU/PyTorch or blockchain testnet keys, Rubicon automatically falls back to deterministic stubs. The entire application runs and functions 100% end-to-end without crashing.

---

## ⚡ Method 1: Running with Automated Scripts

### 1. Local Run (Native Processes)
Open PowerShell in the `Rubicon/` directory:
```powershell
.\scripts\serve.ps1 -Local
```
*What this does:*
- Spawns 4 dedicated PowerShell windows (AI Engine, Backend, Agent Service, Frontend).
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

If an interviewer asks you to run the services step-by-step from scratch in separate terminals, follow these exact steps:

### Pre-requisites Check
Make sure MongoDB is running:
```powershell
Get-Service -Name MongoDB
```
*(If stopped, start it with `Start-Service MongoDB` in an admin terminal).*

### Step 1: Environment File Configuration
Ensure `.env` exists in the `Rubicon/` root directory:
```powershell
# In Rubicon/
Copy-Item .env.example .env
```
Key settings in `.env`:
- `MONGO_URI=mongodb://localhost:27017/rubicon`
- `JWT_SECRET=rubicon_secret_key_dev_mode_32chars_long`
- `CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000`
- `AI_ENGINE_URL=http://localhost:8000`
- `BACKEND_URL=http://localhost:4000`

---

### Step 2: Terminal 1 — Start the Backend
Open a terminal in `Rubicon/backend`:
```powershell
cd backend
npm.cmd install    # (First time only)
npm.cmd start
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

### Step 5: (Optional) Terminal 4 — AI Engine
If you want to run the native AI engine locally:
```powershell
cd ai-engine
python -m uvicorn ai_engine.serve:app --host 0.0.0.0 --port 8000
```
> **Interview Talking Point:** *"The backend is designed with a resilient circuit breaker pattern. If the AI Engine is not started or reachable, the backend gracefully falls back to deterministic inference (`stubPredict`), returning valid xBD class distributions and preserving 100% of user workflow: upload, polygon mapping, verification, and chat."*

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
python -c "from ai_engine.data.synthetic import generate_synthetic_dataset; generate_synthetic_dataset(output_dir='data/synthetic_xbd', num_pairs=8)"
```

### 2. Run Exploratory Data Analysis (EDA)
Inspect xBD class distribution, polygon pixel areas, disaster event breakdown, and computed inverse-frequency loss weights:
```powershell
cd ai-engine
# Fast synthetic run
python scripts/eda_xbd.py --synthetic

# Or run against full downloaded xBD directory:
python scripts/eda_xbd.py --data-dir data/xbd --save-summary data/eda_summary.json
```
Interactive Jupyter Notebook:
```powershell
cd ai-engine
jupyter lab notebooks/eda.ipynb
```

### 3. Run Hyperparameter Tuning
Search optimal backbone encoders (`resnet18`, `resnet34`), learning rates, localization loss weights, and focal loss gamma:
```powershell
cd ai-engine
# Fast synthetic test run (5 trials)
python scripts/tune_hyperparams.py --synthetic --trials 5 --epochs 1 --dry-run

# Full GPU search run:
python scripts/tune_hyperparams.py --data-dir data/xbd --trials 20 --epochs 10 --device cuda
```

### 4. Interactive GIS Heatmap on Frontend
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
