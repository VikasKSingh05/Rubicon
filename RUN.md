# Rubicon — Complete Run & Deployment Guide

This guide explains **how Rubicon works**, how to run it **both via automated scripts and manually in terminal windows**, and **how to answer questions and handle errors smoothly during an interview or live demo**.

---

## 🏗️ Architecture Overview

Rubicon consists of 4 core services communicating via HTTP:

| Service | Technology | Port | Purpose |
|---|---|---|---|
| **Frontend** | React 18 + Vite + Leaflet | `3000` | Map-based assessment dashboard, file upload modal, AI chat panel |
| **Backend** | Node.js + Express + Mongoose | `4000` | REST API, JWT auth, assessment storage, IPFS/blockchain pipeline |
| **Agent Service** | Node.js + Express (Tool-Grounded) | `8001` | Conversational assistant querying live backend data |
| **AI Engine** | FastAPI + PyTorch + Mamba/Transformer | `8000` | Multimodal HSI + LiDAR inference (falls back to deterministic stub if offline) |
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
python -m uvicorn src.inference:app --host 0.0.0.0 --port 8000
```
> **Interview Talking Point:** *"The backend is designed with a resilient circuit breaker pattern. If the AI Engine is not started or reachable, the backend gracefully falls back to deterministic inference (`stubPredict`), preserving 100% of user workflow: upload, polygon mapping, verification, and chat."*

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
3. Click **Upload Scan** on the dashboard.
4. Upload test files (or any sample images/files).
5. Watch the dashboard plot the damage polygon on the Leaflet map with computed severity and risk metrics.
6. Open the **AI Assistant** chat panel on the right and ask:
   - *"How many assessments do we have?"*
   - *"Show me severe damage zones."*
   The assistant uses tool-grounded function calling against the live backend database to answer accurately.
