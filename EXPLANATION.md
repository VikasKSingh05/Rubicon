# Rubicon — Comprehensive System Architecture, Codebase Blueprint & Viva Guide

> **Project Mission**: Autonomous Multimodal Disaster Assessment with an Immutable Proof System. Rubicon ingests airborne/satellite hyperspectral (HSI) and LiDAR point clouds, performs damage severity classification via a hybrid Mamba–Transformer deep learning fusion architecture, pins raw sensory evidence to IPFS, records cryptographic proofs on the Polygon Amoy blockchain, renders interactive spatial GIS damage polygons on a Leaflet map, and provides a tool-grounded AI assistant for situational command.

---

## Table of Contents
1. [Core Problem & Value Proposition](#1-core-problem--value-proposition)
2. [End-to-End System Architecture & Data Flow](#2-end-to-end-system-architecture--data-flow)
3. [The Complete File-by-File Blueprint](#3-the-complete-file-by-file-blueprint)
   - [3.1 Root Configuration & Scripts](#31-root-configuration--scripts)
   - [3.2 Backend Service (`backend/`)](#32-backend-service-backend)
   - [3.3 Frontend Application (`frontend/`)](#33-frontend-application-frontend)
   - [3.4 Agent Service (`agent-service/`)](#34-agent-service-agent-service)
   - [3.5 Deep Learning AI Engine (`ai-engine/`)](#35-deep-learning-ai-engine-ai-engine)
   - [3.6 Smart Contracts & Schemas (`contracts/`)](#36-smart-contracts--schemas-contracts)
4. [The Machine Learning Deep Dive](#4-the-machine-learning-deep-dive)
5. [GIS & Spatial Mapping Mechanics](#5-gis--spatial-mapping-mechanics)
6. [IPFS & Blockchain Proof System](#6-ipfs--blockchain-proof-system)
7. [Agentic AI & Tool-Grounded Reasoning](#7-agentic-ai--tool-grounded-reasoning)
8. [Security, Authentication & Hardening](#8-security-authentication--hardening)
9. [Graceful Degradation & Circuit-Breaker Architecture](#9-graceful-degradation--circuit-breaker-architecture)
10. [Top 25 Viva / Technical Interview Questions & Answers](#10-top-25-viva--technical-interview-questions--answers)

---

## 1. Core Problem & Value Proposition

### The Problem
During catastrophic natural disasters (hurricanes, earthquakes, flash floods), emergency responders and insurance assessors face three critical hurdles:
1. **Sensor Limitations**: Standard RGB imagery fails to distinguish structural debris from surface discoloration (e.g., mud on an intact roof looks identical to a collapsed roof).
2. **Data Tampering & Fraud**: Post-disaster relief funds and insurance claims are frequently subjected to fraud, image fabrication, or retroactive metadata alterations.
3. **Fragmented Intelligence**: Decision-makers must manually correlate GIS map coordinates, sensory files, and statistical reports under severe time constraints.

### The Rubicon Solution
1. **Multimodal Fusion**: Combines **Hyperspectral Imagery (HSI)** (capturing hundreds of narrow contiguous spectral bands that reveal material composition) with **LiDAR** (capturing 3D point cloud elevation and structural volume collapse).
2. **Immutable Proof Chain**: Raw sensory files are content-addressed into IPFS CIDs. A cryptographic event is logged to the **Polygon Amoy testnet** using the `PramaanLedger` smart contract. Evidence cannot be forged, manipulated, or backdated.
3. **Tool-Grounded Conversational Agent**: An AI agent queries live MongoDB data via strict function calling, answering operational queries without hallucination.

---

## 2. End-to-End System Architecture & Data Flow

```mermaid
flowchart TD
    User([Emergency Responder / Browser]) -->|HTTP / TLS| Frontend["Frontend (React 18 + Vite :3000)"]
    
    subgraph Core System
        Frontend -->|REST API + httpOnly Cookie| Backend["Backend API (Express :4000)"]
        Frontend -->|Queries / Chat| Agent["Agent Service (Express :8001)"]
        
        Backend -->|Store User & Assessments| Mongo[("MongoDB :27017")]
        Backend -->|Spool Multipart Files| LocalStorage["Local Disk Storage (/storage)"]
        
        Backend -->|POST /predict (Multipart HSI+LiDAR)| AIEngine["AI Engine (FastAPI :8000)"]
        Backend -->|POST /satellite| AIEngineSat["Satellite Engine (:8002)"]
        
        Backend -->|Compute CIDv0 & Pin Buffer| IPFS["IPFS / Pinata Gateway"]
        Backend -->|logAssessment(tx)| Polygon["Polygon Amoy Testnet (Chain ID 80002)"]
        
        Agent -->|Internal Tool Calls (GET /assessments)| Backend
        Agent -->|LLM Function Calling| LLM["Anthropic Claude / OpenAI / OpenRouter"]
    end
    
    AIEngine -.->|If Offline| Stub["Deterministic AI Fallback (stubPredict)"]
    Polygon -.->|If No Gas/Key| SimChain["Deterministic Chain Simulation"]
    LLM -.->|If No API Key| FallbackAgent["Deterministic Heuristic Agent"]
```

### The 7-Step Lifecycle of a Disaster Assessment
1. **Upload & Magic-Byte Validation**: User submits an HSI file (`.tif`/`.tiff`) and LiDAR file (`.las`/`.laz`). The backend inspects the first 4 bytes (magic bytes: `II*\0` or `MM\0*` for TIFF, `LASF` for LiDAR) to verify authenticity before processing.
2. **File Spooling**: Files are streamed to a temporary directory (`os.tmpdir()/rubicon-*`) rather than held in RAM, preventing memory starvation during concurrent uploads.
3. **Inference Pipeline**: The backend sends multipart buffers to the FastAPI AI Engine (`http://localhost:8000/predict`). If unreachable, it triggers `stubPredict()`.
4. **H3 Spatial Indexing & GeoJSON Formulation**: The inference service maps pixel predictions to real-world WGS84 latitude/longitude bounding polygons and generates Uber H3 hexagonal spatial cells.
5. **IPFS Content-Addressing**: Cryptographic CIDv0 identifiers (`Qm...`) are computed via SHA-256 multihash and pinned to Pinata (or calculated locally if offline).
6. **On-Chain Notarization**: The backend signs an Ethereum transaction invoking `PramaanLedger.logAssessment(hsiCid, lidarCid, prediction)` on Polygon Amoy.
7. **Storage & Visualization**: The complete assessment record is persisted in MongoDB. The frontend dashboard updates its Leaflet map, drawing color-coded damage polygons (`None` = green, `Moderate` = yellow, `Severe Collapse` = red).

---

## 3. The Complete File-by-File Blueprint

### 3.1 Root Configuration & Scripts
* **[`package.json`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/package.json)**: Top-level manifest defining end-to-end smoke harness scripts (`e2e` and `e2e:setup`).
* **[`docker-compose.yml`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/docker-compose.yml)**: Multi-container orchestration definition for 5 services: `mongodb` (7.0.14), `ai-engine`, `ai-engine-satellite`, `agent-service`, `backend`, and `frontend`, with internal Docker DNS networking and health check dependencies.
* **[`.env.example`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/.env.example) / [`.env`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/.env)**: Centralized environment template with configurations for MongoDB connection URI, JWT secrets, Pinata JWT, Polygon RPC URL, contract addresses, and LLM provider credentials.
* **[`train.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/train.py)**: Standalone Python training pipeline for the Mamba-Transformer model on the Houston 2013 multimodal dataset.
* **[`eval.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/eval.py)**: Model evaluation script calculating Overall Accuracy (OA), Average Accuracy (AA), and Cohen's Kappa coefficient ($\kappa$).
* **[`scripts/serve.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/serve.ps1)**: Windows PowerShell runner script supporting `-Local` (spawns 4 dedicated PowerShell windows with execution policy bypass) or default Docker Compose.
* **[`scripts/setup.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/setup.ps1)**: Automated dependency installer invoking `npm.cmd install` across all Node services and handling Python pip installation gracefully.
* **[`scripts/test.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/test.ps1)**: Comprehensive multi-service test runner executing backend tests, agent-service tests, frontend Vitest tests, and AI Engine pytest.
* **[`sample-scans/`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/sample-scans)**: Staged valid test scans (`hsi.tiff` and `lidar.las`) formatted with valid magic bytes for immediate testing in the UI.

---

### 3.2 Backend Service (`backend/`)
* **[`backend/src/index.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/index.js)**: Server entry point. Validates production secrets, connects to MongoDB via Mongoose, binds port 4000, and implements graceful shutdown (`SIGTERM`/`SIGINT`) to drain active connections before disconnecting the database.
* **[`backend/src/app.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/app.js)**: Express application factory. Configures request ID tracing, structured JSON logging, Helmet security headers (with CSP for OpenStreetMap tiles), CORS with `credentials: true`, cookie parser, rate limiting, and route mounting.
* **[`backend/src/config.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/config.js)**: Central configuration module parsing environment variables with safe development defaults.
* **[`backend/src/db.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/db.js)**: Mongoose database connector implementing a 5000ms server selection timeout.
* **Routes (`backend/src/routes/`)**:
  - **[`auth.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/auth.js)**: User registration, login, logout, password change, and token refresh. Issues 15-minute JWT access tokens and 7-day rotating `httpOnly` refresh cookies.
  - **[`upload.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/upload.js)**: Multi-part disk spooling via Multer, magic-byte validation, AI inference coordination, IPFS CID hashing, on-chain logging, and file persistence.
  - **[`assessments.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/assessments.js)**: Pagination, filtering by severity, single assessment detail retrieval, H3 hexagonal cell extraction (`/cells`), and on-chain verification trigger (`/verify`).
  - **[`chain.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/chain.js)**: Exposes `/chain/status` to report current Polygon Amoy connectivity, contract address, and latest block.
  - **[`files.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/files.js)**: Secure streaming endpoint allowing authorized users to download original raw HSI and LiDAR files from storage.
  - **[`satellite.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/satellite.js)**: Accepts pre-disaster and post-disaster optical/satellite image pairs for building change detection and damage assessment.
  - **[`drone.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/routes/drone.js)**: High-resolution drone imagery triage endpoint linking targeted drone inspections to specific H3 hexagon cells.
* **Models (`backend/src/models/`)**:
  - **[`User.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/models/User.js)**: Mongoose schema storing email and bcrypt-hashed password (cost factor 10).
  - **[`RefreshToken.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/models/RefreshToken.js)**: Stores SHA-256 hashed refresh tokens with automatic MongoDB TTL index expiry.
  - **[`Assessment.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/models/Assessment.js)**: Comprehensive disaster schema storing GeoJSON polygons, prediction labels, confidence scores, class probabilities, IPFS CIDs, transaction hashes, and state machine status (`uploading` → `analyzed` → `chain_logged`).
* **Services (`backend/src/services/`)**:
  - **[`aiService.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/services/aiService.js)**: Dispatches multipart files to the FastAPI AI Engine. If unreachable or non-2xx, executes `stubPredict()` to generate valid GeoJSON damage polygons.
  - **[`chainService.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/services/chainService.js)**: Ethers.js integration for signing and broadcasting transactions to the `PramaanLedger` contract on Polygon Amoy. Simulates transactions deterministically if offline.
  - **[`ipfs.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/services/ipfs.js)**: Calculates deterministic IPFS CIDv0 identifiers using `crypto.createHash('sha256')` and `base58btc`. Dispatches buffers to Pinata if `PINATA_JWT` is present.
  - **[`assessmentService.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/services/assessmentService.js)**: Manages state machine transitions ensuring sequential lifecycle progression.
* **Middleware (`backend/src/middleware/`)**:
  - **[`auth.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/middleware/auth.js)**: JWT verification middleware extracting `Bearer <token>` from Authorization headers.
  - **[`logger.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/middleware/logger.js)**: Structured NDJSON logger tracking HTTP method, path, status, duration, and UUID request IDs.
  - **[`asyncHandler.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/backend/src/middleware/asyncHandler.js)**: Wrapper catching unhandled Promise rejections and routing them to Express error handlers.

---

### 3.3 Frontend Application (`frontend/`)
* **[`frontend/src/main.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/main.jsx)**: React root mounting `BrowserRouter` and `AuthProvider`. Loads Leaflet CSS and Tailwind styles.
* **[`frontend/src/App.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/App.jsx)**: Client-side routing table: `/login`, `/` (`DashboardPage`), and `/assessments/:id` (`AssessmentPage`).
* **Components (`frontend/src/components/`)**:
  - **[`MapView.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/MapView.jsx)**: Leaflet map component rendering OpenStreetMap tile layers, GeoJSON assessment polygons, H3 hexagon cells, dynamic camera centering (`FlyTo`), and interactive inspection popups.
  - **[`UploadModal.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/UploadModal.jsx)**: Multi-step upload modal featuring drag-and-drop file inputs, file extension checking, progress indicator (`Uploading` → `Analyzing` → `Verifying on-chain`), and automatic navigation upon completion.
  - **[`DataPanel.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/DataPanel.jsx)**: Detail view panel showing class probabilities, confidence progress bars, IPFS CID links, on-chain verification badges, and raw file download triggers.
  - **[`AgentChatSlot.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/AgentChatSlot.jsx)**: Slide-out AI assistant chat interface sending user queries to `http://localhost:8001/agent/query` and rendering tool calls with answers.
  - **[`DroneInspectorModal.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/DroneInspectorModal.jsx)**: Specialized triage modal allowing upload of high-resolution drone imagery targeted at individual damaged H3 hexagons.
  - **[`SeverityBadge.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/SeverityBadge.jsx)** & **[`ChainBadge.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/ChainBadge.jsx)**: Visual indicator tags for damage severity and Polygon Amoy verification status.
  - **[`TopBar.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/components/TopBar.jsx)**: Top navigation bar and assessment sidebar list.
* **Pages (`frontend/src/pages/`)**:
  - **[`DashboardPage.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/pages/DashboardPage.jsx)**: Master dashboard managing the assessment list, pagination, active selection, map display, and upload triggers.
  - **[`AssessmentPage.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/pages/AssessmentPage.jsx)**: Deep-dive single assessment inspection page with raw sensory downloads and live chain re-verification.
  - **[`LoginPage.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/pages/LoginPage.jsx)**: Unified authentication interface toggling between Login and Registration.
* **Context & Lib (`frontend/src/context/` & `lib/`)**:
  - **[`AuthContext.jsx`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/context/AuthContext.jsx)**: Authentication state provider maintaining short-lived access tokens in memory and initializing session restoration via refresh cookies on app boot.
  - **[`api.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/lib/api.js)**: Central HTTP client configuring `credentials: "include"`, automatic Authorization header injection, 401 interceptor with silent token rotation, and blob download utilities.
  - **[`severity.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/frontend/src/lib/severity.js)**: Color mapping utility defining standardized color tokens for `None` (green), `Moderate` (yellow), and `Severe Collapse` (red).

---

### 3.4 Agent Service (`agent-service/`)
* **[`agent-service/src/index.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/agent-service/src/index.js)**: Express microservice on port 8001 exposing `POST /agent/query`, `GET /health`, and `GET /tools`.
* **[`agent-service/src/tools.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/agent-service/src/tools.js)**: Defines the 4 grounding tools exposed to the agent:
  1. `get_recent_assessments(limit, severity)`: Fetches assessment records from backend.
  2. `get_assessment_detail(id)`: Fetches full metadata for a specific assessment.
  3. `get_chain_status()`: Queries backend Polygon Amoy synchronization state.
  4. `summarize_zone(assessmentId)`: Computes approximate geographical area (km²) and bounding box.
* **[`agent-service/src/providers.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/agent-service/src/providers.js)**: Multi-provider LLM executor supporting Anthropic (`claude-sonnet-4-5`), OpenAI (`gpt-4o`), and OpenRouter. Implements an autonomous tool-calling loop (up to 5 iterations) that executes tool calls against the live backend API before returning the final grounded response.
* **[`agent-service/src/fallback.js`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/agent-service/src/fallback.js)**: Zero-dependency deterministic heuristic fallback agent. When no LLM API key is present or the provider fails, this agent parses user intent (e.g. "how many", "severe", "verify", "detail") using keyword pattern matching, executes real tools against the backend, and formats factual answers.

---

### 3.5 Deep Learning AI Engine (`ai-engine/`)
* **[`ai-engine/src/inference.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/inference.py)**: FastAPI service on port 8000 exposing `POST /predict`. Receives HSI and LiDAR uploads, manages temporary files, loads model runtime, runs inference per patch, and returns GeoJSON polygons with class probabilities.
* **Model Architecture (`ai-engine/src/model/`)**:
  - **[`mamba.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/model/mamba.py)**: Selective State Space Model (SSM) processing the continuous spectral dimension of hyperspectral imagery in $O(N)$ linear time.
  - **[`fusion.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/model/fusion.py)**: Multimodal cross-attention fusion block. Combines continuous spectral feature vectors from the Mamba branch with spatial/elevation geometric features from the LiDAR branch.
  - **[`baseline.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/model/baseline.py)**: Standard ResNet / 2D CNN baseline model used for comparative performance benchmarking in research evaluations.
  - **[`factory.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/model/factory.py)**: Model initialization factory instantiating the requested architecture from config files.
  - **[`serving.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/model/serving.py)**: Runtime model loader. Loads weights (`model.pt`), PCA projection matrices (`pca.pkl`), and metadata (`metrics.json`), executing inference batches across spatial patches.
* **Preprocessing Pipeline (`ai-engine/src/preprocessing/`)**:
  - **[`load_hsi.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/preprocessing/load_hsi.py)**: Reads multi-band GeoTIFF files using Rasterio, normalizes reflectance values, and fits/applies 3D Principal Component Analysis (PCA) to reduce 144 spectral bands to 30 salient components.
  - **[`load_lidar.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/preprocessing/load_lidar.py)**: Reads LAS/LAZ point clouds via Laspy and rasterizes point elevations ($Z$), intensity, and return counts into aligned digital surface models (DSM).
  - **[`align.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/preprocessing/align.py)**: Reprojects and spatially resamples disparate HSI and LiDAR grids to matching spatial resolutions and bounding coordinates.
  - **[`patchify.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/preprocessing/patchify.py)**: Slices large raster extents into overlapping $11 \times 11$ or $15 \times 15$ pixel spatial patches for neural network input tensors.
  - **[`dataset.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/ai-engine/src/preprocessing/dataset.py)**: PyTorch `Dataset` and `DataLoader` implementations supporting data augmentation (random flips, rotations).

---

### 3.6 Smart Contracts & Schemas (`contracts/`)
* **[`contracts/contracts/PramaanLedger.sol`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/contracts/contracts/PramaanLedger.sol)**: Solidity 0.8.24 smart contract deployed to Polygon Amoy. Records an array of `LogEntry` structures:
  ```solidity
  struct LogEntry {
      string hsiCid;
      string lidarCid;
      string prediction;
      uint256 timestamp;
      address logger;
  }
  ```
  Emits an indexed event `AssessmentLogged(id, hsiCid, prediction, timestamp)`.
* **JSON API Contracts**:
  - `assessment.schema.json`: Formal JSON Schema defining the assessment document structure.
  - `predict.response.json`: AI Engine `/predict` contract definition.
  - `agent-query.response.json`: Agent Service `/agent/query` schema.
  - `satellite.response.json` & `drone.response.json`: Satellite and drone triage payload schemas.

---

## 4. The Machine Learning Deep Dive

### 4.1 Datasets: Houston 2013 vs. xBD (xView2)

Rubicon integrates **two distinct datasets** because real-world disaster management requires multiple tiers of resolution and modality:

| Metric | Houston 2013 (IEEE GRSS) | xBD / xView2 (DIUx / Maxar) |
|---|---|---|
| **Role in Rubicon** | Core Primary Multimodal Pipeline | Satellite Change Detection Pipeline |
| **Sensory Modality** | Hyperspectral Imagery (HSI) + Airborne LiDAR | Pre- and Post-Disaster Optical Satellite Imagery |
| **Spectral Depth** | 144 contiguous bands (380 nm – 1050 nm) + elevation | 3-channel RGB (pre-disaster) + 3-channel RGB (post-disaster) |
| **Spatial Resolution** | 2.5 m Ground Sample Distance (GSD) | 0.5 m – 0.8 m GSD ($1024 \times 1024$ image tiles) |
| **Target Classes** | 3 Classes: `None`, `Moderate`, `Severe Collapse` | 4 Classes: `No Damage`, `Minor Damage`, `Major Damage`, `Destroyed` |
| **Input Files** | `.tif`/`.tiff` (HSI) + `.las`/`.laz` (LiDAR) | `.png`/`.jpg` (Pre-disaster) + `.png`/`.jpg` (Post-disaster) |
| **Service Endpoint** | `POST /predict` (`ai-engine` port 8000) | `POST /satellite` / `POST /infer/satellite` (port 8002) |

---

### 4.2 Architectures: Mamba-Transformer vs. Siamese U-Net

Rubicon employs **both architectures**, each mathematically matched to its specific sensor modality:

```
                            ┌──────────────────────────────────────────────┐
                            │      RUBICON DUAL-TIER ML ARCHITECTURE       │
                            └──────────────────────────────────────────────┘
                                                    │
                ┌───────────────────────────────────┴───────────────────────────────────┐
                ▼                                                                       ▼
   TIER 1: MULTIMODAL AIRBORNE STREAM                                      TIER 2: OPTICAL SATELLITE STREAM
   Dataset: Houston 2013 (HSI + LiDAR)                                     Dataset: xBD / xView2 (Pre + Post)
   Model: Hybrid Mamba–Transformer Fusion                                  Model: Dual-Branch Siamese U-Net
   
   HSI (144 bands) ──> 3D PCA (30 comp) ──> Mamba SSM (S6)                 Pre-Event RGB ──> ResNet34 Encoder ┐ Shared
                                              │                                                               │ Weights
   LiDAR (Z/DSM)   ──> 2D CNN Encoder  ──> Spatial Tokens                 Post-Event RGB ──> ResNet34 Encoder ┘
                                              │                                                  │
                                              ▼                                                  ▼
                                    Cross-Attention Fusion                              Feature Difference & Concatenation
                                    (Q = HSI, K/V = LiDAR)                                       │
                                              │                                                  ▼
                                              ▼                                         U-Net Decoder with Skip Connections
                                    Classification Head (MLP)                                    │
                                              │                                        ┌─────────┴─────────┐
                                              ▼                                        ▼                   ▼
                                    3-Class Damage Polygon                   Building Localization   4-Class Damage Map
                                    (None / Moderate / Severe)               (Binary Mask)           (None/Minor/Major/Destroyed)
```

#### Architecture 1: Hybrid Mamba–Transformer Fusion (HSI + LiDAR)
* **File Location**: `ai-engine/src/model/` (`mamba.py`, `fusion.py`, `serving.py`)
* **Why Mamba?** Hyperspectral cubes contain 144+ continuous spectral bands. Standard Transformer self-attention scales with quadratic complexity $O(N^2)$ relative to the spectral sequence length. Mamba (Structured State Space Model - S6) processes continuous spectral bands with **linear complexity $O(N)$** and constant memory footprint, selectively propagating absorption features.
* **Why Cross-Attention?** Rather than naive concatenation, HSI spectral features act as Queries ($Q$), while LiDAR geometric elevation features act as Keys ($K$) and Values ($V$). This enables the model to ask: *"Given this abnormal spectral reflectance (e.g. crushed concrete dust), what is the corresponding 3D elevation delta (structural collapse) at this exact coordinate?"*

#### Architecture 2: Siamese U-Net (xBD / xView2 Satellite Stream)
* **File Location**: `ai_engine/model/` (`siamese_unet.py`), `train.py`, `eval.py`, `ai-engine/Dockerfile.satellite`
* **Why Siamese U-Net?** Disaster damage in optical satellite imagery is a **temporal change detection** problem. You cannot reliably identify building damage from a post-event image alone without knowing the baseline structural footprint before the event.
* **How It Operates**:
  1. A shared-weight Siamese encoder (ResNet34 or ResNet18) extracts multi-scale feature hierarchies from both the pre-disaster and post-disaster image.
  2. The network computes the absolute difference $|F_{\text{pre}} - F_{\text{post}}|$ and concatenates it with the raw representations.
  3. A two-headed U-Net decoder reconstructs the spatial segmentation:
     - **Localization Head**: Binary Cross-Entropy (BCE) + Dice loss to segment exact building footprints.
     - **Classification Head**: Focal Loss + Lovász Softmax to classify damage into the 4 xView2 categories.

#### Architecture 3: Stage 2 Drone Triage (`ai_engine/model/drone_serving.py`)
* Once satellite or HSI models identify damaged H3 hexagonal cells, responders upload ultra-high-resolution drone imagery targeted at individual cells via `POST /drone`.
* The drone computer vision module classifies structural failure modes:
  - `roof_collapse`: Dark voids, structural punctures, and planar displacement.
  - `facade_failure`: Vertical edge energy and shear degradation.
  - `debris_field`: High-frequency rubble and Laplacian scatter dispersion.

---

## 5. GIS & Spatial Mapping Mechanics

### Leaflet & GeoJSON Architecture
* **Coordinate Reference System (CRS)**: Raw sensor files use UTM (Universal Transverse Mercator) projections (e.g., UTM Zone 15N for Houston). The backend converts these coordinates to **WGS84 (EPSG:4326)** longitude and latitude for web mapping.
* **GeoJSON Representation**: Damage zones are stored as valid GeoJSON `Polygon` features:
  ```json
  {
    "type": "Polygon",
    "coordinates": [[[-95.35, 29.76], [-95.33, 29.76], [-95.33, 29.78], [-95.35, 29.78], [-95.35, 29.76]]]
  }
  ```
* **Color Tokens & Styling**:
  - `None`: Emerald Green (`#22C55E`)
  - `Moderate`: Amber/Yellow (`#EAB308`)
  - `Severe Collapse`: Crimson Red (`#DC2626`)
* **Uber H3 Hexagonal Hierarchical Spatial Index**:
  - To enable drone triage and localized search, the backend partitions damage zones into discrete H3 hexagonal cells (Resolution 8–9, ~100m–400m cell diameter).
  - Responders can click any individual hexagon on the map to inspect drone imagery specific to that cell.

---

## 6. IPFS & Blockchain Proof System

### How Evidence Tamper-Proofing Works
1. **CIDv0 Content-Addressing**:
   - When files are uploaded, Rubicon hashes the exact binary buffer using SHA-256.
   - The digest is prefixed with the multihash header (`0x12` for SHA-256, `0x20` for 32-byte length) and encoded in **Base58BTC** to produce a standard IPFS CIDv0 string (e.g. `QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco`).
   - If an attacker modifies even a single byte of the sensory image, the CID changes completely.
2. **On-Chain Notarization (`PramaanLedger.sol`)**:
   - The backend wallet broadcasts an Ethereum transaction to the Polygon Amoy testnet:
     `PramaanLedger.logAssessment(hsiCid, lidarCid, prediction)`.
   - The contract logs the entry with the blockchain's block timestamp and emits an `AssessmentLogged` event.
   - The transaction hash (`txHash`) is stored on the MongoDB assessment record.
3. **On-Chain Re-Verification**:
   - Anyone (insurers, government auditors, public citizens) can click **Re-verify** in the UI.
   - The backend queries the Polygon Amoy blockchain using Ethers.js, reads the immutable smart contract event log, and cryptographically confirms that the CIDs and prediction recorded on-chain match the database record.

---

## 7. Agentic AI & Tool-Grounded Reasoning

### Anti-Hallucination Guarantee
In disaster relief, an AI hallucinating casualty counts or building locations can be fatal. Rubicon strictly enforces **Tool-Grounded Reasoning**:
1. The system prompt instructs the agent that it possesses **no direct internal knowledge** of assessments.
2. The agent is equipped with 4 formal JSON tools:
   - `get_recent_assessments`
   - `get_assessment_detail`
   - `get_chain_status`
   - `summarize_zone`
3. When a user asks: *"How many severe zones are in Houston?"*, the model cannot guess. It must issue a tool call:
   `get_recent_assessments({ severity: "Severe Collapse" })`.
4. The agent service executes this tool call against the backend REST API, receives the exact JSON result from MongoDB, and generates an answer derived strictly from that data.

### Deterministic Heuristic Fallback Agent
If no external LLM API key (Anthropic/OpenAI/OpenRouter) is configured, Rubicon switches to `agent-service/src/fallback.js`. This deterministic agent parses incoming queries using intent pattern matching, executes the exact same underlying tools, and formats structured factual replies.

---

## 8. Security, Authentication & Hardening

1. **In-Memory JWT Access Tokens**:
   - Access tokens have a short lifespan (**15 minutes**).
   - They are stored strictly in JavaScript/React application memory (`AuthContext`), **never** in `localStorage` or `sessionStorage`. This prevents Cross-Site Scripting (XSS) attacks from stealing bearer credentials.
2. **Rotating httpOnly Refresh Cookies**:
   - Sessions are maintained across page refreshes via an `httpOnly`, `SameSite: strict` (or `lax`), `path: /auth` cookie named `rubicon_refresh`.
   - JavaScript cannot read or modify this cookie.
   - Upon token expiry, `frontend/src/lib/api.js` automatically intercepts 401 responses, silently calls `/auth/refresh`, receives a new access token, and retries the original request.
   - Refresh tokens are stored in MongoDB as **SHA-256 hashes**, preventing token reuse even if the database is leaked.
3. **Password Security**:
   - Passwords are salted and hashed using `bcryptjs` with 10 salt rounds.
4. **Rate Limiting**:
   - `express-rate-limit` guards sensitive routes: `/auth` (10 requests / 15 min), `/upload` (20 requests / 15 min).
5. **Magic Byte Validation**:
   - The backend reads the actual binary header of uploads before inference. Renaming a `.exe` or script to `.tif` fails magic byte validation.
6. **Cross-Origin Resource Sharing (CORS)**:
   - Configured with `credentials: true` and explicit origin allowlists (`http://localhost:3000`, `http://127.0.0.1:3000`) to enable secure cookie transmission without exposing API endpoints to unauthorized domains.

---

## 9. Graceful Degradation & Circuit-Breaker Architecture

| Layer | Primary Technology | Degradation Trigger | Fallback Behavior |
|---|---|---|---|
| **AI Inference** | PyTorch Mamba-Transformer (`:8000`) | Service offline, model missing, or CUDA OOM | Automatically invokes `stubPredict()`, returning contract-identical randomized Houston damage polygons. |
| **IPFS Pinning** | Pinata IPFS Cloud Gateway | `PINATA_JWT` unset or API rate limited | Generates deterministic cryptographic CIDv0 locally via SHA-256 + base58; marks upload as local-only. |
| **Blockchain** | Polygon Amoy Testnet (Chain 80002) | RPC offline, wallet unfunded, or key unset | Writes a deterministic simulated on-chain record; flags record as `simulated: true` without failing upload. |
| **AI Assistant** | Claude Sonnet / GPT-4o | API key unset or network timeout | Switches to `fallback.js` deterministic heuristic agent, answering strictly from live database tool calls. |

---

## 10. Top 25 Viva / Technical Interview Questions & Answers

### Architecture & System Design

#### Q1: What is Rubicon in one sentence?
**Answer:** Rubicon is an autonomous multimodal disaster assessment platform that combines Hyperspectral and LiDAR sensor imagery via a hybrid Mamba–Transformer model, records tamper-proof cryptographic evidence to IPFS and Polygon blockchain, and visualizes damage via a GIS map dashboard and tool-grounded AI assistant.

#### Q2: Why did you divide the system into four separate microservices instead of a single monolith?
**Answer:** The architecture separates distinct computational concerns:
- `ai-engine` requires a Python runtime, CUDA GPU drivers, and native geospatial C libraries (GDAL/Rasterio).
- `backend` requires high-concurrency Node.js asynchronous I/O to manage HTTP streams, MongoDB transactions, and blockchain event polling.
- `agent-service` isolates external LLM latency and streaming token responses from core API operations.
- `frontend` is decoupled as a static React SPA built with Vite for optimal CDN delivery.

#### Q3: How does the system handle high traffic or service failures during a real disaster?
**Answer:** Rubicon uses circuit breakers and graceful degradation at every layer:
- Uploads are spooled to disk, preventing memory overflow.
- If the AI engine is unreachable, the backend degrades to `stubPredict()`.
- If Polygon Amoy RPC is congested, the system logs a deterministic simulated hash and marks state as `analyzed`.
- If external LLMs are down, the deterministic heuristic agent answers user questions directly from MongoDB.

---

### Machine Learning & Data Processing

#### Q4: Why use both Hyperspectral (HSI) and LiDAR instead of standard satellite photography?
**Answer:** Standard RGB satellite imagery only captures 3 visible spectral bands (Red, Green, Blue) and lacks depth. HSI captures hundreds of contiguous spectral bands that distinguish specific chemical and material compositions (e.g., exposed asbestos, fractured concrete, waterlogged drywall). LiDAR adds precise 3D spatial elevations $(X, Y, Z)$ to directly detect structural volume collapse and debris height deltas.

#### Q5: What is Mamba, and why is it preferred over a standard Transformer for Hyperspectral bands?
**Answer:** Mamba is a State Space Model (SSM) featuring input-dependent selection (S6). Traditional Transformers suffer from quadratic computational complexity $O(N^2)$ relative to sequence length. Because hyperspectral pixels contain 144+ continuous spectral bands, processing them with full self-attention is computationally expensive and memory-intensive. Mamba operates with linear time complexity $O(N)$ and constant-state memory, modeling continuous spectral dependencies much more efficiently.

#### Q6: How does the Cross-Attention Fusion module work?
**Answer:** In `ai-engine/src/model/fusion.py`, HSI spectral features from the Mamba branch are used as Queries ($Q$), while LiDAR geometric/elevation features act as Keys ($K$) and Values ($V$). This allows the network to dynamically weight spectral material signatures based on structural height changes before feeding the fused representation to the classification MLP.

#### Q7: How do you reduce the high dimensionality of 144 hyperspectral bands?
**Answer:** We apply 3D Principal Component Analysis (PCA) in `ai-engine/src/preprocessing/load_hsi.py`, projecting the 144 spectral bands down to the 30 most salient orthogonal components while retaining over 98% of the spectral variance.

#### Q8: What dataset was used, and what are the target damage classes?
**Answer:** The model is trained on the IEEE GRSS Houston 2013 multimodal dataset, which features co-registered hyperspectral imagery (144 bands, 380–1050 nm) and LiDAR elevation data. The damage assessment classes are `None` (intact), `Moderate` (partial damage), and `Severe Collapse` (complete structural destruction).

---

### GIS & Spatial Mapping

#### Q9: How do you map pixel coordinate predictions to real-world latitude and longitude on Leaflet?
**Answer:** The raw GeoTIFF and LAS files contain spatial metadata (affine transformation matrices and UTM projections). We reproject UTM Zone 15N meter coordinates into standard WGS84 (EPSG:4326) degrees latitude and longitude. The bounding coordinates are serialized into a GeoJSON `Polygon` and rendered as a vector layer on OpenStreetMap using React-Leaflet.

#### Q10: What is Uber H3, and why is it used in this project?
**Answer:** Uber H3 is a hexagonal hierarchical spatial indexing system. Unlike square grid partitions, every neighboring hexagon in an H3 grid is equidistant, eliminating directional distortion. Rubicon indexes disaster damage zones into H3 cells so emergency responders can deploy targeted drone inspections (`/drone`) to specific hexagonal sectors.

---

### Blockchain & Cryptography

#### Q11: Why is blockchain necessary in a disaster assessment application?
**Answer:** Disaster recovery involves billions of dollars in insurance payouts and municipal relief funding, creating strong incentives for post-event insurance fraud, altered assessment dates, or fabricated imagery. By recording the IPFS cryptographic hash (CID) of the raw sensor data and model prediction to Polygon Amoy, Rubicon creates an unalterable, publicly auditable, timestamped proof of evidence.

#### Q12: What exact data is stored on the Polygon blockchain?
**Answer:** We do **not** store large sensory files directly on-chain due to gas costs and block size limits. Instead, we store a lightweight `LogEntry` struct in `PramaanLedger.sol` containing:
- `hsiCid`: IPFS content identifier string for the HSI file.
- `lidarCid`: IPFS content identifier string for the LiDAR file.
- `prediction`: Damage classification string (`Severe Collapse`, etc.).
- `timestamp`: Block timestamp.
- `logger`: Ethereum address of the recording authority.

#### Q13: What is CIDv0, and how is it generated?
**Answer:** An IPFS CIDv0 is a content identifier starting with `Qm...`. It is generated by taking the SHA-256 hash of the file buffer, prepending the multihash function code (`0x12`) and length (`0x20`), and encoding the result in Base58BTC. Because it is purely content-addressed, any alteration to the underlying sensory file produces a completely different CID.

---

### AI Agent & Grounding

#### Q14: How do you prevent the AI assistant from hallucinating disaster statistics?
**Answer:** Through strict tool grounding. The system prompt instructs the agent that it has zero intrinsic knowledge of disaster assessments. It is provided with JSON tool definitions (`get_recent_assessments`, `get_assessment_detail`, `get_chain_status`, `summarize_zone`). The agent must make a tool call to query the backend REST API, and its response must be derived exclusively from the returned JSON.

#### Q15: How does the deterministic fallback agent work when no LLM API key is present?
**Answer:** In `agent-service/src/fallback.js`, regex pattern matching analyzes the user's intent (e.g. keywords like *"how many"*, *"severe"*, *"verify"*, *"assessment ID"*). The script invokes the appropriate backend tool directly and formats the output into structured text, guaranteeing that the chat panel functions without external API dependencies.

---

### Security, Web Development & Troubleshooting

#### Q16: Why did Sign Up and Login fail with "Failed to fetch", and how was it fixed?
**Answer:** In `frontend/src/lib/api.js`, the fetch request includes `credentials: "include"` so the browser sends and receives `httpOnly` authentication cookies. Under the W3C CORS specification, if a credentialed request is made across origins, the server **must** return the header `Access-Control-Allow-Credentials: true`. Because the backend previously had `app.use(cors({ origin }))` without `credentials: true`, the browser blocked the response and threw `TypeError: Failed to fetch`. Setting `credentials: true` on both the backend and agent-service resolved the issue.

#### Q17: Where are authentication tokens stored on the frontend, and why?
**Answer:** The JWT access token is kept strictly in **JavaScript React memory** (`AuthContext`). It is **never** saved in `localStorage` or `sessionStorage` to eliminate the risk of token exfiltration via Cross-Site Scripting (XSS). Long-term session persistence is managed by an `httpOnly`, `SameSite` refresh cookie that cannot be read by client-side scripts.

#### Q18: What are magic bytes, and why does Rubicon validate them?
**Answer:** File extensions (like `.tif` or `.las`) can easily be spoofed by renaming malicious scripts or binaries. In `backend/src/routes/upload.js`, `validateMagic()` inspects the first 4 bytes of the uploaded buffer:
- TIFF files must start with `0x49 0x49 0x2A 0x00` (little-endian `II*`) or `0x4D 0x4D 0x00 0x2A` (big-endian `MM*`).
- LAS point cloud files must start with the ASCII signature `LASF`.
This ensures that only authentic sensor files enter the processing pipeline.

#### Q19: Why was `npm install` failing on Windows PowerShell, and how was it solved?
**Answer:** Windows PowerShell restricts the execution of unsigned `.ps1` scripts by default under its system `ExecutionPolicy`. When running `npm`, PowerShell attempts to load `npm.ps1`. Invoking `npm.cmd` directly or wrapping execution with `-ExecutionPolicy Bypass` bypasses this limitation.

#### Q20: How does the backend prevent memory exhaustion when users upload large 100 MB files?
**Answer:** Rather than storing incoming file buffers in RAM, `backend/src/routes/upload.js` uses Multer's `diskStorage` engine to spool incoming streams directly to a temporary directory on disk (`os.tmpdir()/rubicon-*`). The files are read and processed in controlled streams and deleted in a `finally` block once persisted to `/storage`.

#### Q21: What is silent token refresh, and how is it implemented?
**Answer:** In `frontend/src/lib/api.js`, if an API call receives an HTTP 401 Unauthorized, the fetch wrapper automatically halts, invokes `POST /auth/refresh` using the `httpOnly` cookie to obtain a fresh access token, updates the in-memory token state, and retries the original request seamlessly without redirecting the user to the login screen.

#### Q22: What is the purpose of the `/metrics` endpoint on the backend?
**Answer:** Located at `backend/src/app.js`, `/metrics` provides a Prometheus-compatible plain-text metrics exposition tracking service uptime in seconds, total uploads accepted, completed blockchain proofs, and total application errors.

#### Q23: Why do we have separate `Dockerfile` and `Dockerfile.satellite` in `ai-engine/`?
**Answer:** The main `ai-engine` Dockerfile packages PyTorch, Mamba, and the multimodal fusion pipeline for HSI and LiDAR. `Dockerfile.satellite` packages a specialized computer vision pipeline tailored for optical satellite image pairs (`pre` and `post` disaster imagery) using change detection and building damage segmentation models.

#### Q24: What is the purpose of `tests/auth-hardening.test.js` in the backend?
**Answer:** It uses `supertest` and `mongodb-memory-server` to rigorously verify session security:
- Verifies that access tokens expire within the TTL window.
- Verifies that refresh tokens are stored as SHA-256 hashes in MongoDB.
- Confirms that password changes invalidate all existing refresh tokens except the caller's active session.
- Ensures rate limiting triggers HTTP 429 when abuse thresholds are exceeded.

#### Q25: If an interviewer asks you to demonstrate the application right now, what exact steps will you take?
**Answer:**
1. Navigate to **`http://localhost:3000`** in the browser.
2. Register an account (e.g. `evaluator@rubicon.dev`).
3. Click **Upload Scan**, drag in `sample-scans/hsi.tiff` and `sample-scans/lidar.las`, and click **Start Assessment**.
4. Show the multi-step progress bar transitioning through Uploading, Analyzing, and On-Chain Verification.
5. Demonstrate the newly plotted damage polygon on the Leaflet map over Houston, clicking it to show severity, confidence percentage, and blockchain transaction hash.
6. Open the AI Assistant chat panel on the right, ask *"How many assessments do we have?"*, and show how the agent executes a live tool call against the backend to return an accurate, grounded answer.
