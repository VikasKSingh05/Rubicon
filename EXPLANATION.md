# Rubicon — Comprehensive System Architecture, Codebase Blueprint & Viva Guide

> **Project Mission**: Disaster assessment with an immutable proof system. The primary satellite prototype ingests paired pre- and post-disaster optical imagery, performs building localization and four-level xBD damage classification with a **Dual-Task Siamese U-Net**, and connects the resulting assessment to the map, optional IPFS/Polygon proof flow, and tool-grounded AI assistant. The encoder is selected from the available hardware profile (ResNet-18 on CPU, ResNet-34 on supported GPU profiles).

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
During catastrophic natural disasters (hurricanes, earthquakes, wildfires, flash floods), emergency responders and insurance assessors face three critical hurdles:
1. **Change Detection Complexity**: Standard post-event photography alone cannot identify collapsed or destroyed structures without an accurate pre-disaster reference baseline footprint.
2. **Data Tampering & Fraud**: Post-disaster relief funds and insurance claims are frequently subjected to fraud, image fabrication, or retroactive metadata alterations.
3. **Fragmented Intelligence**: Decision-makers must manually correlate GIS map coordinates, sensory files, and statistical reports under severe time constraints.

### The Rubicon Solution
1. **Dual-Task Siamese U-Net**: Ingests paired pre- and post-disaster optical satellite tiles, sharing weights across a ResNet-34 encoder, fusing multi-scale feature differences, and using dual decoder heads to simultaneously localize building footprints and classify damage into 4 discrete xBD severity levels (`No Damage`, `Minor Damage`, `Major Damage`, `Destroyed`).
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
        
        Backend -->|POST /infer/satellite (pre + post)| AIEngineSat["xBD Satellite Engine (FastAPI :8002)"]
        Backend -.->|Legacy HSI/LiDAR POST /predict| AIEngine["Legacy Engine (FastAPI :8000)"]
        
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
1. **Upload & Magic-Byte Validation**: User submits paired pre-disaster and post-disaster optical satellite scans (`.png`/`.jpg`/`.tif`). The backend inspects the header magic bytes (e.g. PNG signature `\x89PNG`, JPEG `\xff\xd8`, or TIFF `II*\0` / `MM\0*`) to verify authenticity before processing.
2. **File Spooling**: Files are streamed to a temporary directory (`os.tmpdir()/rubicon-*`) rather than held in RAM, preventing memory starvation during concurrent uploads.
3. **Inference Pipeline**: The backend routes the pre/post image pair to the Siamese U-Net inference engine (`/infer/satellite`). If unreachable, it triggers `stubPredict()` returning xBD class distributions.
4. **H3 Spatial Indexing & GeoJSON Formulation**: The inference service generates building polygons and maps damage distributions to WGS84 latitude/longitude bounding polygons and Uber H3 hexagonal spatial cells.
5. **IPFS Content-Addressing**: Cryptographic CIDv0 identifiers (`Qm...`) are computed via SHA-256 multihash for both pre- and post-disaster images and pinned to Pinata (or calculated locally if offline).
6. **On-Chain Notarization**: The backend signs an Ethereum transaction invoking `PramaanLedger.logAssessment(preCid, postCid, prediction)` on Polygon Amoy.
7. **Storage & Visualization**: The complete assessment record is persisted in MongoDB. The frontend dashboard updates its Leaflet map, drawing color-coded damage polygons (`No Damage` = green, `Minor Damage` = yellow, `Major Damage` = orange, `Destroyed` = red).

---

## 3. The Complete File-by-File Blueprint

### 3.1 Root Configuration & Scripts
* **[`package.json`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/package.json)**: Top-level manifest defining end-to-end smoke harness scripts (`e2e` and `e2e:setup`).
* **[`docker-compose.yml`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/docker-compose.yml)**: Multi-container orchestration definition for 5 services: `mongodb` (7.0.14), `ai-engine`, `ai-engine-satellite`, `agent-service`, `backend`, and `frontend`, with internal Docker DNS networking and health check dependencies.
* **[`.env.example`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/.env.example) / [`.env`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/.env)**: Centralized environment template with configurations for MongoDB connection URI, JWT secrets, Pinata JWT, Polygon RPC URL, contract addresses, and LLM provider credentials.
* **[`train.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/train.py)**: Standalone Python training pipeline for the Dual-Task Siamese U-Net model on the xBD dataset (featuring auto-hardware profiling, AMP fp16, Cosine schedule, and automatic CUDA OOM crop-size halving).
* **[`eval.py`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/eval.py)**: Model evaluation script calculating xView2 building localization $F_1$, damage classification $F_1$, and harmonic mean score.
* **[`scripts/serve.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/serve.ps1)**: Windows PowerShell runner script supporting `-Local` (spawns 4 dedicated PowerShell windows with execution policy bypass) or default Docker Compose.
* **[`scripts/setup.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/setup.ps1)**: Automated dependency installer invoking `npm.cmd install` across all Node services and handling Python pip installation gracefully.
* **[`scripts/test.ps1`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/scripts/test.ps1)**: Comprehensive multi-service test runner executing backend tests, agent-service tests, frontend Vitest tests, and AI Engine pytest.
* **[`sample-scans/`](file:///c:/Users/Lenovo/Documents/Projects/Rubicon/Rubicon/sample-scans)**: Staged valid test scans formatted with valid magic bytes for immediate testing in the UI.

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
  - **`backend/src/routes/files.js`**: Secure, owner-scoped streaming endpoint for persisted upload originals. HSI/LiDAR file types are retained for the legacy scan contract; xBD assessments use pre/post optical images.
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
* **Primary xBD model**: `ai-engine/ai_engine/model/siamese_unet.py` defines the shared ResNet encoder, multi-scale temporal feature fusion, U-Net decoder, and localization/damage heads. `ai-engine/ai_engine/training/train.py` provides training; the repository-root `train.py` CLI writes best, last, and serving checkpoints plus `training_metrics.json`.
* **xBD data and evaluation**: `ai-engine/ai_engine/data/` discovers labeled pre/post image pairs, rasterizes xBD polygon labels, and creates disaster-event-disjoint train/validation splits. The repository-root `eval.py` evaluates `val`, `hold`, or `test` and writes metrics, a confusion matrix, and prediction overlays. `hold/` is a supplemental tile-level split, not an unseen-event test.
* **Satellite inference**: `ai-engine/ai_engine/serve.py` serves `POST /infer/satellite` on port 8002. It loads `siamese_unet.pt` from `RUBICON_MODEL_DIR` (default `ai-engine/data/models/xbd`) and returns building/cell GeoJSON. `Dockerfile.satellite` packages this service. If no checkpoint exists, the endpoint responds with a clearly versioned stub; it is not model inference.
* **Separate legacy experiment**: `ai-engine/src/` contains the Houston 2013 HSI/LiDAR preprocessing and Mamba/Transformer model exposed through the legacy `/predict` service on port 8000. It is not the xBD Siamese U-Net and its model/checkpoint/metrics must not be used to claim xBD training results.

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

### 4.1 Primary Dataset: xBD (xView2)

The xBD pipeline uses paired pre-disaster and post-disaster optical RGB satellite
tiles with polygon labels. HSI/LiDAR refers to a separate legacy experiment,
not an input to this Siamese U-Net.

| Property | xBD implementation |
|---|---|
| **Inputs** | Paired pre/post RGB images (`.png`, `.jpg`, `.tif`, `.tiff`) and xBD JSON labels |
| **Network input** | Three-channel pre and post tensors; crop size selected by hardware profile |
| **Encoder** | Shared ResNet-18 (CPU profile) or ResNet-34 (GPU profiles) |
| **Outputs** | Binary building-localization mask and five-way per-pixel damage logits |
| **Damage channels** | `0: background`, `1: no-damage`, `2: minor-damage`, `3: major-damage`, `4: destroyed` |
| **Training entry point** | `python train.py --xbd-root <path>` from `ai-engine/` |
| **Evaluation entry point** | `python eval.py --xbd-root <path> --split val` from `ai-engine/` |
| **Inference endpoint** | `POST /infer/satellite` on the satellite engine (`:8002`) |

---

### 4.2 Research Paper Architecture: Dual-Task Siamese U-Net

Rubicon implements the dual-task Siamese U-Net change-detection architecture illustrated in the research paper draft workflow:

```
                            ┌────────────────────────────────────────────────────────┐
                            │      RUBICON SIAMESE U-NET RESEARCH ARCHITECTURE       │
                            └────────────────────────────────────────────────────────┘
                                                        │
                   ┌────────────────────────────────────┴────────────────────────────────────┐
                   ▼                                                                         ▼
     Pre (Satellite Image) [B, 3, H, W]                                        Post (Satellite Image) [B, 3, H, W]
                   │                                                                         │
                   ▼                                                                         ▼
      ┌─────────────────────────┐                                               ┌─────────────────────────┐
      │  Backbone (ResNet34)    │ ◀ ─ ─ ─ ─ ─ ─  Shared Weights  ─ ─ ─ ─ ─ ─ ─ ─│  Backbone (ResNet34)    │
      └─────────────────────────┘                                               └─────────────────────────┘
         │    │    │    │                                                          │    │    │    │
        F₁   F₂   F₃   F₄                                                         F₁   F₂   F₃   F₄
         │    │    │    │                                                          │    │    │    │
         └────┼────┼────┼───────────────────────┐         ┌────────────────────────┘    │    │    │
              │    │    │                       ▼         ▼                             │    │    │
              │    │    │                ┌─────────────────────────┐                    │    │    │
              │    │    │                │  Multi-Scale Feature    │                    │    │    │
              │    │    │                │  Fusion:                │                    │    │    │
              │    │    │                │  [F_pre, F_post, |Diff|]│                    │    │    │
              │    │    │                └─────────────────────────┘                    │    │    │
              │    │    │                             │                                 │    │    │
              ▼    ▼    ▼                             ▼                                 ▼    ▼    ▼
          (Multi-scale skip connections to decoder at stages 1, 2, 3, 4)
                                                      │
                                                      ▼
                                       ┌───────────────────────────────┐
                                       │   Decoder (U-Net style)       │
                                       └───────────────────────────────┘
                                                      │
                                      ┌───────────────┴───────────────┐
                                      ▼                               ▼
                      ┌───────────────────────────────┐ ┌───────────────────────────────┐
                      │   Building Localization Head  │ │  Damage Classification Head   │
                      │   (1 Channel, Sigmoid)        │ │  (5 Classes, Softmax)         │
                      └───────────────────────────────┘ └───────────────────────────────┘
                                      │                               │
                                      ▼                               ▼
                                 L_loc (BCE + Dice)            L_dmg (Weighted CE + Focal)
                                      │                               │
                                      └───────────────┬───────────────┘
                                                      ▼
                                      L_total = 0.4 * L_loc + 0.6 * L_dmg
```

#### Dual Heads and Loss Formulation
1. **Building Localization Head**:
   - Single-channel output followed by Sigmoid activation, generating the binary building footprint mask.
   - Loss function:
     $$\mathcal{L}_{\text{loc}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$
2. **Damage Classification Head**:
   - 5-channel output (Background, No Damage, Minor Damage, Major Damage, Destroyed) with Softmax activation.
   - Loss function combines class-weighted Cross-Entropy with Focal Loss ($\gamma = 2.0$) to counter severe class imbalance in xBD:
     $$\mathcal{L}_{\text{dmg}} = \mathcal{L}_{\text{CE, weighted}} + \mathcal{L}_{\text{Focal}}$$
3. **Joint Multi-Task Objective**:
   $$\mathcal{L}_{\text{total}} = 0.4 \times \mathcal{L}_{\text{loc}} + 0.6 \times \mathcal{L}_{\text{dmg}}$$

#### Training Pipeline (`train.py` & `ai_engine/model/`):
- **Hardware Auto-Detection**: Dynamic profiles for Local RTX 3050 (6GB VRAM, crop size 512 with automatic fallback to 384/256 on CUDA OOM), Kaggle GPU (16GB+ VRAM), and CPU fallback.
- **Optimization**: AdamW (default learning rate $3 \times 10^{-4}$, weight decay $10^{-4}$), one-epoch warmup followed by cosine scheduling, localization/damage loss weights 0.4/0.6, focal gamma 2.0, and gradient clipping (max norm 1.0). AMP is enabled on supported CUDA profiles.
- **Validation Metric**: Comprehensive xView2 evaluation metric ($F_1^{\text{loc}}$ and harmonic mean $F_1^{\text{dmg}}$ across damage classes).
- **Reproducibility report**: `training_metrics.json` records actual data mode, tiles/events, hardware, parameter count, hyperparameters, epoch history, best epoch, and checkpoint hash. The evaluator writes separate results for `val`, `hold`, and `test`.

### 4.3 Exploratory Data Analysis (EDA)

The EDA pipeline (`ai-engine/scripts/eda_xbd.py` and `ai-engine/notebooks/eda.ipynb`) profiles the xBD dataset before training:

| EDA Analysis | What It Computes |
|---|---|
| **Class Distribution** | Counts and percentages for No Damage, Minor Damage, Major Damage, Destroyed |
| **Polygon Area Statistics** | Mean and median building footprint pixel area per damage class |
| **Temporal Image Delta** | Pre vs Post image intensity difference `\|I_post - I_pre\|` distribution |
| **Disaster Event Representation** | Frequency of each disaster event (hurricanes, earthquakes, floods) |
| **Optimal Class Weights** | Inverse-frequency weights $w_c = \frac{N}{K \cdot N_c}$ for `RubiconLoss` |

Key finding: **No Damage** dominates destroyed/major by roughly **10:1**, empirically justifying Focal Loss ($\gamma = 2.0$) + Weighted Cross-Entropy.

Run the EDA:
```bash
# Standalone CLI script
python ai-engine/scripts/eda_xbd.py --synthetic
# Or use the Jupyter notebook
jupyter notebook ai-engine/notebooks/eda.ipynb
```

### 4.4 Hyperparameter Tuning Pipeline

The xBD tuning script (`ai-engine/scripts/tune_hyperparams.py`) samples the
following candidate space. These are brief trial runs (a few batches per trial),
not full model training and not final performance evidence.

| Hyperparameter | Search Values |
|---|---|---|
| **Encoder Backbone** | `resnet18`, `resnet34` |
| **Learning Rate** | $1 \times 10^{-4}$, $3 \times 10^{-4}$, $5 \times 10^{-4}$ |
| **Localization Loss Weight** | 0.3, 0.4, 0.5 (damage weight is the remainder) |
| **Focal $\gamma$** | 1.5, 2.0, 2.5 |
| **Weight Decay** | $1 \times 10^{-4}$, $1 \times 10^{-3}$ |

The script ranks candidates by its reported validation xView2 score. The full
training CLI's defaults and run artifacts are documented in
[`docs/training.md`](docs/training.md).

Run a short synthetic pipeline check:
```bash
python ai-engine/scripts/tune_hyperparams.py --synthetic --trials 5 --dry-run
```

For a real dataset use `--xbd-root <path> --trials 5 --output <json-path>`.
The tuning CLI does not accept `--epochs`, `--data-dir`, or `--device`.

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
* **Color Tokens & Styling** (xBD Damage Severity Palette):
  - `No Damage`: Emerald Green (`#22C55E`)
  - `Minor Damage`: Yellow (`#EAB308`)
  - `Major Damage`: Amber (`#F59E0B`)
  - `Destroyed`: Crimson Red (`#DC2626`)
  - `On-Chain Verified`: Polygon Violet (`#7C3AED`)
* **Global Trained xBD Disaster Corpus & GIS Mapping**:
  - The `MapView` component includes the full geographic mapping for 10 benchmark disaster training regions:
    1. 🌲 **Santa Rosa Wildfire** (California, USA — 38.44°N, 122.71°W)
    2. 🌀 **Hurricane Michael** (Mexico Beach, FL, USA — 29.94°N, 85.42°W)
    3. 🏢 **Mexico City Earthquake** (Puebla, Mexico — 19.34°N, 99.15°W)
    4. 🌋 **Volcán de Fuego Eruption** (Guatemala — 14.47°N, 90.88°W)
    5. 🌊 **Palu Earthquake & Tsunami** (Sulawesi, Indonesia — 0.89°S, 119.87°E)
    6. 🌾 **Midwest River Basin Floods** (Nebraska/Iowa, USA — 41.25°N, 95.93°W)
    7. 🌪️ **Joplin EF5 Tornado Corridor** (Missouri, USA — 37.08°N, 94.51°W)
    8. 🔥 **Woolsey Wildfire** (Malibu/LA, California — 34.02°N, 118.77°W)
    9. 🇦🇺 **Pinery Bushfire** (South Australia — 34.33°S, 138.56°E)
    10. 🇵🇹 **Pedrógão Grande Wildfire** (Portugal — 39.91°N, 8.14°W)
* **GIS Concentric Radial Intensity Heatmaps**: Renders 4-tier concentric gradient circles (750m, 480m, 260m, 110m radiuses) centered on disaster epicenters, depicting damage diffusion from the core collapse zone.
* **Responsive Non-Overlapping Layout Architecture**:
  - A unified flex overlay bar across the top of the map positions the **xBD Scenes Quick-Jump Pills** on the left and the **GIS Layer Controls** (`All Layers`, `🔥 Heatmap`, `⬡ H3`, `🏢 Footprints`, and `xBD Data`) on the right, eliminating header crowding and control collisions.
  - Leaflet zoom controls (`+ / -`) are cleanly offset below the top toolbar.
  - The bottom-left displays a unified, single **xBD Damage & Heatmap Legend** (`SeverityLegend.jsx`), avoiding duplicate legend collisions.
* **Database Seeding (`backend/scripts/seed_xbd.js`)**: Seeds realistic xBD pre/post satellite scene assessments with building footprints, class probabilities, IPFS CIDs, and Polygon Amoy transaction proofs.
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
     `PramaanLedger.logAssessment(preCid, postCid, prediction)`.
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
| **Satellite Inference (xBD)** | Siamese U-Net (`:8002`) | Checkpoint absent | Returns a versioned empty stub response; this is not trained-model inference. |
| **Legacy Inference** | HSI/LiDAR Mamba-Transformer (`:8000`) | Legacy service/model unavailable | Uses the separate contract-compatible `stubPredict()` fallback; these results are not xBD predictions. |
| **IPFS Pinning** | Pinata IPFS Cloud Gateway | `PINATA_JWT` unset or API rate limited | Generates deterministic cryptographic CIDv0 locally via SHA-256 + base58; marks upload as local-only. |
| **Blockchain** | Polygon Amoy Testnet (Chain 80002) | RPC offline, wallet unfunded, or key unset | Writes a deterministic simulated on-chain record; flags record as `simulated: true` without failing upload. |
| **AI Assistant** | Claude Sonnet / GPT-4o | API key unset or network timeout | Switches to `fallback.js` deterministic heuristic agent, answering strictly from live database tool calls. |

---

## 10. Top 25 Viva / Technical Interview Questions & Answers

### Architecture & System Design

#### Q1: What is Rubicon in one sentence?
**Answer:** Rubicon's primary computer-vision prototype assesses paired pre- and post-disaster optical imagery with an xBD Siamese U-Net, then presents assessment results through a GIS dashboard and tool-grounded assistant. Optional IPFS and Polygon Amoy integrations provide evidence records. A separate legacy HSI/LiDAR experiment remains in the repository but is not the xBD model.

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

#### Q4: Why use dual-temporal optical satellite pairs (xBD) instead of single post-disaster imagery?
**Answer:** Disaster assessment is fundamentally a temporal change detection problem. Single post-event images cannot reliably distinguish whether an empty lot was always empty or previously had a building that was completely leveled. The xBD dataset pairs pre-disaster optical imagery (providing baseline building geometry) with post-disaster imagery. Our Siamese architecture extracts features from both epochs and analyzes the difference $|F_{\text{post}} - F_{\text{pre}}|$ to isolate structural damage.

#### Q5: How does the Siamese U-Net architecture operate, and why use shared weights?
**Answer:** In `ai_engine/model/siamese_unet.py`, one shared-weight ResNet encoder processes both pre- and post-disaster images. The hardware profile selects ResNet-18 for CPU fallback and ResNet-34 for supported GPU profiles. Shared weights project both temporal images into the same feature space before multi-scale fusion.

#### Q6: How does multi-scale feature difference fusion work?
**Answer:** At each of the 4 hierarchical feature stages ($F_1$ to $F_4$), the network computes the element-wise absolute difference $\text{diff} = |F_{\text{post}} - F_{\text{pre}}|$ and concatenates $[F_{\text{pre}}, F_{\text{post}}, \text{diff}]$. This concatenated representation is passed via skip connections into the U-Net decoder, preserving fine-grained edge details and high-level semantic damage indicators.

#### Q7: What is the loss formulation, and why combine BCE, Dice, CE, and Focal Loss?
**Answer:** In `ai_engine/model/losses.py`, we optimize a joint multi-task objective:
$$\mathcal{L}_{\text{total}} = 0.4 \times \mathcal{L}_{\text{loc}} + 0.6 \times \mathcal{L}_{\text{dmg}}$$
- **Localization Head** uses BCE + Dice Loss ($\mathcal{L}_{\text{loc}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$) to handle binary building footprint delineation.
- **Damage Classification Head** uses Class-Weighted Cross-Entropy + Focal Loss ($\gamma = 2.0$) ($\mathcal{L}_{\text{dmg}} = \mathcal{L}_{\text{CE, weighted}} + \mathcal{L}_{\text{Focal}}$) to counter the extreme class imbalance in xBD where ~70% of buildings have no damage.

#### Q8: What dataset is used, and what are the target damage classes?
**Answer:** Rubicon is trained on the benchmark **xBD (xView2)** dataset developed by DIUx and Maxar. It covers over 45,000 km² across multiple global disaster events (earthquakes, tsunamis, wildfires, hurricanes, and volcanic eruptions). The 4 standardized damage assessment classes are:
1. `No Damage` (intact building structure, green)
2. `Minor Damage` (partial roof or non-structural damage, yellow)
3. `Major Damage` (significant structural failure or wall collapse, orange)
4. `Destroyed` (complete structural leveling or foundation collapse, red)

---

### GIS & Spatial Mapping

#### Q9: How do you map pixel coordinate predictions to real-world latitude and longitude on Leaflet?
**Answer:** Raw satellite files contain spatial metadata (GeoTIFF affine matrices or GeoJSON polygon bounding boxes). We convert bounding coordinates to standard WGS84 (EPSG:4326) degrees latitude and longitude. The bounding coordinates are serialized into a GeoJSON `Polygon` and rendered as an interactive vector layer on OpenStreetMap using React-Leaflet.

#### Q10: What is Uber H3, and why is it used in this project?
**Answer:** Uber H3 is a hexagonal hierarchical spatial indexing system. Unlike square grid partitions, every neighboring hexagon in an H3 grid is equidistant, eliminating directional distortion. Rubicon indexes disaster damage zones into H3 cells (Resolutions 8–9) so emergency responders can deploy targeted drone inspections (`/drone`) to specific hexagonal sectors.

---

### Blockchain & Cryptography

#### Q11: Why is blockchain necessary in a disaster assessment application?
**Answer:** Disaster recovery involves billions of dollars in insurance payouts and municipal relief funding, creating strong incentives for post-event insurance fraud, altered assessment dates, or fabricated imagery. By recording the IPFS cryptographic hash (CID) of the raw sensor data and model prediction to Polygon Amoy, Rubicon creates an unalterable, publicly auditable, timestamped proof of evidence.

#### Q12: What exact data is stored on the Polygon blockchain?
**Answer:** We do **not** store large imagery files directly on-chain due to gas costs and block size limits. Instead, we store a lightweight `LogEntry` struct in `PramaanLedger.sol` containing:
- `hsiCid`: IPFS content identifier string for the primary pre-disaster / optical file.
- `lidarCid`: IPFS content identifier string for the post-disaster / event file.
- `prediction`: Damage classification string (`Destroyed`, `Major Damage`, etc.).
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
**Answer:** `Dockerfile.satellite` packages the xBD Siamese U-Net API, served on port 8002 and configured to load the checkpoint from `RUBICON_MODEL_DIR`. The separate main `Dockerfile` packages the legacy HSI/LiDAR `/predict` service on port 8000; its model artifacts are independent of the xBD checkpoint.

#### Q24: What is the purpose of `tests/auth-hardening.test.js` in the backend?
**Answer:** It uses `supertest` and `mongodb-memory-server` to rigorously verify session security:
- Verifies that access tokens expire within the TTL window.
- Verifies that refresh tokens are stored as SHA-256 hashes in MongoDB.
- Confirms that password changes invalidate all existing refresh tokens except the caller's active session.
- Ensures rate limiting triggers HTTP 429 when abuse thresholds are exceeded.

#### Q25: If an interviewer asks you to demonstrate the application right now, what exact steps will you take?
**Answer:**
1. Navigate to **`http://localhost:3000`** in the browser.
2. Register an account (e.g. `evaluator@rubicon.dev` / `Password123`).
3. Click **+ Upload** on the top bar to open the **xBD Satellite Assessment** modal.
4. Drag in the **Pre-Disaster Optical Scan (Reference)** and **Post-Disaster Optical Scan (Event)** images, and click **Run Assessment**.
5. Show the multi-step progress bar transitioning through Uploading, Analyzing, and On-Chain Verification.
6. Demonstrate the newly plotted damage polygon on the Leaflet map over the disaster area, clicking it to show severity (`No Damage`, `Minor Damage`, `Major Damage`, or `Destroyed`), class confidence percentage, IPFS CIDs, and blockchain transaction hash.
7. Open the AI Assistant chat panel on the right, ask *"How many assessments do we have?"* or *"Show me severe zones"*, and show how the agent executes a live tool call against the backend to return an accurate, grounded answer.
