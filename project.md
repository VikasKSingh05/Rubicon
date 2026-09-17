# Rubicon — Project Documentation & Codebase Architecture

**Autonomous Multimodal Disaster Assessment with Immutable Proof System**

---

## 1. Executive Summary

**Rubicon** is an end-to-end, enterprise-grade AI and Web3 platform designed for autonomous disaster damage assessment. It ingests multimodal remote-sensing aerial imagery — specifically **Hyperspectral Imaging (HSI)** and **LiDAR elevation rasters/point clouds** — processes them through a state-of-the-art **Mamba–Transformer deep learning fusion architecture**, generates spatial damage severity assessments, and anchors the assessment metadata and raw evidence onto **IPFS** (content-addressed storage via Pinata) and an immutable blockchain ledger (**Polygon Amoy testnet** smart contract).

The platform features:
1. **AI Engine (`ai-engine`)**: Python/FastAPI microservice implementing custom PyTorch dataset loading, spatial co-registration, PCA dimensionality reduction, Mamba–Transformer late-fusion deep neural networks, and inference pipelines.
2. **Backend Service (`backend`)**: Express/Node.js REST API enforcing JWT authentication, rate limiting, magic-byte upload validation, IPFS CIDv0 cryptographic calculation, Pinata pinning, MongoDB persistence, and smart contract interaction via Ethers.js.
3. **Frontend Dashboard (`frontend`)**: React/Vite single-page web app with interactive Leaflet GIS maps, raster visualizations, real-time assessment status tracking, evidence verification cards, and AI chat assistant interface.
4. **Agentic Assistant Service (`agent-service`)**: Express/Node.js conversational service powered by tool-grounded AI (LLM integration with fallback deterministic tool execution) to query live backend state.
5. **Smart Contracts (`contracts`)**: Solidity contract (`PramaanLedger.sol`) deployed on Polygon Amoy testnet to log immutable proof of disaster assessments on-chain.

---

## 2. Project Monorepo Structure

```
Rubicon/
├── ai-engine/                  # Python FastAPI AI service
│   ├── configs/                # YAML configurations (phase2, phase3, preprocess_eda)
│   ├── data/                   # Data directories (raw, processed, models)
│   ├── notebooks/              # Jupyter notebooks for EDA & visualization
│   │   ├── eda.ipynb
│   │   ├── data_visualization.ipynb
│   │   └── data_visulaization.ipynb
│   ├── scripts/                # Download & preprocessing runner scripts
│   ├── src/                    # Source code
│   │   ├── inference.py        # FastAPI service endpoints
│   │   ├── model/              # FusionModel, MambaBlock, TransformerBlock, BaselineCNN
│   │   ├── preprocessing/      # Raster loading, alignment, PCA, patchification, pipeline
│   │   └── training/           # PyTorch training & evaluation loops
│   ├── tests/                  # Pytest test suite
│   ├── Dockerfile
│   └── requirements.txt
├── agent-service/              # Node.js Agentic AI Assistant
│   ├── src/                    # Express app, agent routes, tool execution logic
│   ├── tests/                  # Jest test suite
│   ├── Dockerfile
│   └── package.json
├── backend/                    # Express API Backend
│   ├── src/
│   │   ├── abis/               # Smart contract ABI artifacts
│   │   ├── middleware/         # Auth JWT, rate limiter, upload validator
│   │   ├── models/             # Mongoose schemas (User, Assessment, ChainIndex)
│   │   ├── routes/             # Auth, Assessment, File download, Chain verify endpoints
│   │   ├── services/           # IPFS service, Blockchain Ethers service, AI engine client
│   │   ├── app.js
│   │   ├── config.js
│   │   └── db.js
│   ├── tests/                  # Integration tests (Supertest + In-memory MongoDB)
│   ├── Dockerfile
│   └── package.json
├── contracts/                  # Blockchain Smart Contracts & Schemas
│   ├── contracts/
│   │   └── PramaanLedger.sol   # Solidity ledger smart contract
│   ├── scripts/                # Deployment scripts
│   ├── *.json                  # JSON API-contracts & schemas
│   └── package.json
├── frontend/                   # React + Vite Map Dashboard
│   ├── src/
│   │   ├── components/         # Map, Upload, AssessmentList, DetailView, ChatAgent, Auth
│   │   ├── context/            # AuthContext (short-lived tokens + httpOnly refresh)
│   │   ├── lib/                # API client & helpers
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── docs/                       # Comprehensive documentation & reports
│   ├── api-contracts.md
│   ├── deploy.md
│   ├── operations.md
│   ├── phase0-setup.md
│   └── report.md
├── docker-compose.yml          # Container orchestration for all microservices
└── .env.example                # Unified environment template
```

---

## 3. Microservice Architecture & Data Flow

```
[User Browser]
      │
      ├── (HTTP/REST) ──► Frontend (React/Vite :3000)
      │                       │
      │        ┌──────────────┴──────────────┐
      │        ▼                             ▼
      │   Backend (Express :4000)      Agent-Service (Node :8001)
      │        │                             │
      │        ├── Mongo Atlas DB            └── LLM API / Tool Fallback
      │        │                             
      │        ├──► ai-engine (FastAPI :8000) ──► PyTorch Mamba-Transformer Model
      │        │                             
      │        ├──► IPFS / Pinata ────────────► CIDv0 Cryptographic Hashing
      │        │                             
      │        └──► Polygon Amoy Testnet ─────► PramaanLedger Smart Contract
```

### Complete End-to-End Request Pipeline
1. **Authentication**: User logs in or registers via `POST /api/auth/login`. Short-lived access JWT (15-min TTL) is stored in browser memory, and a rotating 7-day `httpOnly` refresh cookie is set.
2. **File Upload & Validation**: User uploads `.tif/.tiff` (Hyperspectral) and optional `.tif/.las/.laz` (LiDAR) files through the Frontend map interface. Backend rate limiter checks request counts, and magic-byte header validation verifies file integrity before saving to disk.
3. **AI Inference (`ai-engine`)**: Backend forwards files to FastAPI `POST /predict`.
   - `ai-engine` loads raw GeoTIFF/LAS files.
   - Spatial rasters are co-registered and resampled to an identical spatial grid.
   - PCA reduces 144 spectral bands to $k$ principal components retaining 99% variance.
   - Modalities are converted into spatial patch tensors $X_{\text{hsi}} \in \mathbb{R}^{B \times k \times 32 \times 32}$ and $X_{\text{dem}} \in \mathbb{R}^{B \times 1 \times 32 \times 32}$.
   - PyTorch `FusionModel` executes dual-branch processing (Mamba + Transformer for HSI, CNN for LiDAR) and outputs 3-class damage severity probabilities (**None**, **Moderate**, **Severe Collapse**).
4. **Cryptographic Proof Generation (IPFS)**:
   - Backend reads raw file bytes, computes sha2-256 multihash, and converts it to a deterministic **IPFS CIDv0** string (`Qm...`).
   - If `PINATA_JWT` is configured, file buffers are pinned to global IPFS storage via Pinata.
5. **On-Chain Audit Logging (Polygon Amoy)**:
   - Backend signs an Ethereum transaction invoking `logAssessment(bytes32 assessmentId, string cid, uint8 severity)` on `PramaanLedger.sol`.
   - On-chain transaction hash and block number are logged to the MongoDB assessment record.
6. **Visualization & Agentic Querying**:
   - Assessment bounds and damage polygon are rendered on the Leaflet map in the frontend.
   - Users can chat with the `agent-service`, which queries live assessment records via backend REST APIs to answer natural language questions.

---

## 4. Key Module Explanations

### A. AI Engine (`ai-engine`)
- **`src/preprocessing/preprocess_eda_pipeline.py`**: Pipeline script that handles dataset ingestion (Houston 2013), synthetic dataset generation fallback, spatial alignment, spectral PCA, patch extraction, and export of `.npz` tensors.
- **`src/preprocessing/dataset.py`**: Leak-free training pipeline. Fits PCA statistics and DEM min-max scaling strictly on the training split before transforming validation and test splits.
- **`src/model/mamba.py`**: Pure PyTorch hardware-agnostic selective state-space model layer (`MambaBlock`), implementing continuous-to-discrete parameter discretization ($\Delta, A, B, C, D$) without requiring NVIDIA CUDA C++ extensions.
- **`src/model/fusion.py`**: Late-fusion architecture combining `HsiBranch` (Convolutional stem + Mamba blocks + Vision Transformer blocks) and `DemBranch` (CNN trunk) with a LayerNorm MLP classification head.
- **`src/model/baseline.py`**: `BaselineCNN` residual 2D convolutional network for comparative benchmark and resource-constrained fallback.
- **`src/inference.py`**: FastAPI server exposing `/health` and `/predict` endpoints.

### B. Backend API (`backend`)
- **`src/routes/auth.js`**: User registration, login, token refresh with rotation, password update, and logout.
- **`src/routes/assessments.js`**: Multipart file upload handling, triggering AI inference, generating IPFS CID, writing to MongoDB, and dispatching blockchain transactions.
- **`src/services/chainService.js`**: Interacts with Polygon Amoy smart contracts via Ethers.js v6. Includes event lookup verification (`verifyOnChain`).
- **`src/services/ipfs.js`**: Computes CIDv0 offline using multiformats/digest and pins files to Pinata.

### C. Smart Contracts (`contracts`)
- **`PramaanLedger.sol`**: Solidity contract featuring:
  - `logAssessment(bytes32 assessmentId, string cid, uint8 severity)`: Emits `AssessmentLogged` event.
  - `getAssessment(bytes32 assessmentId)`: Returns logged CID, severity, timestamp, and recorder address.

### D. Agent Service (`agent-service`)
- **`src/agent.js`**: Conversational agent. If an LLM API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) is present, it uses function-calling tools to query backend stats. If no key is present, it uses a deterministic tool-grounded rule engine.

### E. Frontend (`frontend`)
- **`src/components/Map.jsx`**: Leaflet map displaying disaster assessment polygons colored by damage severity.
- **`src/components/ChatAgent.jsx`**: Sidebar drawer to interact with the AI assistant.
- **`src/components/UploadModal.jsx`**: File drag-and-drop dialog supporting `.tif`, `.tiff`, `.las`, `.laz`.

---

## 5. Security, Degradation & Production Design

- **Short-Lived Auth**: Access JWTs are kept only in React state memory (not `localStorage`). Refresh tokens use secure `httpOnly` cookies with automatic rotation.
- **Rate Limiting & Validation**: Express `express-rate-limit` prevents brute force; file uploads check magic bytes (`II*\x00` / `MM\x00*` for TIFF, `LASF` for LAS/LAZ).
- **Graceful Degradation**:
  1. Engine down $\rightarrow$ Backend returns `stub-v0` prediction without crashing.
  2. Pinata key missing $\rightarrow$ Offline local CIDv0 calculation is performed (`pinned: false`).
  3. Blockchain key missing $\rightarrow$ System logs deterministic fallback txHash (`chainVerified: false`).
  4. LLM key missing $\rightarrow$ Agent falls back to deterministic rule-based assistant.

---

## 6. Verification and Testing

All services include comprehensive automated test suites:
- **Backend**: `cd backend && npm test` (Uses in-memory MongoDB + Supertest).
- **Agent Service**: `cd agent-service && npm test`.
- **Frontend**: `cd frontend && npm test && npm run build`.
- **AI Engine**: `cd ai-engine && python -m pytest`.
- **Contracts**: `cd contracts && npm run compile`.
