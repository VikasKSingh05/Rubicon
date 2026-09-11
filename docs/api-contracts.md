# Rubicon — API Contracts

> **Contracts-first.** These are the exact JSON shapes that all components build against.
> No feature code is written against a different shape. If a shape must change, update this
> document **and** the mirrored artifacts in [`contracts/`](../contracts) in the same commit.
>
> These contracts are copied in during **Phase 0** ahead of implementation so all three
> teammates (AI engine, backend, frontend, agent) build against the same interfaces.

## Table of contents

1. [AI engine `POST /predict` response](#1-ai-engine-post-predict-response)
2. [Backend `POST /upload` response](#2-backend-post-upload-response)
3. [Agent `POST /agent/query` response](#3-agent-post-agentquery-response)
4. [MongoDB `Assessment` schema](#4-mongodb-assessment-schema)
5. [Assessment state machine](#5-assessment-state-machine)
6. [Severity color mapping](#6-severity-color-mapping)

---

## 1. AI engine `POST /predict` response

`ai-engine` exposes a single inference endpoint at **`POST /predict`**. The backend calls it
internally; the **frontend never talks to the AI engine directly** — it only ever receives an
assessment via the backend.

**Request body** (multipart `form-data`; Phase 6 sends the real uploaded buffers):

```json
{
  "hsi": "<file>",
  "lidar": "<file>"
}
```

**Response `200 OK`** — exact shape:

```json
{
  "prediction": "Severe Collapse",
  "confidence": 0.91,
  "class_probs": {
    "None": 0.02,
    "Moderate": 0.07,
    "Severe Collapse": 0.91
  },
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [[[lng, lat], [lng, lat], [lng, lat], [lng, lat]]]
  },
  "model_version": "mamba_transformer-v1"
}
```

> `model_version` is `mamba_transformer-v1` when the real engine responds, or `stub-v0`
> when the engine is unreachable and the backend degrades to the contract-identical stub.

| Field             | Type                 | Notes                                                |
| ----------------- | -------------------- | ---------------------------------------------------- |
| `prediction`      | string               | Human-readable damage class label.                   |
| `confidence`      | number (0–1)         | Model confidence in the predicted class.             |
| `class_probs`     | object<string,number>| Softmax probabilities across all damage classes.     |
| `geojson_polygon` | GeoJSON Polygon      | Map polygon for the assessed zone (lng, lat coords). |
| `model_version`   | string               | Real model: `mamba_transformer-v1`. When the engine is unreachable the backend degrades to `stub-v0`. |

**Contract guarantee:** swapping the Phase-1 stub for the real Phase-3 model requires **zero**
frontend changes, because the response shape never changes.

---

## 2. Backend `POST /upload` response

The backend accepts an uploaded scan (hyperspectral + LiDAR, `accept .tiff/.las`) at
**`POST /upload`** and returns the created assessment.

**Request:** `multipart/form-data` with fields `hsi` and `lidar` (files).

**Response `200 OK`** — exact shape:

```json
{
  "assessmentId": "665d8f3e2f3a4b5c6d7e8f90",
  "state": "analyzed",
  "prediction": "Severe Collapse",
  "confidence": 0.91,
  "class_probs": {
    "None": 0.02,
    "Moderate": 0.07,
    "Severe Collapse": 0.91
  },
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [[[lng, lat], [lng, lat], [lng, lat], [lng, lat]]]
  },
  "model_version": "stub-v0",
  "createdAt": "2026-09-07T12:00:00.000Z",
  "links": {
    "detail": "/assessments/665d8f3e2f3a4b5c6d7e8f90"
  }
}
```

> The analysis step is **real**: the backend forwards the uploaded files to `POST /predict`
> and stores the engine's response. `state` is `analyzed` (or `chain_logged` when the Amoy
> proof layer is configured — see §5). If the engine is unreachable the backend degrades to
> the contract-identical stub (`model_version: "stub-v0"`).

| Field      | Type                        | Notes                                        |
| ---------- | --------------------------- | -------------------------------------------- |
| `assessmentId` | string (ObjectId hex)  | Stored assessment identifier.                |
| `state`    | enum (see §5)               | `uploaded` \| `analyzing` \| `analyzed` \| … |
| `prediction` | string                   | Damage class label.                          |
| `confidence` | number (0–1)             | Model confidence.                            |
| `class_probs` | object<string,number>  | Per-class probabilities.                     |
| `geojson_polygon` | GeoJSON Polygon      | Assessment polygon.                          |
| `model_version` | string                | Model identifier.                            |
| `createdAt` | string (ISO-8601)          | Server timestamp.                            |
| `links.detail` | string                 | Relative URL to the detail endpoint.         |

---

## 2-b. Backend `GET /assessments` (list)

Returns the caller's assessments, newest first. Paginated with `?limit=` (default 50, max 200)
and `?offset=` (default 0).

**Response `200 OK`:**

```json
{
  "assessments": [
    {
      "id": "665d8f3e2f3a4b5c6d7e8f90",
      "prediction": "Severe Collapse",
      "confidence": 0.91,
      "severity": "severe",
      "state": "analyzed",
      "center": { "lng": -95.36, "lat": 29.76 },
      "geojson_polygon": { "type": "Polygon", "coordinates": [[[-95.36, 29.76], [-95.36, 29.77], [-95.35, 29.77], [-95.35, 29.76], [-95.36, 29.76]]] },
      "txHash": null,
      "chainVerified": false,
      "createdAt": "2026-09-07T12:00:00.000Z"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

| Field         | Type              | Notes                                             |
| ------------- | ----------------- | ------------------------------------------------- |
| `assessments` | array<ListItem>   | Map-marker payloads (envelope `center`).          |
| `total`       | number            | Total items for the user (for pagination).        |
| `limit`       | number            | Echo of the applied limit.                        |
| `offset`      | number            | Echo of the applied offset.                       |

`severity` uses the lowercase enum `none | moderate | severe` (see §6). `txHash`/`chainVerified`
power the map-popup chain affordance (null/empty when the proof layer was simulated).

---

## 3. Agent `POST /agent/query` response

`agent-service` exposes **`POST /agent/query`** for the chat panel. The agent is
**tool-grounded** — it answers only from tool results, never from the LLM's internal knowledge.

**Request body:**

```json
{
  "query": "Which zones are severe and unlogged?",
  "assessmentId": "665d8f3e2f3a4b5c6d7e8f90",
  "token": "<optional> user JWT forwarded to backend tool calls"
}
```

**Response `200 OK`** — exact shape:

```json
{
  "answer": "There are 3 severe zones. Two are already verified on-chain; one (zone 665d…8f90) is not yet logged.",
  "tool_calls": [
    {
      "tool": "get_recent_assessments",
      "input": { "severity_filter": "severe", "limit": 20 },
      "ok": true
    },
    {
      "tool": "get_chain_status",
      "input": { "cid": "Qm123…" },
      "ok": true
    }
  ],
  "createdAt": "2026-09-07T12:30:00.000Z"
}
```

| Field        | Type                  | Notes                                                     |
| ------------ | --------------------- | --------------------------------------------------------- |
| `answer`     | string                | Final plain-language answer.                              |
| `tool_calls` | array<ToolCall>       | Enables the "🔧 checked recent assessments" chips in UI. |
| `createdAt`  | string (ISO-8601)     | Timestamp.                                                |

**ToolCall object:**

| Field   | Type           | Notes                          |
| ------- | -------------- | ------------------------------ |
| `tool`  | string         | Tool name.                     |
| `input` | object         | Arguments passed to the tool.  |
| `ok`    | boolean        | Whether the tool call succeeded. |

**Grounding rule:** the agent must answer strictly from tool results. If a tool returns no
data, it says so (e.g. `"I don't have data for that zone."`) rather than guessing. This is a
**demo-grade, tool-grounded agent**, not a certified emergency system.

**Defined tools (Phase 4):**

- `get_recent_assessments(severity_filter, limit)` → queries backend Mongo-backed API.
- `get_assessment(id)` → full record for one assessment (confidence, probs, chain fields).
- `get_chain_status(cid)` → calls backend `/chain/verify/:cid` (on-chain event lookup when the Amoy layer is configured; DB-backed otherwise).
- `summarize_zone(geojson)` → stats summary (area, class, confidence).

The agent may run the tools through an LLM tool-calling loop (Anthropic or OpenAI, see
`LLM_PROVIDER`) or, when no key is set, through a deterministic fallback agent. Both paths
obey the same grounding rule above.

---

## 4. MongoDB `Assessment` schema

Stored by the backend in MongoDB. Shape (document):

```json
{
  "_id": "665d8f3e2f3a4b5c6d7e8f90",
  "user": "665d00000000000000000001",
  "state": "analyzed",
  "filename": {
    "hsi": "houstn_2013_region1.tiff",
    "lidar": "houstn_2013_region1.las"
  },
  "prediction": "Severe Collapse",
  "confidence": 0.91,
  "class_probs": {
    "None": 0.02,
    "Moderate": 0.07,
    "Severe Collapse": 0.91
  },
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [[[lng, lat], [lng, lat], [lng, lat], [lng, lat]]]
  },
  "model_version": "stub-v0",
  "hsiCid": null,
  "lidarCid": null,
  "txHash": null,
  "chainVerified": false,
  "timestamps": {
    "uploaded": "2026-09-07T12:00:00.000Z",
    "analyzing": "2026-09-07T12:00:01.000Z",
    "analyzed": "2026-09-07T12:00:04.000Z",
    "chainPending": null,
    "chainLogged": null
  },
  "createdAt": "2026-09-07T12:00:00.000Z",
  "updatedAt": "2026-09-07T12:00:04.000Z"
}
```

| Field          | Type                       | Notes                                                      |
| -------------- | -------------------------- | ---------------------------------------------------------- |
| `_id`          | ObjectId                   | Primary key.                                               |
| `user`         | ObjectId                   | Owning user (from JWT).                                    |
| `state`        | enum (see §5)              | Current assessment state.                                  |
| `filename`     | object `{hsi, lidar}`      | Original uploaded filenames.                               |
| `prediction`   | string                     | Damage class label.                                        |
| `confidence`   | number (0–1)               | Model confidence.                                          |
| `class_probs`  | object<string,number>      | Per-class probabilities.                                   |
| `geojson_polygon` | GeoJSON Polygon        | Assessment polygon.                                        |
| `model_version`| string                     | Model identifier.                                          |
| `storage`      | object \| null             | `{ hsi: "<relative path>", lidar: "<relative path>" }` persisted originals (Phase 7). |
| `hsiCid`       | string \| null             | IPFS CID of HSI file (Phase 5).                            |
| `lidarCid`     | string \| null             | IPFS CID of LiDAR file (Phase 5).                          |
| `txHash`       | string \| null             | Amoy transaction hash for the on-chain log (Phase 5).      |
| `chainVerified`| boolean                    | Whether verified on-chain (Phase 5).                       |
| `timestamps`   | object of ISO-8601 \| null | Per-state timestamps; mirrors the state machine.           |
| `createdAt` / `updatedAt` | ISO-8601        | Audit timestamps.                                          |

---

## 5. Assessment state machine

```
uploaded ──► analyzing ──► analyzed ──► chain_pending ──► chain_logged
                  │
                  └────────────► error
```

| State           | Phase of origin | Meaning                                          |
| --------------- | --------------- | ------------------------------------------------ |
| `uploaded`      | Phase 1         | File received, validation passed.                |
| `analyzing`     | Phase 1         | AI inference in progress (simulated in Ph 1).    |
| `analyzed`      | Phase 1         | Inference complete, result stored.               |
| `chain_pending` | Phase 5         | IPFS pinning / on-chain tx submitted.            |
| `chain_logged`  | Phase 5         | On-chain log confirmed (real txHash).            |
| `error`         | Phase 7         | Upload pipeline failed after files were accepted (persistence, inference, or proof). |

> Phase 5 runs the real proof pipeline for every upload: uploads are content-addressed
> as IPFS CIDv0 values (pinned via Pinata when `PINATA_JWT` is set) and, when the Amoy
> layer is configured (`AMOY_RPC_URL` + `DEPLOYER_PRIVATE_KEY` + `CONTRACT_ADDRESS`),
> the assessment is logged on-chain (`chain_pending → chain_logged`, `chainVerified: true`).
> Without chain config the upload falls back to a **deterministic simulated record**
> (fake `txHash`, `chainVerified: false`, state stays `analyzed`) so the app runs fully
> offline. Real CIDv0 hashes are produced locally regardless of Pinata availability.

---

## 6. Severity color mapping

From the **Global Design System** — pairs icon + label with color (never color alone).

| Severity | Hex        | Icon | Label            |
| -------- | ---------- | ---- | ---------------- |
| None/Moderate-low | `#22C55E` | —        | None / Minimal |
| Moderate | `#F59E0B`  | ⚠          | Moderate        |
| Severe   | `#DC2626`  | ⚠          | Severe         |

Verified-on-chain badge: `#0EA5E9` with a checkmark icon (visually prominent — core differentiator).

---

## 7. Auth (Phase 7 short-lived token + refresh cookie)

Registration and login return the **same session shape** (the difference is only
the HTTP status, `201` vs `200`):

```json
{
  "token": "<short-lived JWT, held in browser memory only>",
  "user": { "id": "66aa…", "email": "a@example.com" }
}
```

The refresh token is **not** in the body. It is set as an `httpOnly;
SameSite=Strict; Path=/auth` cookie named `rubicon_refresh` (7-day TTL,
rotated on every use, revocable server-side; the DB stores only its sha-256).

| Endpoint | Action |
| --- | --- |
| `POST /auth/register` `{ email, password }` | Creates the user; min password length **8**. `409` on duplicate email. |
| `POST /auth/login` `{ email, password }` | Issues a session; `401` on bad credentials. |
| `POST /auth/refresh` (cookie) | Rotates the refresh cookie and mints a fresh access token. Reused/revoked cookies → `401`. |
| `POST /auth/logout` (cookie) | Revokes the refresh session and clears the cookie. |

Protected routes take `Authorization: Bearer <accessToken>`. The frontend's API
client replays a `401` once after refreshing; `/auth/*` paths are never replayed.
Uploads additionally spool to disk and are magic-byte validated (TIFF `II*\0` /
`MM\0*`, LAS `LASF`) before inference.

Rate limits: `/auth` 10 req / 15 min / IP, `/upload` 20 req / 15 min / IP
(return `429 { "error": "too many requests" }`).

---

## Appendix — artifact map

| Shape                        | Mirror file                                         |
| ---------------------------- | --------------------------------------------------- |
| AI `/predict` response       | [`contracts/predict.response.json`](../contracts/predict.response.json) |
| Backend `/upload` response   | [`contracts/upload.response.json`](../contracts/upload.response.json)   |
| Backend `GET /assessments`   | [`contracts/assessments.list.response.json`](../contracts/assessments.list.response.json) |
| Agent `/agent/query` response| [`contracts/agent-query.response.json`](../contracts/agent-query.response.json) |
| Mongo `Assessment` schema     | [`contracts/assessment.schema.json`](../contracts/assessment.schema.json) |
