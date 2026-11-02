# Rubicon API Contracts

These are the current service contracts. Satellite assessment uses xBD
pre/post optical imagery and the Siamese U-Net. The separate Stage 2 endpoint
accepts drone imagery. New uploads do not accept the retired scan format.

## 1. AI engine

### `POST /infer/satellite`

Multipart fields `pre` and `post` are required. Each accepts PNG, JPEG, or
GeoTIFF. The response is defined in
[`contracts/satellite.response.json`](../contracts/satellite.response.json)
and contains the source image SHA-256 values, building and H3-cell
FeatureCollections, `damage_classes`, execution time, and `model_version`.

The no-checkpoint result is labeled `stub-satellite-v0`. Check
`GET /health` and require `model_loaded: true` before reporting model output.

### `POST /infer/drone`

Multipart field `image` is required. Optional fields are `parent_cell_id`,
`lat`, and `lon`. It returns damage severity, failure-mode analysis, image
hash, and model version as defined in
[`contracts/drone.response.json`](../contracts/drone.response.json).

## 2. Backend assessment upload

### `POST /upload`

Requires authentication and two image files:

| Field | Accepts |
| --- | --- |
| `pre` | PNG, JPEG, or TIFF |
| `post` | PNG, JPEG, or TIFF |

The backend validates the file extension and magic bytes, persists the
originals, runs satellite inference, and attaches proof metadata. Example
response:

```json
{
  "assessmentId": "665d8f3e2f3a4b5c6d7e8f90",
  "state": "analyzed",
  "prediction": "No Damage",
  "confidence": 0.8,
  "class_probs": {
    "No Damage": 0.8,
    "Minor Damage": 0.1,
    "Major Damage": 0.05,
    "Destroyed": 0.05
  },
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [[[ -122.71, 38.44 ], [ -122.70, 38.44 ], [ -122.70, 38.45 ], [ -122.71, 38.45 ], [ -122.71, 38.44 ]]]
  },
  "model_version": "stub-satellite-v0",
  "createdAt": "2026-11-02T00:00:00.000Z",
  "links": { "detail": "/assessments/665d8f3e2f3a4b5c6d7e8f90" }
}
```

The exact response shape is mirrored in
[`contracts/upload.response.json`](../contracts/upload.response.json). A
successful request may still be a stub; inspect `model_version`.

## 3. Assessments

`GET /assessments?limit=&offset=` returns the authenticated user's newest
assessments with pagination. List items contain the prediction, confidence,
severity, map polygon and center, chain status, and creation time. The example
is in [`contracts/assessments.list.response.json`](../contracts/assessments.list.response.json).

`GET /assessments/:id` returns the complete assessment, including `filename`,
`storage`, `class_probs`, `preCid`, `postCid`, chain status, and timestamps.
Existing records may also contain historical CID aliases; the backend retains
read compatibility but new uploads populate the pre/post fields.

`GET /assessments/:id/cells` returns a GeoJSON FeatureCollection for map
visualization. `GET /assessments/:id/verify` checks the first image CID against
the configured ledger and returns database state when chain verification is
unavailable.

## 4. File downloads

`GET /files/:id/pre` and `GET /files/:id/post` stream the original paired
images. Both routes require authentication and enforce assessment ownership.
Historical file-kind aliases remain readable for existing records.

## 5. Assessment states and severity

```text
uploaded -> analyzing -> analyzed -> chain_pending -> chain_logged
                 \-> error
```

Without Polygon configuration, the proof is simulated and the state remains
`analyzed`. `chainVerified` is false for simulated records.

The xBD damage classes are `No Damage`, `Minor Damage`, `Major Damage`, and
`Destroyed`. The dashboard maps these classes to the existing `none`,
`moderate`, and `severe` display severities.

## 6. Compatibility boundary

The deployed PramaanLedger contract has a positional pair of CID arguments.
New satellite image CIDs are supplied in those positions without changing the
existing contract interface. Historical MongoDB fields and file-download
aliases are retained only to keep old assessment records readable.

Authentication, agent, and health response contracts remain documented by
their service routes and the JSON artifacts in [`contracts/`](../contracts/).
