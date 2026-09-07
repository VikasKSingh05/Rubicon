"""Rubicon ai-engine FastAPI stub.

Returns the exact /predict contract shape (see docs/api-contracts.md) so the
backend can be built against it before the real model exists in Phase 3.
"""
from fastapi import FastAPI

app = FastAPI(title="Rubicon AI Engine", version="0.1.0")

PREDICT_STUB = {
    "prediction": "Severe Collapse",
    "confidence": 0.91,
    "class_probs": {"None": 0.02, "Moderate": 0.07, "Severe Collapse": 0.91},
    "geojson_polygon": {
        "type": "Polygon",
        "coordinates": [
            [[-95.4, 29.8], [-95.3, 29.8], [-95.3, 29.9], [-95.4, 29.9], [-95.4, 29.8]]
        ],
    },
    "model_version": "stub-v0",
}


@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-engine"}


@app.post("/predict")
def predict():
    """Stub inference. In Phase 3 this accepts .tiff/.las multipart and runs the fusion model."""
    return PREDICT_STUB
