from fastapi.testclient import TestClient

from src.inference import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "ai-engine"}


def test_predict_matches_contract():
    res = client.post("/predict")
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {
        "prediction",
        "confidence",
        "class_probs",
        "geojson_polygon",
        "model_version",
    }
    assert body["prediction"] == "Severe Collapse"
    assert 0 <= body["confidence"] <= 1
    assert body["geojson_polygon"]["type"] == "Polygon"
