import { config } from "../config.js";

const CLASSES = ["None", "Moderate", "Severe Collapse"];
const STUB_MODEL_VERSION = "stub-v0";

// Basae around Houston (UTM zone 15 / ~29.76N, 95.36W).
const HOUSTON = { lng: -95.36, lat: 29.76 };

function randomClassProbs() {
  const weights = CLASSES.map(() => Math.random());
  const sum = weights.reduce((a, b) => a + b, 0);
  const probs = {};
  CLASSES.forEach((cls, i) => {
    probs[cls] = Number((weights[i] / sum).toFixed(4));
  });
  return probs;
}

function stubPredict() {
  const class_probs = randomClassProbs();
  const prediction = CLASSES.reduce((best, cls) =>
    class_probs[cls] > class_probs[best] ? cls : best,
    CLASSES[0],
  );
  const dLng = (Math.random() - 0.5) * 0.02;
  const dLat = (Math.random() - 0.5) * 0.02;
  const coords = [
    HOUSTON.lng + dLng,
    HOUSTON.lat + dLat,
    HOUSTON.lng + dLng + 0.015,
    HOUSTON.lat + dLat,
    HOUSTON.lng + dLng + 0.015,
    HOUSTON.lat + dLat + 0.015,
    HOUSTON.lng + dLng,
    HOUSTON.lat + dLat + 0.015,
    HOUSTON.lng + dLng,
    HOUSTON.lat + dLat,
  ];
  const ring = [];
  for (let i = 0; i < coords.length; i += 2) ring.push([coords[i], coords[i + 1]]);

  return {
    prediction,
    confidence: class_probs[prediction],
    class_probs,
    geojson_polygon: { type: "Polygon", coordinates: [ring] },
    model_version: STUB_MODEL_VERSION,
  };
}

/**
 * Run inference. Phase 1: returns a randomized stub that matches the /predict contract
 * exactly. If the real AI engine is reachable, it is used instead — this is the only
 * seam that changes when the Phase 3 model lands.
 */
export async function runInference() {
  try {
    const res = await fetch(`${config.aiEngineUrl}/predict`, {
      method: "POST",
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const body = await res.json();
      if (body && body.prediction && body.geojson_polygon) return body;
    }
  } catch {
    // fall through to the stub
  }
  return stubPredict();
}