import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("aiService");

const CLASSES = ["None", "Moderate", "Severe Collapse"];
const STUB_MODEL_VERSION = "stub-v0";

// Based around Houston (UTM zone 15 / ~29.76N, 95.36W).
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

// Graceful-degrade seam: when the real engine is reachable it gets the actual
// uploaded buffers as multipart; on any transport/parse failure we fall back to
// the contract-identical randomized stub so uploads never hard-fail. The stub is
// distinguishable via model_version "stub-v0".
export async function runInference(files = {}, { requestId } = {}) {
  try {
    const form = new FormData();
    if (files?.hsi?.buffer) {
      form.append("hsi", new Blob([files.hsi.buffer]), files.hsi.originalname || "scan.tif");
    }
    if (files?.lidar?.buffer) {
      form.append("lidar", new Blob([files.lidar.buffer]), files.lidar.originalname || "scan.las");
    }

    const res = await fetch(`${config.aiEngineUrl}/predict`, {
      method: "POST",
      body: form,
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) {
      logger.warn("engine non-2xx, using stub", { requestId, status: res.status });
      return stubPredict();
    }
    const body = await res.json();
    if (body && body.prediction && body.geojson_polygon) return body;
    logger.warn("unexpected /predict payload, using stub", { requestId });
  } catch (err) {
    logger.warn("engine unreachable, using stub", {
      requestId,
      message: err?.message || String(err),
    });
  }
  return stubPredict();
}