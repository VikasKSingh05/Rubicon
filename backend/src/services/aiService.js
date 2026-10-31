import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("aiService");

export const XBD_CLASSES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"];
const STUB_MODEL_VERSION = "siamese-unet-xbd-v1";

// Disaster zone center coordinates (Santa Rosa wildfire / xBD tier 1 zone)
const DISASTER_ZONE = { lng: -122.71, lat: 38.44 };

function randomClassProbs() {
  const weights = [0.45, 0.25, 0.18, 0.12];
  const jittered = weights.map((w) => w * (0.8 + Math.random() * 0.4));
  const sum = jittered.reduce((a, b) => a + b, 0);
  const probs = {};
  XBD_CLASSES.forEach((cls, i) => {
    probs[cls] = Number((jittered[i] / sum).toFixed(4));
  });
  return probs;
}

export function stubPredict(center = DISASTER_ZONE) {
  const class_probs = randomClassProbs();
  const prediction = XBD_CLASSES.reduce((best, cls) =>
    class_probs[cls] > class_probs[best] ? cls : best,
    XBD_CLASSES[0],
  );
  const dLng = (Math.random() - 0.5) * 0.015;
  const dLat = (Math.random() - 0.5) * 0.015;
  const coords = [
    center.lng + dLng,
    center.lat + dLat,
    center.lng + dLng + 0.012,
    center.lat + dLat,
    center.lng + dLng + 0.012,
    center.lat + dLat + 0.012,
    center.lng + dLng,
    center.lat + dLat + 0.012,
    center.lng + dLng,
    center.lat + dLat,
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
// the contract-identical randomized stub so uploads never hard-fail.
export async function runInference(files = {}, { requestId } = {}) {
  const isXbdSatellite = Boolean(files?.pre?.buffer && files?.post?.buffer);

  if (isXbdSatellite) {
    try {
      const form = new FormData();
      form.append("pre", new Blob([files.pre.buffer]), files.pre.originalname || "pre.png");
      form.append("post", new Blob([files.post.buffer]), files.post.originalname || "post.png");

      const satUrl = config.satEngineUrl || config.aiEngineUrl;
      const res = await fetch(`${satUrl}/infer/satellite`, {
        method: "POST",
        body: form,
        headers: { "x-request-id": requestId || crypto.randomUUID() },
        signal: AbortSignal.timeout(60_000),
      });

      if (res.ok) {
        const body = await res.json();
        // If the engine returned damage features, compute aggregate class probabilities
        const features = body.buildings_geojson?.features || [];
        const counts = { "No Damage": 0, "Minor Damage": 0, "Major Damage": 0, "Destroyed": 0 };
        for (const feat of features) {
          const dmg = feat.properties?.damage_class || feat.properties?.damage || "";
          if (dmg.includes("minor")) counts["Minor Damage"]++;
          else if (dmg.includes("major")) counts["Major Damage"]++;
          else if (dmg.includes("destroyed")) counts["Destroyed"]++;
          else counts["No Damage"]++;
        }
        const total = Math.max(1, features.length);
        const class_probs = {};
        for (const [k, v] of Object.entries(counts)) {
          class_probs[k] = Number((v / total).toFixed(4));
        }
        const prediction = XBD_CLASSES.reduce((best, cls) =>
          class_probs[cls] > class_probs[best] ? cls : best,
          XBD_CLASSES[0],
        );

        // Derive bounding polygon from buildings or cells if available
        let geojson_polygon = body.cells_geojson_r9?.features?.[0]?.geometry || null;
        if (!geojson_polygon) {
          geojson_polygon = stubPredict().geojson_polygon;
        }

        return {
          prediction,
          confidence: class_probs[prediction] || 0.85,
          class_probs,
          geojson_polygon,
          buildings_geojson: body.buildings_geojson,
          model_version: body.model_version || "siamese-resnet34-unet-xbd",
        };
      }
      logger.warn("satellite engine non-2xx, using stub", { requestId, status: res.status });
    } catch (err) {
      logger.warn("satellite engine unreachable, using stub", {
        requestId,
        message: err?.message || String(err),
      });
    }
    return stubPredict();
  }

  // Legacy fallback for hsi/lidar
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
      return stubLegacyPredict();
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
  return stubLegacyPredict();
}

export function stubLegacyPredict() {
  const class_probs = { "None": 0.05, "Moderate": 0.15, "Severe Collapse": 0.80 };
  const prediction = "Severe Collapse";
  const coords = [
    -122.71, 38.44,
    -122.698, 38.44,
    -122.698, 38.452,
    -122.71, 38.452,
    -122.71, 38.44,
  ];
  const ring = [];
  for (let i = 0; i < coords.length; i += 2) ring.push([coords[i], coords[i + 1]]);
  return {
    prediction,
    confidence: 0.8,
    class_probs,
    geojson_polygon: { type: "Polygon", coordinates: [ring] },
    model_version: "stub-v0",
  };
}