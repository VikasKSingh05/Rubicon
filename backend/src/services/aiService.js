import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("aiService");

export const XBD_CLASSES = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"];
const STUB_MODEL_VERSION = "stub-satellite-v0";

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

// Forward xBD pre/post imagery to the Siamese U-Net engine. If it is
// unavailable, return the contract-compatible demo result and log the cause.
export async function runInference(files = {}, { requestId } = {}) {
  try {
    if (!files?.pre?.buffer || !files?.post?.buffer) {
      throw new Error("xBD inference requires both pre- and post-disaster images");
    }

    const form = new FormData();
    form.append("pre", new Blob([files.pre.buffer]), files.pre.originalname || "pre.png");
    form.append("post", new Blob([files.post.buffer]), files.post.originalname || "post.png");

    const res = await fetch(`${config.satEngineUrl}/infer/satellite`, {
      method: "POST",
      body: form,
      headers: { "x-request-id": requestId || crypto.randomUUID() },
      signal: AbortSignal.timeout(30_000),
    });

    if (!res.ok) throw new Error(`satellite engine returned HTTP ${res.status}`);

    const body = await res.json();
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

    const geojson_polygon =
      body.cells_geojson_r9?.features?.[0]?.geometry || stubPredict().geojson_polygon;

    return {
      prediction,
      confidence: class_probs[prediction] || 0.85,
      class_probs,
      geojson_polygon,
      buildings_geojson: body.buildings_geojson,
      model_version: body.model_version || "siamese-resnet34-unet-xbd",
    };
  } catch (err) {
    logger.warn("satellite engine unavailable, using demo fallback", {
      requestId,
      message: err?.message || String(err),
    });
  }
  return stubPredict();
}