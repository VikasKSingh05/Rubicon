import test from "node:test";
import assert from "node:assert/strict";
import { config } from "../src/config.js";
import { runInference, XBD_CLASSES } from "../src/services/aiService.js";

const ENGINE_BODY = {
  buildings_geojson: {
    type: "FeatureCollection",
    features: [
      { properties: { damage_class: "minor-damage" } },
      { properties: { damage_class: "major-damage" } },
      { properties: { damage_class: "destroyed" } },
    ],
  },
  cells_geojson_r9: {
    type: "FeatureCollection",
    features: [{ geometry: { type: "Polygon", coordinates: [[[]]] } }],
  },
  model_version: "siamese-resnet34-unet-xbd",
};

const imagePair = {
  pre: { buffer: Buffer.from("pre-image"), originalname: "before.png" },
  post: { buffer: Buffer.from("post-image"), originalname: "after.png" },
};

test("runInference forwards the xBD pair to the Siamese U-Net endpoint", async () => {
  config.satEngineUrl = "http://engine.test";
  const originalFetch = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, opts) => {
    captured = { url, ...opts };
    return { ok: true, json: async () => ENGINE_BODY };
  };
  try {
    const result = await runInference(imagePair, { requestId: "request-1" });
    assert.equal(result.model_version, ENGINE_BODY.model_version);
    assert.deepEqual(Object.keys(result.class_probs), XBD_CLASSES);
    assert.equal(result.class_probs["Minor Damage"], 0.3333);
    assert.equal(result.class_probs["Major Damage"], 0.3333);
    assert.equal(result.class_probs.Destroyed, 0.3333);
    assert.equal(result.prediction, "Minor Damage");
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(captured.url, "http://engine.test/infer/satellite");
  assert.equal(captured.headers["x-request-id"], "request-1");
  assert.ok(captured.body instanceof FormData);
  assert.equal(captured.body.get("pre").name, "before.png");
  assert.equal(captured.body.get("post").name, "after.png");
});

test("runInference returns an xBD-labelled demo result when the engine is unreachable", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("ECONNREFUSED");
  };
  try {
    const result = await runInference(imagePair);
    assert.equal(result.model_version, "stub-satellite-v0");
    assert.deepEqual(Object.keys(result.class_probs), XBD_CLASSES);
    assert.ok(result.prediction && result.geojson_polygon);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("runInference returns the xBD demo result for non-2xx engine responses", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 503 });
  try {
    const result = await runInference(imagePair);
    assert.equal(result.model_version, "stub-satellite-v0");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
