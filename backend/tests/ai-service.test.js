import test from "node:test";
import assert from "node:assert/strict";
import { config } from "../src/config.js";
import { runInference } from "../src/services/aiService.js";

const ENGINE_BODY = {
  prediction: "Moderate",
  confidence: 0.8,
  class_probs: { None: 0.2, Moderate: 0.8, "Severe Collapse": 0 },
  geojson_polygon: { type: "Polygon", coordinates: [[[]]] },
  model_version: "mamba_transformer-v1",
};

test("runInference forwards real buffers as multipart and returns the engine body", async () => {
  config.aiEngineUrl = "http://engine.test";
  const originalFetch = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, opts) => {
    captured = { url, body: opts.body };
    return { ok: true, json: async () => ENGINE_BODY };
  };
  try {
    const result = await runInference({
      hsi: { buffer: Buffer.from("fake-hsi-bytes"), originalname: "scan.tiff" },
      lidar: { buffer: Buffer.from("fake-lidar-bytes"), originalname: "scan.las" },
    });
    assert.deepEqual(result, ENGINE_BODY);
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(captured.url, "http://engine.test/predict");
  assert.ok(captured.body instanceof FormData);
  const hsi = captured.body.get("hsi");
  const lidar = captured.body.get("lidar");
  assert.equal(hsi.name, "scan.tiff");
  assert.equal(lidar.name, "scan.las");
  assert.deepEqual(
    new Uint8Array(await hsi.arrayBuffer()),
    new Uint8Array(Buffer.from("fake-hsi-bytes")),
  );
});

test("runInference degrades to the stub when the engine is unreachable", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("ECONNREFUSED");
  };
  try {
    const result = await runInference({
      hsi: { buffer: Buffer.from("x"), originalname: "a.tif" },
    });
    assert.equal(result.model_version, "stub-v0");
    assert.ok(result.prediction && result.geojson_polygon);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("runInference degrades to the stub on a non-2xx engine response", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: false, status: 500 });
  try {
    const result = await runInference({});
    assert.equal(result.model_version, "stub-v0");
  } finally {
    globalThis.fetch = originalFetch;
  }
});