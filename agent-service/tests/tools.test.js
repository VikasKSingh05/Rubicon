import test from "node:test";
import assert from "node:assert/strict";
import { execTool, roughAreaKm2, manifest } from "../src/tools.js";

const baseOpts = {
  backendUrl: "http://backend.test",
  token: "jwt-token",
  fetchImpl: async (url) => {
    if (url.endsWith("/assessments")) {
      return {
        ok: true,
        json: async () => ({
          assessments: [
            { id: "a1", prediction: "Severe Collapse", severity: "severe", confidence: 0.9, createdAt: "2026-09-08T00:00:00Z" },
            { id: "a2", prediction: "Moderate", severity: "moderate", confidence: 0.5, createdAt: "2026-09-07T00:00:00Z" },
            { id: "a3", prediction: "None", severity: "none", confidence: 0.4, createdAt: "2026-09-06T00:00:00Z" },
          ],
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  },
};

test("manifest lists the four Phase-4 tools", () => {
  const names = manifest().map((t) => t.name);
  assert.deepEqual(names, [
    "get_recent_assessments",
    "get_assessment",
    "get_chain_status",
    "summarize_zone",
  ]);
});

test("get_recent_assessments filters by severity and honors the auth header", async () => {
  let seenUrl = "";
  let seenAuth = "";
  const opts = {
    ...baseOpts,
    fetchImpl: async (url, init) => {
      seenUrl = url;
      seenAuth = init.headers.Authorization;
      return baseOpts.fetchImpl(url);
    },
  };
  const r = await execTool("get_recent_assessments", { severity_filter: "severe", limit: 1 }, opts);
  assert.equal(r.ok, true);
  assert.match(r.summary, /"Severe Collapse"/);
  assert.doesNotMatch(r.summary, /Moderate/);
  assert.equal(seenUrl, "http://backend.test/assessments");
  assert.equal(seenAuth, "Bearer jwt-token");
});

test("get_recent_assessments reports an empty result without inventing data", async () => {
  const opts = {
    ...baseOpts,
    fetchImpl: async () => ({ ok: true, json: async () => ({ assessments: [] }) }),
  };
  const r = await execTool("get_recent_assessments", { severity_filter: "severe" }, opts);
  assert.equal(r.ok, true);
  assert.match(r.summary, /No severe assessments found/);
});

test("backend failure surfaces ok:false so the agent can refuse", async () => {
  const opts = {
    ...baseOpts,
    fetchImpl: async () => ({ ok: false, status: 500, json: async () => ({ error: "boom" }) }),
  };
  const r = await execTool("get_recent_assessments", {}, opts);
  assert.equal(r.ok, false);
  assert.match(r.summary, /failed \(500\)/);
});

test("summarize_zone computes rough area and bbox for a small polygon", () => {
  const { area_km2, bbox } = roughAreaKm2({
    type: "Polygon",
    coordinates: [[[-95.34, 29.72], [-95.33, 29.72], [-95.33, 29.73], [-95.34, 29.72]]],
  });
  assert.ok(area_km2 > 1.0 && area_km2 < 1.2, `unexpected area ${area_km2}`);
  assert.deepEqual(bbox, [-95.34, 29.72, -95.33, 29.73]);
});

test("unknown tool returns a bounded error", async () => {
  const r = await execTool("trigger_alert", {}, baseOpts);
  assert.equal(r.ok, false);
  assert.match(r.summary, /unknown tool/);
});