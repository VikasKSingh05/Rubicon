import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import request from "supertest";

const ASSESSMENTS = [
  {
    id: "66aa1f0b0000000000000001",
    prediction: "Severe Collapse",
    severity: "severe",
    confidence: 0.91,
    state: "analyzed",
    center: { lng: -95.34, lat: 29.72 },
    createdAt: "2026-09-08T12:00:00.000Z",
  },
  {
    id: "66aa1f0b0000000000000002",
    prediction: "Moderate",
    severity: "moderate",
    confidence: 0.62,
    state: "analyzed",
    center: { lng: -95.38, lat: 29.76 },
    createdAt: "2026-09-07T09:00:00.000Z",
  },
];

const DETAIL = {
  id: ASSESSMENTS[0].id,
  prediction: "Severe Collapse",
  severity: "severe",
  confidence: 0.91,
  class_probs: { None: 0.02, Moderate: 0.07, "Severe Collapse": 0.91 },
  geojson_polygon: { type: "Polygon", coordinates: [[[-95.34, 29.72], [-95.33, 29.72], [-95.33, 29.73], [-95.34, 29.72]]] },
  state: "analyzed",
  hsiCid: "QmKnownHsiCid123456",
  lidarCid: "QmKnownLidar",
  txHash: null,
  chainVerified: false,
  createdAt: "2026-09-08T12:00:00.000Z",
};

let server;
let baseUrl;
let app;

test.before(async () => {
  server = http.createServer((req, res) => {
    const respond = (data) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    };
    if (req.url === "/assessments") return respond({ assessments: ASSESSMENTS });
    if (req.url.startsWith("/assessments/")) {
      const id = req.url.slice("/assessments/".length);
      return id === DETAIL.id ? respond(DETAIL) : respond({ error: "assessment not found" });
    }
    if (req.url.startsWith("/chain/verify/")) {
      const cid = decodeURIComponent(req.url.slice("/chain/verify/".length));
      if (cid === DETAIL.hsiCid) {
        return respond({
          cid,
          verified: false,
          txHash: null,
          assessmentId: DETAIL.id,
          state: DETAIL.state,
        });
      }
      return respond({ cid, verified: false, txHash: null, assessmentId: null, state: null });
    }
    return respond({ error: "not found" });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;

  process.env.BACKEND_URL = baseUrl;
  process.env.LLM_PROVIDER = "";
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.OPENAI_API_KEY;

  const mod = await import("../src/index.js");
  app = mod.default;
});

test.after(() => new Promise((resolve) => server.close(resolve)));

function assertContract(body) {
  assert.equal(typeof body.answer, "string");
  assert.ok(Array.isArray(body.tool_calls));
  assert.ok(body.createdAt);
  for (const tc of body.tool_calls) {
    assert.equal(typeof tc.tool, "string");
    assert.equal(typeof tc.input, "object");
    assert.equal(typeof tc.ok, "boolean");
  }
}

test("query about severe zones is answered from list tool data", async () => {
  const res = await request(app).post("/agent/query").send({ query: "show severe zones" });
  assert.equal(res.status, 200);
  assertContract(res.body);
  assert.ok(res.body.tool_calls.some((t) => t.tool === "get_recent_assessments" && t.ok));
  assert.match(res.body.answer, /severe/i);
});

test("counts query returns a severity breakdown", async () => {
  const res = await request(app).post("/agent/query").send({ query: "how many assessments do we have?" });
  assert.equal(res.status, 200);
  assertContract(res.body);
  assert.match(res.body.answer, /2 assessment/);
  assert.match(res.body.answer, /Severe: 1, Moderate: 1, None\/Minimal: 0/);
});

test("chain verification query is answered from get_chain_status", async () => {
  const res = await request(app).post("/agent/query").send({ query: "is the last upload verified on-chain?" });
  assert.equal(res.status, 200);
  assertContract(res.body);
  const chain = res.body.tool_calls.find((t) => t.tool === "get_chain_status");
  assert.ok(chain, "expected get_chain_status tool call");
  assert.equal(chain.ok, true);
  assert.match(res.body.answer, /not yet verified on-chain/i);
});

test("assessmentId is honored for single-assessment questions", async () => {
  const res = await request(app)
    .post("/agent/query")
    .send({ query: "tell me about this assessment", assessmentId: DETAIL.id });
  assert.equal(res.status, 200);
  assertContract(res.body);
  assert.match(res.body.answer, new RegExp(DETAIL.id));
});

test("agent refuses to fabricate when the backend is unreachable", async () => {
  const { config } = await import("../src/config.js");
  const previous = config.backendUrl;
  config.backendUrl = "http://127.0.0.1:1"; // dead port
  try {
    const res = await request(app).post("/agent/query").send({ query: "show severe zones" });
    assert.equal(res.status, 200);
    assertContract(res.body);
    const list = res.body.tool_calls.find((t) => t.tool === "get_recent_assessments");
    assert.equal(list.ok, false);
    assert.match(res.body.answer, /couldn't retrieve/i);
    assert.doesNotMatch(res.body.answer, /\bThere (is|are)\b \d+\s+severe/i);
  } finally {
    config.backendUrl = previous;
  }
});

test("provider failure degrades to the fallback agent", async () => {
  const { config } = await import("../src/config.js");
  const previousProvider = config.llm.provider;
  const originalFetch = globalThis.fetch;
  const realFetch = originalFetch.bind(globalThis);
  config.llm.provider = "openai";
  process.env.OPENAI_API_KEY = "sk-flaky";
  globalThis.fetch = async (url, init) => {
    if (String(url).startsWith("https://api.openai.com")) throw new Error("provider down");
    return realFetch(url, init);
  };
  try {
    const res = await request(app).post("/agent/query").send({ query: "show severe zones" });
    assert.equal(res.status, 200);
    assertContract(res.body);
    assert.ok(res.body.tool_calls.some((t) => t.tool === "get_recent_assessments" && t.ok));
    assert.match(res.body.answer, /severe/i);
  } finally {
    globalThis.fetch = originalFetch;
    delete process.env.OPENAI_API_KEY;
    config.llm.provider = previousProvider;
  }
});