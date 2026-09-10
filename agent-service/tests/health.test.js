import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";

import app from "../src/index.js";

test("agent-service exposes a health endpoint", async () => {
  const res = await request(app).get("/health");
  assert.equal(res.status, 200);
  assert.equal(res.body.status, "ok");
  assert.equal(res.body.service, "agent-service");
  assert.ok(["anthropic", "openai", "openrouter", "fallback"].includes(res.body.llm));
});

test("agent-service exposes a tools manifest", async () => {
  const res = await request(app).get("/tools");
  assert.equal(res.status, 200);
  const names = res.body.tools.map((t) => t.name);
  for (const n of ["get_recent_assessments", "get_assessment", "get_chain_status", "summarize_zone"]) {
    assert.ok(names.includes(n), `missing tool ${n}`);
  }
});

test("CORS preflight is answered for the cross-origin chat panel", async () => {
  const res = await request(app)
    .options("/agent/query")
    .set("origin", "http://localhost:3000")
    .set("access-control-request-method", "POST")
    .set("access-control-request-headers", "content-type");
  assert.equal(res.status, 204);
  assert.equal(res.headers["access-control-allow-origin"], "*");
  assert.ok(
    (res.headers["access-control-allow-methods"] || "").includes("POST"),
    "expected POST in allow-methods",
  );
});

test("POST /agent/query requires a query", async () => {
  const res = await request(app).post("/agent/query").send({});
  assert.equal(res.status, 400);
  assert.equal(res.body.error, "query is required");
});