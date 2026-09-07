import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";

import app from "../src/index.js";

test("frontend exposes a health endpoint", async () => {
  const res = await request(app).get("/health");
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, { status: "ok", service: "frontend" });
});