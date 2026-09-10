import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";

import createApp from "../src/app.js";

const app = createApp();

test("health reports degraded when Mongo is not connected", async () => {
  const res = await request(app).get("/health");
  // This test process never connects to Mongo, so the dependency check is 503.
  assert.equal(res.status, 503);
  assert.equal(res.body.service, "backend");
  assert.equal(typeof res.body.mongo, "string");
});