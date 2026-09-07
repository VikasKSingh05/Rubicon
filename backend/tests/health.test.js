import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";

import createApp from "../src/app.js";

const app = createApp();

test("backend exposes a health endpoint", async () => {
  const res = await request(app).get("/health");
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, { status: "ok", service: "backend" });
});