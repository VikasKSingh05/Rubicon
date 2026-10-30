import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import mongoose from "mongoose";
import { MongoMemoryServer } from "mongodb-memory-server";
import createApp from "../src/app.js";
import User from "../src/models/User.js";
import Assessment from "../src/models/Assessment.js";

let mongod;
let app;
let token;
let userId;

test.before(async () => {
  mongod = await MongoMemoryServer.create();
  await mongoose.connect(mongod.getUri());
  app = createApp();

  const user = await User.create({
    email: "triage-officer@rubicon.test",
    password: "dummyPassword123",
  });
  userId = user._id.toString();

  const regRes = await request(app)
    .post("/auth/register")
    .send({ email: "satellite-drone@rubicon.dev", password: "Password123!" });
  token = regRes.body.token;
});

test.after(async () => {
  await mongoose.disconnect();
  await mongod.stop();
});

test("POST /satellite rejects unauthenticated callers", async () => {
  const res = await request(app).post("/satellite");
  assert.equal(res.status, 401);
});

test("POST /satellite requires both pre and post files", async () => {
  const res = await request(app)
    .post("/satellite")
    .set("Authorization", `Bearer ${token}`)
    .attach("pre", Buffer.from("fake-pre-png"), "pre.png");
  assert.equal(res.status, 400);
  assert.match(res.body.error, /pre and post/i);
});

test("POST /satellite handles image pair and returns GeoJSON damage structures", async () => {
  const res = await request(app)
    .post("/satellite")
    .set("Authorization", `Bearer ${token}`)
    .attach("pre", Buffer.from("fake-pre-image-content"), "pre.png")
    .attach("post", Buffer.from("fake-post-image-content"), "post.png");

  assert.equal(res.status, 200);
  assert.ok(res.body.pre_sha256);
  assert.ok(res.body.post_sha256);
  assert.ok(res.body.cells_geojson_r9);
  assert.ok(res.body.model_version);
});

test("POST /drone rejects unauthenticated callers", async () => {
  const res = await request(app).post("/drone");
  assert.equal(res.status, 401);
});

test("POST /drone requires an image file", async () => {
  const res = await request(app)
    .post("/drone")
    .set("Authorization", `Bearer ${token}`);
  assert.equal(res.status, 400);
});

test("POST /drone inspects high-res image and links to cell", async () => {
  const res = await request(app)
    .post("/drone")
    .set("Authorization", `Bearer ${token}`)
    .field("parent_cell_id", "8928308280fffff")
    .field("lat", "29.7604")
    .field("lon", "-95.3698")
    .attach("image", Buffer.from("fake-drone-image-bytes"), "drone.png");

  assert.equal(res.status, 200);
  assert.ok(res.body.image_sha256);
  assert.equal(res.body.parent_cell_id, "8928308280fffff");
  assert.ok(res.body.failure_modes);
  assert.ok("roof_collapse" in res.body.failure_modes);
  assert.ok("debris_field" in res.body.failure_modes);
});

test("GET /assessments/:id/cells returns hexagonal features for visualization", async () => {
  const doc = await Assessment.create({
    user: userId,
    prediction: "Severe Collapse",
    confidence: 0.92,
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[[-95.37, 29.76], [-95.36, 29.76], [-95.36, 29.77], [-95.37, 29.77], [-95.37, 29.76]]],
    },
  });

  // Call with the logged-in user's token
  const userDoc = await User.findOne({ email: "satellite-drone@rubicon.dev" });
  const ownAssessment = await Assessment.create({
    user: userDoc._id,
    prediction: "Severe Collapse",
    confidence: 0.95,
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[[-95.37, 29.76], [-95.36, 29.76], [-95.36, 29.77], [-95.37, 29.77], [-95.37, 29.76]]],
    },
  });

  const res = await request(app)
    .get(`/assessments/${ownAssessment._id}/cells`)
    .set("Authorization", `Bearer ${token}`);

  assert.equal(res.status, 200);
  assert.equal(res.body.assessmentId, ownAssessment._id.toString());
  assert.equal(res.body.cells.type, "FeatureCollection");
  assert.ok(res.body.cells.features.length >= 1);
  assert.ok(res.body.cells.features[0].properties.cell_id);
  assert.ok(res.body.cells.features[0].properties.severity);
});

test("GET /assessments/:id/verify returns cryptographic proof verification", async () => {
  const userDoc = await User.findOne({ email: "satellite-drone@rubicon.dev" });
  const doc = await Assessment.create({
    user: userDoc._id,
    prediction: "Severe Collapse",
    confidence: 0.91,
    hsiCid: "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco",
    txHash: "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    chainVerified: false,
  });

  const res = await request(app)
    .get(`/assessments/${doc._id}/verify`)
    .set("Authorization", `Bearer ${token}`);

  assert.equal(res.status, 200);
  assert.equal(res.body.assessmentId, doc._id.toString());
  assert.equal(res.body.cid, "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco");
  assert.equal(res.body.txHash, doc.txHash);
});
