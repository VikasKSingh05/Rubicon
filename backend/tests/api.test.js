import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import { MongoMemoryServer } from "mongodb-memory-server";
import mongoose from "mongoose";

import createApp from "../src/app.js";
import { config } from "../src/config.js";
import { contentCid } from "../src/services/ipfs.js";

const app = createApp();
let mongo;

// Force the Phase 5 proof layers into offline/simulated mode regardless of the
// host environment, so tests never touch Pinata or Amoy.
config.pinataJwt = "";
config.amoyRpcUrl = "";
config.deployerPrivateKey = "";
config.contractAddress = "";

const EMAIL = "demo@rubicon.dev";
const PASSWORD = "secret123";

async function registerUser(email = EMAIL, password = PASSWORD) {
  const res = await request(app).post("/auth/register").send({ email, password });
  return res;
}

// Users are registered once per email; later requests log in instead. Returns a token.
async function getToken(email = EMAIL, password = PASSWORD) {
  const registered = await registerUser(email, password);
  if (registered.status === 409) {
    const login = await request(app).post("/auth/login").send({ email, password });
    return login.body.token;
  }
  return registered.body.token;
}

test.before(async () => {
  process.env.JWT_SECRET = "test-secret";
  config.jwtSecret = "test-secret";
  if (process.env.MONGO_TEST_URI) {
    await mongoose.connect(process.env.MONGO_TEST_URI);
  } else {
    mongo = await MongoMemoryServer.create();
    await mongoose.connect(mongo.getUri());
  }
});

test.after(async () => {
  await mongoose.disconnect();
  await mongo?.stop();
});

test("health endpoint works", async () => {
  const res = await request(app).get("/health");
  assert.equal(res.status, 200);
  assert.equal(res.body.status, "ok");
});

test("register creates a user and returns a JWT", async () => {
  const res = await registerUser();
  assert.equal(res.status, 201);
  assert.ok(res.body.token);
  assert.equal(res.body.user.email, EMAIL);
});

test("register rejects duplicate email", async () => {
  const res = await registerUser();
  assert.equal(res.status, 409);
});

test("login works with correct credentials", async () => {
  const res = await request(app).post("/auth/login").send({ email: EMAIL, password: PASSWORD });
  assert.equal(res.status, 200);
  assert.ok(res.body.token);
});

test("login rejects bad password", async () => {
  const res = await request(app).post("/auth/login").send({ email: EMAIL, password: "wrong" });
  assert.equal(res.status, 401);
});

test("auth endpoints require email and password", async () => {
  const res = await request(app).post("/auth/register").send({});
  assert.equal(res.status, 400);
});

function authHeader(token) {
  return { Authorization: `Bearer ${token}` };
}

async function uploadFiles(token, hsiName = "scan.tiff", lidarName = "scan.las") {
  return request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", Buffer.from("fake-hsi"), hsiName)
    .attach("lidar", Buffer.from("fake-lidar"), lidarName);
}

test("upload requires auth", async () => {
  const res = await request(app).post("/upload").attach("hsi", Buffer.from("x"), "a.tiff");
  assert.equal(res.status, 401);
});

test("upload rejects missing files", async () => {
  const token = await getToken();
  const res = await request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", Buffer.from("x"), "a.tiff");
  assert.equal(res.status, 400);
});

test("upload rejects disallowed file types", async () => {
  const token = await getToken();
  const res = await request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", Buffer.from("x"), "a.exe")
    .attach("lidar", Buffer.from("y"), "b.las");
  assert.equal(res.status, 400);
});

test("upload returns the exact /predict contract shape", async () => {
  const token = await getToken();
  const res = await uploadFiles(token);
  assert.equal(res.status, 201);
  const keys = [
    "assessmentId",
    "state",
    "prediction",
    "confidence",
    "class_probs",
    "geojson_polygon",
    "model_version",
    "createdAt",
    "links",
  ];
  assert.deepEqual(Object.keys(res.body).sort(), keys.sort());
  assert.equal(res.body.state, "analyzed");
  assert.equal(res.body.geojson_polygon.type, "Polygon");
  for (const cls of ["None", "Moderate", "Severe Collapse"]) {
    assert.ok(cls in res.body.class_probs, `missing class ${cls}`);
  }
  assert.ok(res.body.links.detail.startsWith("/assessments/"));
});

test("assessments list returns uploaded assessments for the user", async () => {
  const token = await getToken();
  await uploadFiles(token);
  const res = await request(app).get("/assessments").set(authHeader(token));
  assert.equal(res.status, 200);
  assert.ok(Array.isArray(res.body.assessments));
  assert.ok(res.body.assessments.length >= 1);
  const item = res.body.assessments[0];
  assert.ok(item.id);
  assert.ok(item.prediction);
  assert.ok(item.center && typeof item.center.lng === "number");
});

test("assessment detail returns full document and chain fields exist", async () => {
  const token = await getToken();
  const up = await uploadFiles(token);
  const detail = await request(app)
    .get(up.body.links.detail)
    .set(authHeader(token));
  assert.equal(detail.status, 200);
  assert.equal(detail.body.id, up.body.assessmentId);
  // Phase-5 fields present so no migration is needed later:
  assert.ok("hsiCid" in detail.body);
  assert.ok("lidarCid" in detail.body);
  assert.ok("txHash" in detail.body);
  assert.ok("chainVerified" in detail.body);
  assert.equal(detail.body.state, "analyzed");
});

test("upload content-addresses files as real CIDv0 and reports simulated chain status", async () => {
  const token = await getToken();
  const res = await uploadFiles(token, "proof.tiff", "proof.las");
  assert.equal(res.status, 201);

  const detail = await request(app).get(res.body.links.detail).set(authHeader(token));
  assert.equal(detail.body.hsiCid, contentCid(Buffer.from("fake-hsi")));
  assert.equal(detail.body.lidarCid, contentCid(Buffer.from("fake-lidar")));
  // Offline proof: a deterministic fake tx, chain not yet verified.
  assert.equal(detail.body.chainVerified, false);
  assert.ok(detail.body.txHash.startsWith("0x"));
  assert.equal(detail.body.state, "analyzed");
});

test("assessment detail is scoped to the owning user", async () => {
  const firstToken = await getToken();
  const up = await uploadFiles(firstToken);
  const otherToken = await getToken("other@rubicon.dev", "secret456");
  const res = await request(app).get(up.body.links.detail).set(authHeader(otherToken));
  assert.equal(res.status, 404);
});

test("chain verify stub reports the assessment for a known CID", async () => {
  const token = await getToken();
  const up = await uploadFiles(token);
  const detail = await request(app).get(up.body.links.detail).set(authHeader(token));
  const res = await request(app).get(`/chain/verify/${detail.body.hsiCid}`).set(authHeader(token));
  assert.equal(res.status, 200);
  assert.equal(res.body.cid, detail.body.hsiCid);
  assert.equal(res.body.verified, false);
  // CIDs are content-derived, so any upload of the same file matches; the
  // stub must always report a concrete assessment once at least one exists.
  assert.ok(res.body.assessmentId);
  assert.equal(res.body.state, "analyzed");
  assert.ok(res.body.txHash);
});

test("chain verify requires auth and handles unknown CIDs", async () => {
  const anon = await request(app).get("/chain/verify/QmUnknown123");
  assert.equal(anon.status, 401);

  const token = await getToken();
  const res = await request(app).get("/chain/verify/QmUnknown123").set(authHeader(token));
  assert.equal(res.status, 200);
  assert.deepEqual(res.body, {
    cid: "QmUnknown123",
    verified: false,
    txHash: null,
    assessmentId: null,
    state: null,
  });
});