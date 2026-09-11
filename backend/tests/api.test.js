import test from "node:test";
import assert from "node:assert/strict";
import request from "supertest";
import { MongoMemoryServer } from "mongodb-memory-server";
import mongoose from "mongoose";

import createApp from "../src/app.js";
import { config } from "../src/config.js";
import { contentCid } from "../src/services/ipfs.js";
import Assessment from "../src/models/Assessment.js";

// Phase 7 rate limiting is exercised in auth-hardening.test.js against a fresh
// app; the main suite keeps it off so the shared app instance never 429s.
config.rateLimitEnabled = false;

const app = createApp();
let mongo;

// Force the Phase 5 proof layers into offline/simulated mode regardless of the
// host environment, so tests never touch Pinata or Amoy.
config.pinataJwt = "";
config.amoyRpcUrl = "";
config.deployerPrivateKey = "";
config.contractAddress = "";
// Force a fast, guaranteed-unreachable engine so uploads always take the stub path.
config.aiEngineUrl = "http://127.0.0.1:65530";

const EMAIL = "demo@rubicon.dev";
const PASSWORD = "secret123";

async function registerUser(email = EMAIL, password = PASSWORD) {
  const res = await request(app).post("/auth/register").send({ email, password });
  return res;
}

// Unwraps the JWT subject (user id) so pagination tests can create docs for a user.
function jwtSub(token) {
  const payload = token.split(".")[1];
  return JSON.parse(Buffer.from(payload, "base64url").toString("utf8")).sub;
}

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
  assert.equal(res.body.mongo, "connected");
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

// Magic-valid upload payloads (Phase 7 validates content, not just extension).
const FAKE_HSI = Buffer.concat([Buffer.from([0x49, 0x49, 0x2a, 0x00]), Buffer.from("fake-hsi")]);
const FAKE_LIDAR = Buffer.concat([Buffer.from("LASF"), Buffer.from("fake-lidar")]);

async function uploadFiles(token, hsiName = "scan.tiff", lidarName = "scan.las") {
  return request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", FAKE_HSI, hsiName)
    .attach("lidar", FAKE_LIDAR, lidarName);
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
  assert.ok(
    item.geojson_polygon &&
      item.geojson_polygon.type === "Polygon" &&
      Array.isArray(item.geojson_polygon.coordinates),
  );
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
  assert.equal(detail.body.hsiCid, contentCid(FAKE_HSI));
  assert.equal(detail.body.lidarCid, contentCid(FAKE_LIDAR));
  // Offline proof: a deterministic fake tx, chain not yet verified.
  assert.equal(detail.body.chainVerified, false);
  assert.ok(detail.body.txHash.startsWith("0x"));
  assert.equal(detail.body.state, "analyzed");
});

test("register rejects a malformed email", async () => {
  const res = await request(app)
    .post("/auth/register")
    .send({ email: "not-an-email", password: "secret123" });
  assert.equal(res.status, 400);
  assert.equal(res.body.error, "email must be a valid email address");
});

test("unknown routes return a JSON 404", async () => {
  const res = await request(app).get("/no/such/route");
  assert.equal(res.status, 404);
  assert.deepEqual(res.body, { error: "not found" });
});

test("async handler surfaces internal errors as JSON 500s (no hung request)", async () => {
  const token = await getToken();
  // "zzz" is not a valid ObjectId → mongoose CastError inside the handler.
  const res = await request(app).get("/assessments/zzz").set(authHeader(token));
  assert.equal(res.status, 500);
  assert.deepEqual(res.body, { error: "internal server error" });
});

test("assessments list paginates with limit/offset and reports total", async () => {
  const token = await getToken("page@rubicon.dev", "secret789");
  const userId = jwtSub(token);
  for (let i = 0; i < 3; i += 1) {
    await Assessment.create({
      user: userId,
      state: "analyzed",
      prediction: "Moderate",
      confidence: 0.75,
      model_version: "stub-v0",
      geojson_polygon: {
        type: "Polygon",
        coordinates: [[[-95.4, 29.8], [-95.2, 29.8], [-95.2, 30.0], [-95.4, 29.8]]],
      },
      timestamps: { uploaded: new Date(), analyzing: new Date(), analyzed: new Date() },
    });
  }

  const page = await request(app).get("/assessments?limit=1&offset=1").set(authHeader(token));
  assert.equal(page.status, 200);
  assert.equal(page.body.total, 3);
  assert.equal(page.body.limit, 1);
  assert.equal(page.body.offset, 1);
  assert.equal(page.body.assessments.length, 1);
  assert.ok("txHash" in page.body.assessments[0]);
  assert.ok("chainVerified" in page.body.assessments[0]);

  const second = await request(app).get("/assessments?offset=2").set(authHeader(token));
  assert.equal(second.body.limit, 50, "default limit applies");
  assert.equal(second.body.assessments.length, 1);
});

test("upload accepts .laz LiDAR alongside .tiff HSI", async () => {
  const token = await getToken();
  const res = await request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", FAKE_HSI, "scans.tiff")
    .attach("lidar", FAKE_LIDAR, "scans.laz");
  assert.equal(res.status, 201);
  assert.equal(res.body.model_version, "stub-v0");
});

test("upload rejects files whose content does not match the extension", async () => {
  const token = await getToken();
  const res = await request(app)
    .post("/upload")
    .set(authHeader(token))
    .attach("hsi", Buffer.from("garbage bytes not a tiff"), "scan.tiff")
    .attach("lidar", FAKE_LIDAR, "scan.las");
  assert.equal(res.status, 400);
  assert.equal(res.body.error, "hsi file is not a valid TIFF");
});

test("files endpoint streams the persisted originals to the owner", async () => {
  const token = await getToken();
  const up = await uploadFiles(token);

  const hsi = await request(app).get(`/files/${up.body.assessmentId}/hsi`).set(authHeader(token));
  assert.equal(hsi.status, 200);
  assert.equal(hsi.type, "image/tiff");
  assert.ok(hsi.body.equals(FAKE_HSI));
  assert.ok(hsi.headers["content-disposition"].includes("scan.tiff"));

  const lidar = await request(app).get(`/files/${up.body.assessmentId}/lidar`).set(authHeader(token));
  assert.equal(lidar.status, 200);
  assert.equal(lidar.type, "application/octet-stream");
  assert.ok(lidar.body.equals(FAKE_LIDAR));
});

test("files endpoint enforces ownership, valid kinds, and 404s", async () => {
  const token = await getToken();
  const up = await uploadFiles(token);
  const otherToken = await getToken("other@rubicon.dev", "secret456");

  const other = await request(app).get(`/files/${up.body.assessmentId}/hsi`).set(authHeader(otherToken));
  assert.equal(other.status, 404, "other users must not read files");

  const anon = await request(app).get(`/files/${up.body.assessmentId}/hsi`);
  assert.equal(anon.status, 401);

  const badKind = await request(app).get(`/files/${up.body.assessmentId}/pointcloud`).set(authHeader(token));
  assert.equal(badKind.status, 400);
  assert.equal(badKind.body.error, "kind must be 'hsi' or 'lidar'");

  const missing = await request(app).get(`/files/000000000000000000000000/hsi`).set(authHeader(token));
  assert.equal(missing.status, 404);
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