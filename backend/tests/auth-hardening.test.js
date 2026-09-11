import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import request from "supertest";
import jwt from "jsonwebtoken";
import { MongoMemoryServer } from "mongodb-memory-server";
import mongoose from "mongoose";

import createApp from "../src/app.js";
import { config, validateProductionSecrets } from "../src/config.js";
import RefreshToken from "../src/models/RefreshToken.js";

// Main app runs with rate limiting disabled so the auth-flow tests never 429.
config.rateLimitEnabled = false;

const app = createApp();
let mongo;

const EMAIL = "hardening@rubicon.dev";
const PASSWORD = "super-secret-1";

function uniqueEmail() {
  return `u${Math.random().toString(36).slice(2)}${Date.now()}@rubicon.dev`;
}

// The DB only ever stores sha-256(token); tests must compare against hashes.
function hash(token) {
  return createHash("sha256").update(token).digest("hex");
}

async function registerUser() {
  const email = uniqueEmail();
  const res = await request(app).post("/auth/register").send({ email, password: PASSWORD });
  assert.equal(res.status, 201);
  return { res, cookie: cookieFrom(res), userId: res.body.user.id };
}

function cookieFrom(res) {
  const setCookie = res.headers["set-cookie"];
  assert.ok(setCookie, "expected a Set-Cookie header");
  const c = setCookie.find((s) => s.startsWith("rubicon_refresh="));
  assert.ok(c, "expected rubicon_refresh cookie");
  return c;
}

test.before(async () => {
  process.env.JWT_SECRET = "test-secret";
  config.jwtSecret = "test-secret";
  config.rateLimitEnabled = false;
  mongo = await MongoMemoryServer.create();
  await mongoose.connect(mongo.getUri());
});

test.after(async () => {
  await mongoose.disconnect();
  await mongo?.stop();
});

test("login sets an httpOnly refresh cookie and a short-lived access token", async () => {
  const reg = await request(app).post("/auth/register").send({ email: EMAIL, password: PASSWORD });
  assert.equal(reg.status, 201);

  const cookie = cookieFrom(reg);
  assert.match(cookie, /HttpOnly/i);
  assert.match(cookie, /SameSite=Strict/i);
  assert.match(cookie, /Path=\/auth/i);

  const payload = jwt.decode(reg.body.token);
  const ttlHours = Math.round((payload.exp - payload.iat) / 3600);
  assert.ok(ttlHours <= 1, "access token must be short-lived, got hours: " + ttlHours);
});

test("refresh rotates the refresh token and mints a new access token", async () => {
  const { cookie: firstCookie, userId } = await registerUser();
  const oldToken = firstCookie.split(";")[0].split("=")[1];

  const refreshed = await request(app).post("/auth/refresh").set("Cookie", firstCookie);
  assert.equal(refreshed.status, 200);
  assert.ok(refreshed.body.token);

  const newToken = cookieFrom(refreshed).split(";")[0].split("=")[1];
  assert.notEqual(newToken, oldToken, "refresh token must rotate");

  assert.equal(await RefreshToken.exists({ user: userId, hash: hash(oldToken) }), null);
  assert.ok(await RefreshToken.exists({ user: userId, hash: hash(newToken) }));
  assert.equal((await RefreshToken.find({ user: userId })).length, 1);
});

test("a rotated (reused) refresh token is rejected", async () => {
  const { cookie: firstCookie } = await registerUser();
  const refreshed = await request(app).post("/auth/refresh").set("Cookie", firstCookie);
  assert.equal(refreshed.status, 200);
  const newToken = cookieFrom(refreshed).split(";")[0].split("=")[1];

  const replayed = await request(app).post("/auth/refresh").set("Cookie", firstCookie);
  assert.equal(replayed.status, 401, "reused refresh token must be rejected");
  assert.ok(
    await RefreshToken.exists({ hash: hash(newToken) }),
    "the rotated token must remain usable",
  );
});

test("logout revokes the refresh session and clears the cookie", async () => {
  const { res: login, cookie } = await registerUser();
  const userId = login.body.user.id;

  const logout = await request(app).post("/auth/logout").set("Cookie", cookie);
  assert.equal(logout.status, 200);

  assert.equal(
    (await RefreshToken.find({ user: userId })).length,
    0,
    "refresh session must be revoked",
  );

  const after = await request(app).post("/auth/refresh").set("Cookie", cookie);
  assert.equal(after.status, 401);
});

test("access token authorizes authenticated routes", async () => {
  const login = await request(app).post("/auth/login").send({ email: EMAIL, password: PASSWORD });
  const res = await request(app)
    .get("/assessments")
    .set("Authorization", `Bearer ${login.body.token}`);
  assert.equal(res.status, 200);
});

test("register enforces a minimum password length of 8", async () => {
  const res = await request(app)
    .post("/auth/register")
    .send({ email: "shortpw@rubicon.dev", password: "short" });
  assert.equal(res.status, 400);
  assert.equal(res.body.error, "password must be at least 8 characters");
});

test("rate limiting returns 429 on the auth endpoints after the threshold", async () => {
  config.rateLimitEnabled = true;
  const limitedApp = createApp();
  config.rateLimitEnabled = false;

  let last;
  for (let i = 0; i < 11; i += 1) {
    last = await request(limitedApp).post("/auth/login").send({ email: EMAIL, password: "nope" });
  }
  assert.equal(last.status, 429);
  assert.equal(last.body.error, "too many requests");
});

test("validateProductionSecrets refuses the default JWT secret in production", () => {
  const originalIsProd = config.isProd;
  const originalSecret = process.env.JWT_SECRET;
  const originalConfigSecret = config.jwtSecret;
  try {
    config.isProd = true;
    process.env.JWT_SECRET = "test-secret";
    config.jwtSecret = "dev-secret-change-me";
    assert.throws(() => validateProductionSecrets(), /JWT_SECRET/);

    config.jwtSecret = "a-strong-prod-secret";
    assert.doesNotThrow(() => validateProductionSecrets());

    process.env.JWT_SECRET = "";
    config.jwtSecret = "";
    assert.throws(() => validateProductionSecrets(), /JWT_SECRET/);
  } finally {
    config.isProd = originalIsProd;
    process.env.JWT_SECRET = originalSecret;
    config.jwtSecret = originalConfigSecret;
  }
});