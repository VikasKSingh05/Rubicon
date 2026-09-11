import test from "node:test";
import assert from "node:assert/strict";
import { MongoMemoryServer } from "mongodb-memory-server";
import mongoose from "mongoose";

import { config } from "../src/config.js";
import { startServer } from "../src/index.js";

let mongo;

test.before(async () => {
  config.isProd = false;
  process.env.JWT_SECRET = "test-secret";
  config.jwtSecret = "test-secret";
  mongo = await MongoMemoryServer.create();
});

test.after(async () => {
  await mongo?.stop();
});

test("startServer serves, drains, and disconnects Mongo on shutdown", async () => {
  const { server, shutdown } = await startServer({
    port: 0,
    mongoUri: mongo.getUri(),
  });
  try {
    assert.ok(server.listening, "server must be listening");
    assert.equal(mongoose.connection.readyState, 1, "Mongo must be connected");
  } finally {
    await shutdown("SIGTERM");
  }

  await new Promise((r) => setTimeout(r, 50));
  assert.ok(!server.listening, "server must drain on shutdown");
  assert.equal(mongoose.connection.readyState, 0, "Mongo must disconnect on shutdown");
});

test("shutdown is idempotent", async () => {
  const { shutdown } = await startServer({ port: 0, mongoUri: mongo.getUri() });
  await shutdown("SIGINT");
  await shutdown("SIGINT");
  assert.equal(mongoose.connection.readyState, 0);
});