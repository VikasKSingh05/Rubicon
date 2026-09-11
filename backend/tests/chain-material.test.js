import test from "node:test";
import assert from "node:assert/strict";
import { ethers } from "ethers";
import { MongoMemoryServer } from "mongodb-memory-server";
import mongoose from "mongoose";

// Hermetic chain-layer tests: config is set to "configured" but every network
// object is injected via deps with a *real* Provider whose network methods are
// overridden, so no RPC/private-key/contract is ever touched.
import { config } from "../src/config.js";
import { chainConfigured, logAssessment, verifyOnChain } from "../src/services/chainService.js";

let mongo;

test.before(async () => {
  for (const key of ["PINATA_JWT"]) delete process.env[key];
  config.pinataJwt = "";
  config.amoyRpcUrl = "https://rpc.example.invalid";
  config.deployerPrivateKey = "0x" + "1".repeat(64);
  config.contractAddress = "0x" + "2".repeat(40);

  mongo = await MongoMemoryServer.create();
  await mongoose.connect(mongo.getUri());
});

test.after(async () => {
  await mongoose.disconnect();
  await mongo?.stop();
});

const createdProviders = [];

function fakeProvider(event) {
  const events = event ? [event] : [];
  const provider = new ethers.JsonRpcProvider(); // real Provider, no network use
  provider.queryFilter = async (_filter, _from, _to) => events;
  provider.getBlockNumber = async () => 1000;
  provider.getBlock = async () => ({ timestamp: 1_700_000_000 });
  createdProviders.push(provider);
  return provider;
}

test.after(async () => {
  for (const p of createdProviders) await p.destroy();
  await mongoose.disconnect();
  await mongo?.stop();
});

function fakeContract({ events = [], failLog = false } = {}) {
  return {
    filters: { AssessmentLogged: () => ({ filter: "AssessmentLogged" }) },
    queryFilter: async () => events,
    logAssessment: async () => {
      if (failLog) throw new Error("revert: insufficient balance");
      return { wait: async () => ({ hash: "0xabc123", blockNumber: 42 }) };
    },
  };
}

function makeEvent({ cid, txHash, blockNumber }) {
  return {
    args: { hsiCid: cid },
    transactionHash: txHash,
    blockNumber,
  };
}

test("chainConfigured is true when all three params are set", () => {
  assert.equal(chainConfigured(), true);
});

test("logAssessment writes a real record via the contract and reports chainVerified", async () => {
  const contract = fakeContract();
  const logged = await logAssessment(
    { hsiCid: "QmA", lidarCid: "QmB", prediction: "Severe Collapse" },
    { contract },
  );
  assert.equal(logged.simulated, false);
  assert.equal(logged.chainVerified, true);
  assert.equal(logged.txHash, "0xabc123");
});

test("logAssessment propagates a failing transaction (no silent downgrade)", async () => {
  const contract = fakeContract({ failLog: true });
  await assert.rejects(
    logAssessment({ hsiCid: "QmA", lidarCid: "QmB", prediction: "None" }, { contract }),
    /revert/,
  );
});

test("verifyOnChain finds a matching event and stores a scan cursor", async () => {
  const event = makeEvent({ cid: "QmFound", txHash: "0x900d", blockNumber: 500 });
  const contract = fakeContract({ events: [event] });
  const provider = fakeProvider(null);

  const result = await verifyOnChain("QmFound", { provider, contract, latestBlock: 1000 });
  assert.equal(result.txHash, "0x900d");
  assert.equal(result.blockNumber, 500);
  assert.ok(result.timestamp);

  const cursor = await mongoose.models.ChainIndex.findOne({ key: "verify" });
  assert.ok(cursor, "cursor must be persisted");
  assert.equal(cursor.lastScannedBlock, 500);
});

test("verifyOnChain returns null on a miss but still advances the cursor", async () => {
  const contract = fakeContract({ events: [] });
  const provider = fakeProvider(null);
  const result = await verifyOnChain("QmMissing", { provider, contract, latestBlock: 900 });
  assert.equal(result, null);

  const cursor = await mongoose.models.ChainIndex.findOne({ key: "verify" });
  assert.equal(cursor.lastScannedBlock, 900, "cursor advances to latest on a miss");
});

test("verifyOnChain swallows network errors and falls back to null", async () => {
  const contract = {
    filters: { AssessmentLogged: () => ({ filter: "AssessmentLogged" }) },
    queryFilter: async () => {
      throw new Error("network down");
    },
  };
  assert.equal(await verifyOnChain("QmX", { contract, provider: {}, latestBlock: 42 }), null);
});