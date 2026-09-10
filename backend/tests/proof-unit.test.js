import test from "node:test";
import assert from "node:assert/strict";

// Hermetic: force all chain/pinata config to empty BEFORE importing services so
// the offline/simulated paths are guaranteed (config is read at import time).
for (const key of ["PINATA_JWT", "AMOY_RPC_URL", "DEPLOYER_PRIVATE_KEY", "CONTRACT_ADDRESS"]) {
  delete process.env[key];
}

const { contentCid, base58btc, pinBuffer, pinataConfigured } = await import("../src/services/ipfs.js");
const { chainConfigured, simulatedLog, logAssessment, verifyOnChain } = await import("../src/services/chainService.js");

test("contentCid produces a valid deterministic CIDv0", () => {
  const a = contentCid(Buffer.from("rubicon", "utf8"));
  const b = contentCid(Buffer.from("rubicon", "utf8"));
  const c = contentCid(Buffer.from("rubicoN", "utf8"));
  assert.equal(a, b);
  assert.notEqual(a, c);
  assert.ok(a.startsWith("Qm"), "CIDv0 must start with Qm");
  assert.equal(a.length, 46 + 2, "CIDv0 is base58btc of 0x12 0x20 + 32 bytes");
  assert.match(a, /^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+$/);
});

test("base58btc round-trips known values", () => {
  assert.equal(base58btc(Buffer.from([0])), "1");
  assert.equal(base58btc(Buffer.from([0x00, 0x01])), "12");
});

test("pinBuffer reflects offline mode without a Pinata JWT", async () => {
  assert.equal(pinataConfigured(), false, "pinata config must be empty in tests");
  const { cid, pinned } = await pinBuffer(Buffer.from("fake-hsi"), "scan.tiff");
  assert.equal(pinned, false);
  assert.equal(cid, contentCid(Buffer.from("fake-hsi")));
});

test("chain layer reports not configured and simulates deterministically", async () => {
  assert.equal(chainConfigured(), false);
  const sim = simulatedLog({ hsiCid: "QmA", lidarCid: "QmB" });
  assert.equal(sim.simulated, true);
  assert.equal(sim.chainVerified, false);
  assert.ok(sim.txHash.startsWith("0x"));

  const logged = await logAssessment({ hsiCid: "QmA", lidarCid: "QmB", prediction: "Severe Collapse" });
  assert.equal(logged.simulated, true);
  assert.equal(logged.txHash, sim.txHash, "simulated tx is deterministic per seed");
});

test("verifyOnChain returns null when the chain is not configured", async () => {
  assert.equal(await verifyOnChain("QmA"), null);
});