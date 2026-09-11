// Polygon Amoy on-chain layer for the Pramaan Ledger.
// - Not configured (no RPC/private key/contract): deterministic SIMULATED record
//   (fake txHash, chainVerified: false) so the app works fully offline.
// - Configured: writes a real logAssessment(A) tx and reports chainVerified: true.
// - verifyOnChain() looks up AssessmentLogged events by CID; falls back to null so
//   the route answers from DB state when not on-chain.

import { createHash } from "node:crypto";
import { ethers } from "ethers";
import mongoose from "mongoose";
import ABI from "../abis/PramaanLedger.json" with { type: "json" };
import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("chain");

// Persisted cursor so verifyOnChain never rescans from block 0 (which grows
// unbounded and is slow on Amoy). Falls back to the last `SCAN_FALLBACK_BLOCKS`
// when no cursor exists yet.
const SCAN_FALLBACK_BLOCKS = 5000;

const chainIndexSchema = new mongoose.Schema(
  {
    key: { type: String, required: true, unique: true, default: "verify" },
    lastScannedBlock: { type: Number, required: true, default: 0 },
  },
  { timestamps: true },
);

const ChainIndex =
  mongoose.models.ChainIndex || mongoose.model("ChainIndex", chainIndexSchema);

export function chainConfigured() {
  return Boolean(config.contractAddress && config.deployerPrivateKey && config.amoyRpcUrl);
}

function normalizeKey() {
  const key = config.deployerPrivateKey.trim();
  return key.startsWith("0x") ? key : `0x${key}`;
}

function fakeTxHash(seed) {
  return `0x${createHash("sha256").update(seed).digest("hex")}`;
}

export function simulatedLog({ hsiCid, lidarCid }) {
  return { simulated: true, txHash: fakeTxHash(`${hsiCid}-${lidarCid}`), chainVerified: false };
}

export function buildContract(deps = {}) {
  const provider = deps.provider || new ethers.JsonRpcProvider(config.amoyRpcUrl);
  const wallet = deps.wallet || new ethers.Wallet(normalizeKey(), provider);
  return deps.contract || new ethers.Contract(config.contractAddress, ABI, wallet);
}

/**
 * Log an assessment on Amoy. Returns { simulated, txHash, chainVerified }.
 * Throws when configured but the tx fails (real infra problem — never quietly
 * downgraded).
 */
export async function logAssessment({ hsiCid, lidarCid, prediction }, deps = {}) {
  if (!chainConfigured()) {
    logger.warn("Amoy not configured — writing a simulated on-chain record");
    return simulatedLog({ hsiCid, lidarCid });
  }
  const contract = buildContract(deps);
  const tx = await contract.logAssessment(hsiCid, lidarCid, prediction);
  const receipt = await tx.wait(6);
  return { simulated: false, txHash: receipt.hash, chainVerified: true };
}

/**
 * Look for a CID in the ledger's AssessmentLogged events, scanning only from the
 * persisted cursor (never block 0). Returns { txHash, blockNumber, timestamp }
 * or null (unknown / chain not reachable).
 */
export async function verifyOnChain(cid, deps = {}) {
  if (!chainConfigured()) return null;
  try {
    const provider = deps.provider || new ethers.JsonRpcProvider(config.amoyRpcUrl);
    const contract =
      deps.contract || new ethers.Contract(config.contractAddress, ABI, provider);

    const latestBlock = deps.latestBlock ?? (await provider.getBlockNumber());
    let cursor = await ChainIndex.findOne({ key: "verify" });
    let fromBlock = cursor?.lastScannedBlock ?? Math.max(0, latestBlock - SCAN_FALLBACK_BLOCKS);
    if (fromBlock > latestBlock) fromBlock = latestBlock;

    const events = await contract.queryFilter(
      contract.filters.AssessmentLogged(),
      fromBlock,
      latestBlock,
    );
    const match = events.find((e) => String(e.args?.hsiCid || "") === String(cid));

    const newCursorBlock = match ? Number(match.blockNumber) : latestBlock;
    await ChainIndex.updateOne(
      { key: "verify" },
      { $set: { lastScannedBlock: newCursorBlock } },
      { upsert: true },
    );

    if (!match) return null;
    let timestamp = null;
    try {
      const block = await provider.getBlock(match.blockNumber);
      timestamp = block ? new Date(Number(block.timestamp) * 1000).toISOString() : null;
    } catch {
      // non-fatal — timestamp enrichment is best-effort
    }
    return { txHash: match.transactionHash, blockNumber: match.blockNumber, timestamp };
  } catch (err) {
    logger.warn("verify lookup failed — falling back to DB", {
      message: err?.message || String(err),
    });
    return null;
  }
}