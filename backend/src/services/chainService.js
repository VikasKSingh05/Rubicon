// Polygon Amoy on-chain layer for the Pramaan Ledger.
// - Not configured (no RPC/private key/contract): deterministic SIMULATED record
//   (fake txHash, chainVerified: false) so the app works fully offline.
// - Configured: writes a real logAssessment(A) tx and reports chainVerified: true.
// - verifyOnChain() looks up AssessmentLogged events by CID; falls back to null so
//   the route answers from DB state when not on-chain.

import { createHash } from "node:crypto";
import { ethers } from "ethers";
import ABI from "../abis/PramaanLedger.json" with { type: "json" };
import { config } from "../config.js";

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
    console.warn(
      "[chain] Amoy not configured — writing a simulated on-chain record. " +
        "Set AMOY_RPC_URL, DEPLOYER_PRIVATE_KEY and CONTRACT_ADDRESS for a real record.",
    );
    return simulatedLog({ hsiCid, lidarCid });
  }
  const contract = buildContract(deps);
  const tx = await contract.logAssessment(hsiCid, lidarCid, prediction);
  const receipt = await tx.wait(6);
  return { simulated: false, txHash: receipt.hash, chainVerified: true };
}

/**
 * Look for a CID in the ledger's AssessmentLogged events.
 * Returns { txHash, blockNumber, timestamp } or null (unknown / chain not reachable).
 */
export async function verifyOnChain(cid, deps = {}) {
  if (!chainConfigured()) return null;
  try {
    const provider = deps.provider || new ethers.JsonRpcProvider(config.amoyRpcUrl);
    const contract = new ethers.Contract(config.contractAddress, ABI, provider);
    const events = await contract.queryFilter(contract.filters.AssessmentLogged(), 0);
    const match = events.find((e) => String(e.args?.hsiCid || "") === String(cid));
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
    console.warn(`[chain] verify lookup failed (${err.message}) — falling back to DB`);
    return null;
  }
}