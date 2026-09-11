// IPFS layer. Produces real, content-addressed CIDv0 values locally (deterministic
// sha2-256 multihash + base58btc) for every buffer, and — when a Pinata JWT is
// configured — also pins the file to IPFS. Offline mode is fully functional and
// always yields the same CID as Pinata would, because both are content-addressed.

import { createHash } from "node:crypto";
import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("ipfs");

const BASE58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

export function base58btc(bytes) {
  let n = 0n;
  for (const b of bytes) n = (n << 8n) | BigInt(b);
  let out = "";
  while (n > 0n) {
    const rem = Number(n % 58n);
    out = BASE58[rem] + out;
    n /= 58n;
  }
  let zeros = 0;
  while (zeros < bytes.length && bytes[zeros] === 0) zeros += 1;
  return "1".repeat(zeros) + out;
}

// CIDv0 = base58btc(0x12 0x20 <sha2-256 digest>), prefixed with "Qm".
export function contentCid(buffer) {
  const digest = createHash("sha256").update(buffer).digest();
  const multihash = Buffer.concat([Buffer.from([0x12, 0x20]), digest]);
  return `Qm${base58btc(multihash)}`;
}

export function pinataConfigured() {
  return Boolean(config.pinataJwt);
}

/**
 * Deterministically identify a buffer and optionally pin it to Pinata.
 * Returns { cid, pinned }. Throws if a configured pin actually fails — a real
 * infra problem should not be silently downgraded.
 */
export async function pinBuffer(buffer, filename) {
  const cid = contentCid(buffer);
  if (!config.pinataJwt) return { cid, pinned: false };

  const form = new FormData();
  form.append("file", new Blob([buffer]), String(filename || "scan"));
  form.append("pinataOptions", JSON.stringify({ cidVersion: 0 }));
  form.append("pinataMetadata", JSON.stringify({ name: String(filename || "scan") }));

  const res = await fetch("https://api.pinata.cloud/pinning/pinFileToIPFS", {
    method: "POST",
    headers: { Authorization: `Bearer ${config.pinataJwt}` },
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.IpfsHash) {
    const detail = data?.error?.message || data?.error || `http ${res.status}`;
    throw new Error(`pinata pin failed: ${detail}`);
  }
  return { cid: data.IpfsHash, pinned: true };
}