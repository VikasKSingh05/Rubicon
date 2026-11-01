// Phase 5 proof pipeline shared by the synchronous /upload path and the async
// /assessments pipeline.
//
// Content-addresses both uploads as real IPFS CIDv0 values (pinning them when
// Pinata is configured), then logs the assessment on-chain (real Amoy tx when
// the chain layer is configured, deterministic simulated tx otherwise — the
// simulated path leaves state at "analyzed").
import { transition } from "./assessmentService.js";
import { pinBuffer } from "./ipfs.js";
import { logAssessment } from "./chainService.js";

export async function attachProof(assessment, files, prediction) {
  const primaryA = files.pre || files.hsi;
  const primaryB = files.post || files.lidar;
  if (!primaryA || !primaryB) {
    throw new Error("proof requires two uploaded buffers");
  }

  const [pA, pB] = await Promise.all([
    pinBuffer(primaryA.buffer, primaryA.originalname),
    pinBuffer(primaryB.buffer, primaryB.originalname),
  ]);

  if (files.pre) {
    assessment.preCid = pA.cid;
    assessment.postCid = pB.cid;
  }
  assessment.hsiCid = pA.cid;
  assessment.lidarCid = pB.cid;

  const logged = await logAssessment({ hsiCid: pA.cid, lidarCid: pB.cid, prediction });
  assessment.txHash = logged.txHash;
  assessment.chainVerified = logged.chainVerified;
  if (!logged.simulated) {
    transition(assessment, "chain_pending");
    transition(assessment, "chain_logged");
  }
  return assessment;
}
