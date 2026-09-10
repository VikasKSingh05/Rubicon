import { Router } from "express";
import { requireAuth } from "../middleware/auth.js";
import { asyncHandler } from "../middleware/asyncHandler.js";
import Assessment from "../models/Assessment.js";
import { verifyOnChain } from "../services/chainService.js";

const router = Router();

// GET /chain/verify/:cid — Phase 5.
// When the Amoy layer is configured the route looks for a matching
// AssessmentLogged event on-chain; otherwise it answers from the DB.
router.get(
  "/verify/:cid",
  requireAuth,
  asyncHandler(async (req, res) => {
    const cid = String(req.params.cid || "");
    if (!cid) return res.status(400).json({ error: "cid is required" });

    // 1) DB lookup (user-scoped, always authoritative for assessmentId/state).
    const doc = await Assessment.findOne({
      user: req.user.sub,
      $or: [{ hsiCid: cid }, { lidarCid: cid }],
    }).select("state hsiCid lidarCid txHash chainVerified");

    const assessmentId = doc?._id?.toString() ?? null;
    const state = doc?.state ?? null;

    // 2) On-chain verification (real Amoy event lookup when configured).
    const chain = await verifyOnChain(cid);

    return res.json({
      cid,
      verified: chain?.verified ?? doc?.chainVerified ?? false,
      txHash: chain?.txHash ?? doc?.txHash ?? null,
      assessmentId,
      state,
    });
  }),
);

export default router;