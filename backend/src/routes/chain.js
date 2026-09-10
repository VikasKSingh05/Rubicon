import { Router } from "express";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";

const router = Router();

// GET /chain/verify/:cid — stub until Phase 5 wires the real on-chain log.
// Locates the owning user's assessment that references the CID and reports its
// current chain status. Unknown CIDs return verified: false with null metadata.
router.get("/verify/:cid", requireAuth, async (req, res) => {
  const cid = String(req.params.cid || "");
  if (!cid) return res.status(400).json({ error: "cid is required" });

  const doc = await Assessment.findOne({
    user: req.user.sub,
    $or: [{ hsiCid: cid }, { lidarCid: cid }],
  }).select("state hsiCid lidarCid txHash chainVerified");

  if (!doc) {
    return res.json({
      cid,
      verified: false,
      txHash: null,
      assessmentId: null,
      state: null,
    });
  }

  return res.json({
    cid,
    verified: doc.chainVerified,
    txHash: doc.txHash,
    assessmentId: doc._id.toString(),
    state: doc.state,
  });
});

export default router;