import { Router } from "express";
import { requireAuth } from "../middleware/auth.js";
import { asyncHandler } from "../middleware/asyncHandler.js";
import Assessment from "../models/Assessment.js";

const router = Router();

const DEFAULT_LIMIT = 50;
const MAX_LIMIT = 200;

function bboxCenter(polygon) {
  const ring = polygon?.coordinates?.[0] ?? [];
  if (ring.length === 0) return null;
  let minLng = Infinity;
  let maxLng = -Infinity;
  let minLat = Infinity;
  let maxLat = -Infinity;
  for (const [lng, lat] of ring) {
    minLng = Math.min(minLng, lng);
    maxLng = Math.max(maxLng, lng);
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
  }
  return { lng: (minLng + maxLng) / 2, lat: (minLat + maxLat) / 2 };
}

function listItem(doc) {
  return {
    id: doc._id.toString(),
    prediction: doc.prediction,
    confidence: doc.confidence,
    severity: doc.severity,
    state: doc.state,
    center: bboxCenter(doc.geojson_polygon),
    txHash: doc.txHash,
    chainVerified: doc.chainVerified,
    createdAt: doc.createdAt.toISOString(),
  };
}

// GET /assessments?limit=&offset= — newest-first, capped at MAX_LIMIT.
router.get(
  "/",
  requireAuth,
  asyncHandler(async (req, res) => {
    const limit = Math.min(Math.max(Number(req.query.limit) || DEFAULT_LIMIT, 1), MAX_LIMIT);
    const offset = Math.max(Number(req.query.offset) || 0, 0);
    const filter = { user: req.user.sub };

    const [docs, total] = await Promise.all([
      Assessment.find(filter).sort({ createdAt: -1 }).skip(offset).limit(limit),
      Assessment.countDocuments(filter),
    ]);
    return res.json({ assessments: docs.map(listItem), total, limit, offset });
  }),
);

router.get(
  "/:id",
  requireAuth,
  asyncHandler(async (req, res) => {
    const doc = await Assessment.findOne({ _id: req.params.id, user: req.user.sub });
    if (!doc) return res.status(404).json({ error: "assessment not found" });
    return res.json(doc.toJSON());
  }),
);

export default router;