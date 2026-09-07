import { Router } from "express";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";

const router = Router();

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
    createdAt: doc.createdAt.toISOString(),
  };
}

router.get("/", requireAuth, async (req, res) => {
  const docs = await Assessment.find({ user: req.user.sub }).sort({ createdAt: -1 }).limit(200);
  return res.json({ assessments: docs.map(listItem) });
});

router.get("/:id", requireAuth, async (req, res) => {
  const doc = await Assessment.findOne({ _id: req.params.id, user: req.user.sub });
  if (!doc) return res.status(404).json({ error: "assessment not found" });
  return res.json(doc.toJSON());
});

export default router;