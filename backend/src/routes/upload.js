import { Router } from "express";
import path from "node:path";
import multer from "multer";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";
import { runInference } from "../services/aiService.js";
import { transition } from "../services/assessmentService.js";
import { pinBuffer } from "../services/ipfs.js";
import { logAssessment } from "../services/chainService.js";
import { config } from "../config.js";

const router = Router();

const ALLOWED_EXTENSIONS = new Set([".tiff", ".tif", ".las"]);
const MAX_BYTES = config.maxUploadMb * 1024 * 1024;

function multerErrorToResponse(err, res) {
  if (err?.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: `file too large (max ${config.maxUploadMb}MB)` });
  }
  if (err?.message === "unsupported file type") {
    return res.status(400).json({ error: "only .tiff and .las files are supported" });
  }
  return res.status(400).json({ error: err?.message || "upload failed" });
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_BYTES },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) return cb(new Error("unsupported file type"));
    return cb(null, true);
  },
}).fields([
  { name: "hsi", maxCount: 1 },
  { name: "lidar", maxCount: 1 },
]);

// Phase 5 proof pipeline: content-address the uploads as real IPFS CIDv0 values
// (pinning when Pinata is configured), then log the assessment on-chain (real
// Amoy tx when the chain layer is configured, deterministic simulated tx
// otherwise — the simulated path leaves state at "analyzed").
async function attachProof(assessment, files, prediction) {
  const [hp, lp] = await Promise.all([
    pinBuffer(files.hsi.buffer, files.hsi.originalname),
    pinBuffer(files.lidar.buffer, files.lidar.originalname),
  ]);
  assessment.hsiCid = hp.cid;
  assessment.lidarCid = lp.cid;

  const logged = await logAssessment({ hsiCid: hp.cid, lidarCid: lp.cid, prediction });
  assessment.txHash = logged.txHash;
  assessment.chainVerified = logged.chainVerified;
  if (!logged.simulated) {
    transition(assessment, "chain_pending");
    transition(assessment, "chain_logged");
  }
  return assessment;
}

// POST /upload — accepts multipart fields hsi + lidar, runs (fake) AI, stores assessment.
router.post("/", requireAuth, (req, res) => {
  upload(req, res, async (err) => {
    if (err) return multerErrorToResponse(err, res);
    try {
      const hsi = req.files?.hsi?.[0];
      const lidar = req.files?.lidar?.[0];
      if (!hsi || !lidar) {
        return res.status(400).json({ error: "both hsi and lidar files are required" });
      }

      const assessment = await Assessment.create({
        user: req.user.sub,
        state: "uploaded",
        filename: { hsi: hsi.originalname, lidar: lidar.originalname },
        timestamps: { uploaded: new Date() },
      });

      transition(assessment, "analyzing");
      transition(assessment, "analyzed");
      const result = await runInference();

      assessment.prediction = result.prediction;
      assessment.confidence = result.confidence;
      assessment.class_probs = result.class_probs;
      assessment.geojson_polygon = result.geojson_polygon;
      assessment.model_version = result.model_version;

      await attachProof(assessment, { hsi, lidar }, result.prediction);

      await assessment.save();

      return res.status(201).json({
        assessmentId: assessment._id.toString(),
        state: assessment.state,
        ...result,
        createdAt: assessment.createdAt.toISOString(),
        links: { detail: `/assessments/${assessment._id.toString()}` },
      });
    } catch (error) {
      console.error("[upload] failed:", error);
      return res.status(500).json({ error: "internal error during upload" });
    }
  });
});

export default router;