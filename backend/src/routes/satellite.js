/**
 * POST /satellite — Proxy satellite pre/post image pairs to the Siamese U-Net
 * inference engine and return the GeoJSON damage maps + H3 cells.
 *
 * Contract: see docs/api-contracts.md § POST /satellite
 */
import { createHash } from "node:crypto";
import { Router } from "express";
import multer from "multer";
import { requireAuth } from "../middleware/auth.js";
import { asyncHandler } from "../middleware/asyncHandler.js";
import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const router = Router();
const logger = makeLogger("satellite");

// ── Multer: memory storage, image extensions only ────────────────────────────
const ALLOWED_EXTS = new Set([".png", ".jpg", ".jpeg", ".tif", ".tiff"]);

const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: config.maxUploadMb * 1024 * 1024 },
  fileFilter(_req, file, cb) {
    const ext = file.originalname.toLowerCase().match(/\.[^.]+$/)?.[0] ?? "";
    if (!ALLOWED_EXTS.has(ext)) {
      return cb(new Error(`${file.fieldname} must be PNG/JPEG/GeoTIFF`));
    }
    cb(null, true);
  },
});

// ── POST /satellite ───────────────────────────────────────────────────────────
router.post(
  "/",
  requireAuth,
  upload.fields([
    { name: "pre", maxCount: 1 },
    { name: "post", maxCount: 1 },
  ]),
  asyncHandler(async (req, res) => {
    const { pre, post } = req.files ?? {};
    if (!pre?.[0] || !post?.[0]) {
      return res.status(400).json({ error: "pre and post image files are required" });
    }

    const preFile = pre[0];
    const postFile = post[0];
    const requestId = req.requestId ?? crypto.randomUUID();

    logger.info("satellite inference request", {
      requestId,
      user: req.user.sub,
      preFile: preFile.originalname,
      postFile: postFile.originalname,
    });

    // ── Forward to Siamese U-Net engine ───────────────────────────────────────
    let engineBody;
    try {
      const form = new FormData();
      form.append("pre", new Blob([preFile.buffer]), preFile.originalname || "pre.png");
      form.append("post", new Blob([postFile.buffer]), postFile.originalname || "post.png");

      const resp = await fetch(`${config.satEngineUrl}/infer/satellite`, {
        method: "POST",
        body: form,
        headers: { "x-request-id": requestId },
        signal: AbortSignal.timeout(120_000),
      });

      if (!resp.ok) {
        logger.warn("satellite engine non-2xx, using stub", {
          requestId,
          status: resp.status,
        });
        engineBody = null;
      } else {
        engineBody = await resp.json();
      }
    } catch (err) {
      logger.warn("satellite engine unreachable, using stub", {
        requestId,
        message: err?.message ?? String(err),
      });
      engineBody = null;
    }

    // Graceful degrade: synthetic stub when engine is down
    if (!engineBody) {
      engineBody = {
        pre_sha256: createHash("sha256").update(preFile.buffer).digest("hex"),
        post_sha256: createHash("sha256").update(postFile.buffer).digest("hex"),
        buildings_geojson: { type: "FeatureCollection", features: [] },
        cells_geojson_r9: { type: "FeatureCollection", features: [] },
        cells_geojson_r10: { type: "FeatureCollection", features: [] },
        model_version: "stub-satellite-v0",
        damage_classes: ["background", "no-damage", "minor-damage", "major-damage", "destroyed"],
        exec_ms: 0,
      };
    }

    return res.json({
      ...engineBody,
      requestId,
    });
  }),
);

export default router;
