import { Router } from "express";
import path from "node:path";
import { mkdtempSync, rmSync } from "node:fs";
import { readFile } from "node:fs/promises";
import os from "node:os";
import multer from "multer";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";
import { runInference } from "../services/aiService.js";
import { transition } from "../services/assessmentService.js";
import { attachProof } from "../services/proofService.js";
import { persistFiles } from "../services/fileStorage.js";
import { validateMagic, ALLOWED_EXTENSIONS } from "../services/fileValidation.js";
import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("upload");

const router = Router();

const MAX_BYTES = config.maxUploadMb * 1024 * 1024;

function multerErrorToResponse(err, res) {
  if (err?.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: `file too large (max ${config.maxUploadMb}MB)` });
  }
  if (err?.message === "unsupported file type") {
    return res.status(400).json({ error: "only PNG, JPEG, or TIFF satellite images are supported" });
  }
  return res.status(400).json({ error: err?.message || "upload failed" });
}

// Spool uploads to a per-request temp dir instead of holding 2×100 MB in RAM.
// destination() reuses one dir per request; the handler removes it in finally.
const upload = multer({
  storage: multer.diskStorage({
    destination(req, _file, cb) {
      if (!req.uploadDir) req.uploadDir = mkdtempSync(path.join(os.tmpdir(), "rubicon-"));
      cb(null, req.uploadDir);
    },
    filename(req, file, cb) {
      const ext = path.extname(file.originalname).toLowerCase();
      cb(null, `${file.fieldname}${ext}`);
    },
  }),
  limits: { fileSize: MAX_BYTES },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(ext)) return cb(new Error("unsupported file type"));
    return cb(null, true);
  },
}).fields([
  { name: "pre", maxCount: 1 },
  { name: "post", maxCount: 1 },
]);

// POST /upload — accepts an xBD pre/post image pair, stores evidence, and pins/proves it.
router.post("/", requireAuth, (req, res) => {
  upload(req, res, async (err) => {
    try {
      if (err) return multerErrorToResponse(err, res);

      const pre = req.files?.pre?.[0];
      const post = req.files?.post?.[0];
      if (!pre || !post) {
        return res.status(400).json({ error: "Please upload pre- and post-disaster satellite images" });
      }

      const preBuf = await readFile(pre.path);
      const postBuf = await readFile(post.path);
      const magicErr = validateMagic(preBuf, "pre") || validateMagic(postBuf, "post");
      if (magicErr) return res.status(400).json({ error: magicErr });

      const files = {
        pre: { buffer: preBuf, originalname: pre.originalname },
        post: { buffer: postBuf, originalname: post.originalname },
      };
      const filenameObj = { pre: pre.originalname, post: post.originalname };

      const assessment = await Assessment.create({
        user: req.user.sub,
        state: "uploaded",
        filename: filenameObj,
        timestamps: { uploaded: new Date() },
      });

      try {
        assessment.storage = await persistFiles(assessment._id.toString(), files);

        transition(assessment, "analyzing");
        const result = await runInference(files, { requestId: req.requestId });
        transition(assessment, "analyzed");

        assessment.prediction = result.prediction;
        assessment.confidence = result.confidence;
        assessment.class_probs = result.class_probs;
        assessment.geojson_polygon = result.geojson_polygon;
        assessment.buildings_geojson = result.buildings_geojson;
        assessment.model_version = result.model_version;

        await attachProof(assessment, files, result.prediction);

        await assessment.save();
        req.app.locals.metrics.uploads += 1;

        logger.info("upload complete", {
          requestId: req.requestId,
          assessmentId: assessment._id.toString(),
          modelVersion: result.model_version,
        });

        return res.status(201).json({
          assessmentId: assessment._id.toString(),
          state: assessment.state,
          ...result,
          createdAt: assessment.createdAt.toISOString(),
          links: { detail: `/assessments/${assessment._id.toString()}` },
        });
      } catch (error) {
        req.app.locals.metrics.errors += 1;
        logger.error("upload failed", {
          requestId: req.requestId,
          message: error?.message || String(error),
        });
        assessment.state = "error";
        await assessment.save().catch(() => {});
        return res.status(500).json({ error: "internal error during upload" });
      }
    } finally {
      if (req.uploadDir) rmSync(req.uploadDir, { recursive: true, force: true });
    }
  });
});

export default router;
