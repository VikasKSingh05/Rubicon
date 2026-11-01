import { Router } from "express";
import path from "node:path";
import { mkdtempSync, rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import multer from "multer";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";
import { runInference } from "../services/aiService.js";
import { transition } from "../services/assessmentService.js";
import { pinBuffer } from "../services/ipfs.js";
import { logAssessment } from "../services/chainService.js";
import { config } from "../config.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("upload");

const router = Router();

const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".tiff", ".tif", ".las", ".laz"]);
const MAX_BYTES = config.maxUploadMb * 1024 * 1024;

function multerErrorToResponse(err, res) {
  if (err?.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({ error: `file too large (max ${config.maxUploadMb}MB)` });
  }
  if (err?.message === "unsupported file type") {
    return res.status(400).json({ error: "only image files (.png/.jpg/.tif) or point clouds (.las/.laz) are supported" });
  }
  return res.status(400).json({ error: err?.message || "upload failed" });
}

function validateMagic(buf, kind) {
  if (!buf || buf.length < 4) return "file payload too small or empty";
  if (kind === "hsi") {
    const little = buf[0] === 0x49 && buf[1] === 0x49 && buf[2] === 0x2a && buf[3] === 0x00;
    const big = buf[0] === 0x4d && buf[1] === 0x4d && buf[2] === 0x00 && buf[3] === 0x2a;
    return little || big ? null : "hsi file is not a valid TIFF";
  }
  if (kind === "pre" || kind === "post") {
    const isPng = buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47;
    const isJpg = buf[0] === 0xff && buf[1] === 0xd8;
    const isTiffLittle = buf[0] === 0x49 && buf[1] === 0x49 && buf[2] === 0x2a && buf[3] === 0x00;
    const isTiffBig = buf[0] === 0x4d && buf[1] === 0x4d && buf[2] === 0x00 && buf[3] === 0x2a;
    return isPng || isJpg || isTiffLittle || isTiffBig ? null : `${kind} image must be a valid PNG, JPEG, or TIFF`;
  }
  return buf.subarray(0, 4).toString("latin1") === "LASF"
    ? null
    : "lidar file is not a valid LAS/LAZ point cloud";
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
  { name: "hsi", maxCount: 1 },
  { name: "lidar", maxCount: 1 },
]);

async function persistFiles(assessmentId, files) {
  const dir = path.join(config.storageDir, assessmentId);
  await mkdir(dir, { recursive: true });
  const stored = {};
  for (const [kind, { buffer, originalname }] of Object.entries(files)) {
    const ext = path.extname(originalname).toLowerCase();
    const rel = path.join(assessmentId, `${kind}${ext}`);
    await writeFile(path.join(config.storageDir, rel), buffer);
    stored[kind] = rel;
  }
  return stored;
}

// Phase 5 proof pipeline: content-address the uploads as real IPFS CIDv0 values
// (pinning when Pinata is configured), then log the assessment on-chain (real
// Amoy tx when the chain layer is configured, deterministic simulated tx
// otherwise — the simulated path leaves state at "analyzed").
async function attachProof(assessment, files, prediction) {
  const primaryA = files.pre || files.hsi;
  const primaryB = files.post || files.lidar;

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

// POST /upload — accepts multipart fields (pre + post for xBD, or hsi + lidar for legacy),
// runs the real Siamese U-Net engine, stores evidence, and pins/proves the upload.
router.post("/", requireAuth, (req, res) => {
  upload(req, res, async (err) => {
    try {
      if (err) return multerErrorToResponse(err, res);

      const pre = req.files?.pre?.[0];
      const post = req.files?.post?.[0];
      const hsi = req.files?.hsi?.[0];
      const lidar = req.files?.lidar?.[0];

      const isXbd = Boolean(pre && post);
      const isLegacy = Boolean(hsi && lidar);

      if (!isXbd && !isLegacy) {
        return res.status(400).json({ error: "Please upload pre and post disaster images (or hsi + lidar)" });
      }

      let files = {};
      let filenameObj = {};

      if (isXbd) {
        const preBuf = await readFile(pre.path);
        const postBuf = await readFile(post.path);
        const magicErr = validateMagic(preBuf, "pre") || validateMagic(postBuf, "post");
        if (magicErr) return res.status(400).json({ error: magicErr });

        files = {
          pre: { buffer: preBuf, originalname: pre.originalname },
          post: { buffer: postBuf, originalname: post.originalname },
        };
        filenameObj = { pre: pre.originalname, post: post.originalname };
      } else {
        const hsiBuf = await readFile(hsi.path);
        const lidarBuf = await readFile(lidar.path);
        const magicErr = validateMagic(hsiBuf, "hsi") || validateMagic(lidarBuf, "lidar");
        if (magicErr) return res.status(400).json({ error: magicErr });

        files = {
          hsi: { buffer: hsiBuf, originalname: hsi.originalname },
          lidar: { buffer: lidarBuf, originalname: lidar.originalname },
        };
        filenameObj = { hsi: hsi.originalname, lidar: lidar.originalname };
      }

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