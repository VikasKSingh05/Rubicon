import { Router } from "express";
import path from "node:path";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { requireAuth } from "../middleware/auth.js";
import Assessment from "../models/Assessment.js";
import { config } from "../config.js";

const KINDS = new Set(["hsi", "lidar"]);

const router = Router();

function kindToType(kind) {
  return kind === "hsi" ? { mime: "image/tiff", label: "HSI" } : { mime: "application/octet-stream", label: "LiDAR" };
}

// GET /files/:id/:kind — stream the persisted original. Path-traversal safe:
// the stored path comes from the DB, and the resolved absolute path must stay
// inside STORAGE_DIR before we open it.
router.get("/:id/:kind", requireAuth, async (req, res) => {
  const { id, kind } = req.params;
  if (!KINDS.has(kind)) return res.status(400).json({ error: "kind must be 'hsi' or 'lidar'" });

  const assessment = await Assessment.findById(id);
  if (!assessment) return res.status(404).json({ error: "assessment not found" });
  if (assessment.user.toString() !== req.user.sub) return res.status(404).json({ error: "assessment not found" });

  const rel = assessment.storage?.[kind];
  if (!rel) return res.status(404).json({ error: "file not found" });

  const abs = path.resolve(config.storageDir, rel);
  if (!abs.startsWith(path.resolve(config.storageDir) + path.sep)) {
    return res.status(404).json({ error: "file not found" });
  }
  try {
    const info = await stat(abs);
    if (!info.isFile()) return res.status(404).json({ error: "file not found" });
  } catch {
    return res.status(404).json({ error: "file not found" });
  }

  const { mime, label } = kindToType(kind);
  const original = assessment.filename?.[kind];
  const download = original ?? `${label} data`;
  res.type(mime).setHeader("Content-Disposition", `attachment; filename="${download.replace(/"/g, "")}"`);
  createReadStream(abs).on("error", () => res.destroy()).pipe(res);
});

export default router;