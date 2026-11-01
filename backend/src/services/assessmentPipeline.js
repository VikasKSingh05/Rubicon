// Async xBD assessment pipeline: persisted pre/post pair -> inference -> proof.
// Runs inside the in-process assessment queue. Every failure is written to the
// Assessment document (state="error") so the client can poll and retry.
import { readFile } from "node:fs/promises";
import Assessment from "../models/Assessment.js";
import { runInference } from "./aiService.js";
import { transition } from "./assessmentService.js";
import { attachProof } from "./proofService.js";
import { resolveStoredPath } from "./fileStorage.js";
import { makeLogger } from "../middleware/logger.js";

const logger = makeLogger("assessment-pipeline");

function hasCells(cells) {
  return Boolean(cells && Array.isArray(cells.features) && cells.features.length > 0);
}

export async function processSatelliteAssessment({ assessmentId, requestId, metrics } = {}) {
  const assessment = await Assessment.findById(assessmentId);
  if (!assessment) {
    logger.warn("assessment disappeared before processing", { assessmentId });
    return;
  }

  try {
    transition(assessment, "analyzing");
    await assessment.save();

    const files = {};
    for (const kind of ["pre", "post"]) {
      const rel = assessment.storage?.[kind];
      if (!rel) throw new Error(`missing stored ${kind} image`);
      files[kind] = {
        buffer: await readFile(resolveStoredPath(rel)),
        originalname: assessment.filename?.[kind] || `${kind}.png`,
      };
    }

    const result = await runInference(files, { requestId });

    assessment.prediction = result.prediction;
    assessment.confidence = result.confidence;
    assessment.class_probs = result.class_probs;
    assessment.geojson_polygon = result.geojson_polygon;
    assessment.buildings_geojson = result.buildings_geojson;
    assessment.model_version = result.model_version;

    const cellsR9 = result.cells_geojson_r9 ?? null;
    const cellsR10 = result.cells_geojson_r10 ?? null;
    assessment.cells_geojson = { r9: cellsR9, r10: cellsR10 };
    assessment.cells_source = hasCells(cellsR9) || hasCells(cellsR10) ? "engine" : null;

    transition(assessment, "analyzed");
    await attachProof(assessment, files, result.prediction);

    assessment.error = null;
    await assessment.save();
    if (metrics) metrics.uploads += 1;

    logger.info("assessment complete", {
      requestId,
      assessmentId,
      state: assessment.state,
      modelVersion: result.model_version,
      engineCells: assessment.cells_source === "engine",
    });
  } catch (error) {
    if (metrics) metrics.errors += 1;
    logger.error("assessment failed", {
      requestId,
      assessmentId,
      message: error?.message || String(error),
    });
    assessment.state = "error";
    assessment.error = error?.message || String(error);
    await assessment.save().catch(() => {});
  }
}
