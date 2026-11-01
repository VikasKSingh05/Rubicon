// The single assessment queue for this process. Jobs are lost on restart and
// the Assessment document is left in its last state (see docs/decisions.md #4).
import { createJobQueue } from "./jobQueue.js";
import { makeLogger } from "../middleware/logger.js";

export const assessmentQueue = createJobQueue({
  concurrency: 1,
  logger: makeLogger("assessment-queue"),
});
