import { ASSESSMENT_STATES } from "../models/Assessment.js";

const TIMESTAMP_KEY = {
  uploaded: "uploaded",
  analyzing: "analyzing",
  analyzed: "analyzed",
  chain_pending: "chainPending",
  chain_logged: "chainLogged",
};

/**
 * Advance an assessment through the state machine:
 * uploaded -> analyzing -> analyzed -> chain_pending -> chain_logged.
 * Each transition stamps its corresponding timestamp.
 */
export function transition(assessment, to) {
  const from = ASSESSMENT_STATES.indexOf(assessment.state);
  const next = ASSESSMENT_STATES.indexOf(to);
  if (next < 0 || next > ASSESSMENT_STATES.length - 1) throw new Error(`unknown state: ${to}`);
  if (next < from) throw new Error(`cannot transition ${assessment.state} -> ${to}`);
  assessment.state = to;
  assessment.timestamps[TIMESTAMP_KEY[to]] = new Date();
  return assessment;
}