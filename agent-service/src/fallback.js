// Deterministic, tool-grounded fallback agent. Runs when no LLM key is
// configured or the provider errors — answers strictly from backend tool
// results so the grounding guarantee holds without any external API.

import { execTool } from "./tools.js";

function refusal(what) {
  return (
    `I couldn't retrieve ${what} from the Rubicon backend (it may be unreachable, ` +
    "or this session isn't authenticated), so I don't have data to answer that."
  );
}

function severityBreakdown(assessments) {
  const counts = { none: 0, moderate: 0, severe: 0 };
  for (const a of assessments) {
    const key = ["none", "moderate", "severe"].includes(a.severity) ? a.severity : "none";
    counts[key] += 1;
  }
  return counts;
}

function shortId(id) {
  return String(id || "").slice(0, 8) + "\u2026";
}

// opts: { query, assessmentId, token, backendUrl, fetch }
export async function runFallbackAgent(opts) {
  const query = String(opts.query || "").trim();
  const q = query.toLowerCase();
  const toolOpts = { backendUrl: opts.backendUrl, token: opts.token, fetchImpl: opts.fetch };
  const toolCalls = [];
  const run = async (name, input) => {
    const r = await execTool(name, input, toolOpts);
    toolCalls.push({ tool: name, input, ok: r.ok });
    return r;
  };

  // 1) On-chain verification intent.
  if (/verified|verif|chain|on-?chain|proof|logged|cid|notarized|immutable/.test(q)) {
    const list = await run("get_recent_assessments", { limit: 20 });
    if (!list.ok) return { answer: refusal("assessment records"), toolCalls };
    const newest = list.data.assessments[0];
    if (!newest) return { answer: "I have no assessments yet, so there is nothing to verify on-chain.", toolCalls };

    const detail = await run("get_assessment", { id: newest.id });
    const cid = detail.ok && detail.data.hsiCid;
    if (!cid) {
      return {
        answer: `The most recent assessment (${shortId(newest.id)}) has no IPFS CID recorded yet.`,
        toolCalls,
      };
    }
    const chain = await run("get_chain_status", { cid });
    if (!chain.ok) return { answer: refusal("on-chain status"), toolCalls };
    const { verified, txHash, assessmentId } = chain.data;
    return {
      answer:
        `The latest assessment (${shortId(assessmentId || newest.id)}) is ` +
        `${verified ? "verified on-chain" : "NOT yet verified on-chain"}` +
        `${txHash ? ` (tx ${String(txHash).slice(0, 18)}\u2026)` : " (no transaction logged yet)"}.`,
      toolCalls,
    };
  }

  // 2) Severe-zone intent.
  if (/severe|collapse|critical|zone|zones|damaged area|hotspot/.test(q)) {
    const list = await run("get_recent_assessments", { severity_filter: "severe", limit: 20 });
    if (!list.ok) return { answer: refusal("severe-zone records"), toolCalls };
    const severe = list.data.assessments;
    if (severe.length === 0) {
      return { answer: "No severe zones are present in the current assessments.", toolCalls };
    }
    const parts = severe.map((a) => `${shortId(a.id)} (${a.prediction}, conf ${Number(a.confidence).toFixed(2)})`);
    return {
      answer:
        `There ${severe.length === 1 ? "is" : "are"} ${severe.length} severe zone${severe.length === 1 ? "" : "s"}: ` +
        parts.join("; ") + ".",
      toolCalls,
    };
  }

  // 3) Counts / summary intent.
  if (/how many|count|total|summary|overview|stats|breakdown|situation/.test(q)) {
    const list = await run("get_recent_assessments", { limit: 200 });
    if (!list.ok) return { answer: refusal("assessment records"), toolCalls };
    const all = list.data.assessments;
    if (all.length === 0) return { answer: "There are no assessments yet.", toolCalls };
    const counts = severityBreakdown(all);
    const lines = [
      `I have ${all.length} assessment${all.length === 1 ? "" : "s"} on file.`,
      `Severe: ${counts.severe}, Moderate: ${counts.moderate}, None/Minimal: ${counts.none}.`,
    ];
    return { answer: lines.join(" "), toolCalls };
  }

  // 4) Single-assessment / "last upload" intent.
  if (/assessmentId|about this|this assessment/.test(q) || opts.assessmentId) {
    const id = opts.assessmentId;
    const detail = id ? await run("get_assessment", { id }) : await run("get_recent_assessments", { limit: 1 });
    if (!detail.ok) return { answer: refusal("the assessment"), toolCalls };
    const a = detail.data?.assessments?.[0] || detail.data;
    if (!a) return { answer: "I don't have that assessment.", toolCalls };
    return {
      answer:
        `Assessment ${a.id}: ${a.severity} "${a.prediction}" with confidence ${Number(a.confidence).toFixed(3)} ` +
        `(state ${a.state}, chain verified ${a.chainVerified}).`,
      toolCalls,
    };
  }

  // 5) Latest-activity intent.
  if (/last|latest|recent|newest|today|most recent/.test(q)) {
    const list = await run("get_recent_assessments", { limit: 3 });
    if (!list.ok) return { answer: refusal("recent assessments"), toolCalls };
    const items = list.data.assessments;
    if (items.length === 0) return { answer: "No assessments have been uploaded yet.", toolCalls };
    return {
      answer:
        `Most recent: ${items.map((a) => `${a.prediction} (${shortId(a.id)})`).join(", ")}. ` +
        "State status comes from the backend assessment records.",
      toolCalls,
    };
  }

  // 6) Default overview.
  const list = await run("get_recent_assessments", { limit: 20 });
  if (!list.ok) return { answer: refusal("assessment records"), toolCalls };
  const items = list.data.assessments;
  if (items.length === 0) {
    return { answer: "I have no assessment records yet. Upload a scan and ask again.", toolCalls };
  }
  const counts = severityBreakdown(items);
  return {
    answer:
      `Here's the current picture: ${items.length} assessment${items.length === 1 ? "" : "s"} — ` +
      `severe ${counts.severe}, moderate ${counts.moderate}, none ${counts.none}. ` +
      "I can break this down by zone or check on-chain status if you ask.",
    toolCalls,
  };
}