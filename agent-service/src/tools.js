// Tool-grounded backend tools for the agent.
// Every tool answers strictly from fetched data and returns a bounded text
// `summary` for the LLM plus an `ok` flag for the contract's tool_calls chips.

export const TOOL_DEFINITIONS = [
  {
    name: "get_recent_assessments",
    description:
      "List the most recent damage assessments from the backend API (id, prediction, severity, confidence, date). Use this for counts, zones, or recent-activity questions.",
    input_schema: {
      type: "object",
      properties: {
        severity_filter: {
          type: "string",
          enum: ["none", "moderate", "severe"],
          description: "Optional: only return assessments of this severity.",
        },
        limit: { type: "integer", description: "Max items to return (default 20)." },
      },
    },
  },
  {
    name: "get_assessment",
    description:
      "Fetch the full record for one assessment by id: prediction, confidence, class probabilities, geojson polygon, IPFS CIDs and on-chain status.",
    input_schema: {
      type: "object",
      properties: {
        id: { type: "string", description: "Assessment id (24-char hex)." },
      },
      required: ["id"],
    },
  },
  {
    name: "get_chain_status",
    description:
      "Check whether an IPFS CID was submitted and verified on-chain. Answer verification/proof questions strictly from this tool's result.",
    input_schema: {
      type: "object",
      properties: {
        cid: { type: "string", description: "IPFS CID of an uploaded scan." },
      },
      required: ["cid"],
    },
  },
  {
    name: "summarize_zone",
    description:
      "Compute rough area (km2) and bounding-box statistics for a damaged-zone GeoJSON polygon.",
    input_schema: {
      type: "object",
      properties: {
        geojson: { type: "object", description: "GeoJSON Polygon geometry." },
      },
      required: ["geojson"],
    },
  },
];

export function roughAreaKm2(polygon) {
  const ring = polygon?.coordinates?.[0];
  if (!Array.isArray(ring) || ring.length < 3) return 0;
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
  const midLat = ((minLat + maxLat) / 2) * (Math.PI / 180);
  const widthKm = (maxLng - minLng) * 111.32 * Math.cos(midLat);
  const heightKm = (maxLat - minLat) * 111.32;
  return { area_km2: Math.abs(widthKm * heightKm), bbox: [minLng, minLat, maxLng, maxLat] };
}

export function manifest() {
  return TOOL_DEFINITIONS;
}

async function backendGet(opts, path, label) {
  const { backendUrl, token, fetchImpl = globalThis.fetch } = opts;
  try {
    const res = await fetchImpl(`${backendUrl}${path}`, {
      headers: {
        Authorization: token ? `Bearer ${token}` : "",
        Accept: "application/json",
      },
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return { ok: false, summary: `backend ${label} failed (${res.status}): ${body?.error || ""}`.trim() };
    }
    return { ok: true, data: await res.json() };
  } catch (err) {
    return { ok: false, summary: `backend ${label} unreachable: ${err?.message || "network error"}` };
  }
}

function fmtAssessment(a) {
  const conf = a.confidence !== null && a.confidence !== undefined ? ` conf ${Number(a.confidence).toFixed(2)}` : "";
  const when = a.createdAt ? ` on ${String(a.createdAt).slice(0, 10)}` : "";
  return `${a.severity} "${a.prediction}"${conf} id ${a.id}${when}`;
}

async function getRecentAssessments(input, opts) {
  const limit = Number(input?.limit) || 20;
  const severityFilter = input?.severity_filter;
  const r = await backendGet(opts, "/assessments", "assessments list");
  if (!r.ok) return r;
  const all = Array.isArray(r.data.assessments) ? r.data.assessments : [];
  const filtered = severityFilter
    ? all.filter((a) => String(a.severity) === String(severityFilter))
    : all;
  const items = filtered.slice(0, limit);
  if (items.length === 0) {
    return { ok: true, summary: `No ${severityFilter ? `${severityFilter} ` : ""}assessments found.` };
  }
  return {
    ok: true,
    summary: `${items.length} assessment${items.length === 1 ? "" : "s"}:\n${items
      .map(fmtAssessment)
      .join("\n")}`,
    data: { assessments: items },
  };
}

async function getAssessment(input, opts) {
  const id = String(input?.id || "").trim();
  if (!id) return { ok: false, summary: "get_assessment requires an id." };
  const r = await backendGet(opts, `/assessments/${id}`, "assessment detail");
  if (!r.ok) return r;
  const a = r.data;
  const summary = [
    `Assessment ${a.id}: ${a.severity} "${a.prediction}"`,
    `confidence ${Number(a.confidence).toFixed(3)}`,
    `state ${a.state}`,
    a.hsiCid ? `hsiCid ${a.hsiCid}` : "hsiCid not set",
    `chainVerified ${a.chainVerified}`,
  ].join("\n");
  return { ok: true, summary, data: a };
}

async function getChainStatus(input, opts) {
  const cid = String(input?.cid || "").trim();
  if (!cid) return { ok: false, summary: "get_chain_status requires a cid." };
  const r = await backendGet(opts, `/chain/verify/${encodeURIComponent(cid)}`, "chain status");
  if (!r.ok) return r;
  const { verified, txHash, assessmentId, state } = r.data;
  const summary =
    `CID ${cid}: verified on-chain = ${verified}` +
    (assessmentId ? ` (assessment ${assessmentId}, state ${state}, tx ${String(txHash).slice(0, 12)})` : ", no assessment links this CID");
  return { ok: true, summary, data: r.data };
}

async function summarizeZone(input) {
  const { area_km2, bbox } = roughAreaKm2(input?.geojson);
  const summary = `Zone polygon: ~${area_km2.toFixed(3)} km2, bbox [lng ${bbox[0].toFixed(4)}, lat ${bbox[1].toFixed(4)}] .. [lng ${bbox[2].toFixed(4)}, lat ${bbox[3].toFixed(4)}].`;
  return { ok: true, summary, data: { area_km2, bbox } };
}

const EXECUTORS = {
  get_recent_assessments: getRecentAssessments,
  get_assessment: getAssessment,
  get_chain_status: getChainStatus,
  summarize_zone: summarizeZone,
};

// Runs one tool. opts: { backendUrl, token, fetch }.
export async function execTool(name, input, opts) {
  const executor = EXECUTORS[name];
  if (!executor) return { ok: false, summary: `unknown tool: ${name}` };
  try {
    return await executor(input || {}, opts);
  } catch (err) {
    return { ok: false, summary: `tool ${name} threw: ${err?.message || "error"}` };
  }
}