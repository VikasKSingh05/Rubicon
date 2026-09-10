import { getToken } from "./api.js";

const AGENT_BASE = import.meta.env.VITE_AGENT_URL || "http://localhost:8001";

export async function agentQuery(query, { assessmentId } = {}) {
  const token = getToken();
  const res = await fetch(`${AGENT_BASE}/agent/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, assessmentId, token }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(data?.error || `agent request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data; // { answer, tool_calls, createdAt }
}