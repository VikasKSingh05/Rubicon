import { pathToFileURL } from "node:url";
import express from "express";
import cors from "cors";

import { config, llmConfigured } from "./config.js";
import { runAgent } from "./providers.js";
import { runFallbackAgent } from "./fallback.js";
import { manifest } from "./tools.js";

const PORT = config.port;
const app = express();

app.use(cors());
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "agent-service",
    llm: llmConfigured() ? config.llm.provider : "fallback",
  });
});

app.get("/", (_req, res) => {
  res.json({ service: "agent-service", message: "Rubicon agent-service (tool-grounded)" });
});

app.get("/tools", (_req, res) => {
  res.json({ tools: manifest().map((t) => ({ name: t.name, description: t.description })) });
});

// POST /agent/query — contract: { answer, tool_calls: [{tool, input, ok}], createdAt }.
// Grounding rule: answer is composed strictly from tool results.
app.post("/agent/query", async (req, res) => {
  const body = req.body || {};
  const query = typeof body.query === "string" ? body.query.trim() : "";
  if (!query) return res.status(400).json({ error: "query is required" });
  if (query.length > 2000) return res.status(400).json({ error: "query too long (max 2000 chars)" });

  const opts = {
    query,
    assessmentId: typeof body.assessmentId === "string" ? body.assessmentId : null,
    token: typeof body.token === "string" ? body.token : null,
    backendUrl: config.backendUrl,
  };

  try {
    const { answer, toolCalls } = await runAgent(opts);
    return res.json({ answer, tool_calls: toolCalls, createdAt: new Date().toISOString() });
  } catch (err) {
    console.warn(`[agent] LLM unavailable, using fallback agent: ${err.message}`);
  }

  const fallback = await runFallbackAgent(opts);
  return res.json({ answer: fallback.answer, tool_calls: fallback.toolCalls, createdAt: new Date().toISOString() });
});

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  app.listen(PORT, () => {
    console.log(`[agent-service] listening on port ${PORT}`);
  });
}

export default app;