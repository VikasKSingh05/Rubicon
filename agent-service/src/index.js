import { pathToFileURL } from "node:url";
import express from "express";
import cors from "cors";
import helmet from "helmet";

import { config, llmConfigured } from "./config.js";
import { runAgent } from "./providers.js";
import { runFallbackAgent } from "./fallback.js";
import { manifest } from "./tools.js";
import { makeLogger, requestId, requestLogger } from "./logger.js";

const logger = makeLogger("agent-service");

const PORT = config.port;
const app = express();

app.disable("x-powered-by");
app.use(requestId);
app.use(requestLogger(logger));
app.use(helmet());
app.use(cors({ origin: (process.env.CORS_ORIGINS || "http://localhost:3000").split(",").map((s) => s.trim()).filter(Boolean) }));
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

  const started = Date.now();
  try {
    const { answer, toolCalls } = await runAgent(opts);
    logger.info("query answered by LLM", {
      requestId: req.requestId,
      durationMs: Date.now() - started,
      toolCalls: toolCalls.length,
    });
    return res.json({ answer, tool_calls: toolCalls, createdAt: new Date().toISOString() });
  } catch (err) {
    logger.warn("LLM unavailable, using fallback agent", {
      requestId: req.requestId,
      message: err.message,
    });
  }

  const fallback = await runFallbackAgent(opts);
  logger.info("query answered by fallback agent", {
    requestId: req.requestId,
    durationMs: Date.now() - started,
    toolCalls: fallback.toolCalls.length,
  });
  return res.json({ answer: fallback.answer, tool_calls: fallback.toolCalls, createdAt: new Date().toISOString() });
});

app.use((_req, res) => {
  res.status(404).json({ error: "not found" });
});

app.use((err, _req, res, next) => {
  if (res.headersSent) return next(err);
  logger.error("unhandled error", { message: err?.message || String(err) });
  return res.status(500).json({ error: "internal server error" });
});

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  app.listen(PORT, () => {
    logger.info("listening", { port: PORT });
  });
}

export default app;