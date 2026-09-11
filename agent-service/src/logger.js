// Structured JSON logger + per-request correlation IDs for agent-service.
// Request bodies (especially the relayed token) are never logged.

import crypto from "node:crypto";
import { performance } from "node:perf_hooks";

export function makeLogger(service) {
  const write = (level, fields) => {
    const str = JSON.stringify({ level, time: new Date().toISOString(), service, ...fields });
    if (level === "error") process.stderr.write(`${str}\n`);
    else process.stdout.write(`${str}\n`);
  };
  return {
    info: (msg, fields = {}) => write("info", { msg, ...fields }),
    warn: (msg, fields = {}) => write("warn", { msg, ...fields }),
    error: (msg, fields = {}) => write("error", { msg, ...fields }),
  };
}

export function requestId(req, _res, next) {
  req.requestId = req.headers["x-request-id"] || crypto.randomUUID();
  next();
}

export function requestLogger(logger) {
  return (req, res, next) => {
    const start = performance.now();
    res.on("finish", () => {
      logger.info("http", {
        requestId: req.requestId,
        method: req.method,
        path: req.baseUrl + req.path,
        status: res.statusCode,
        durationMs: Math.round((performance.now() - start) * 100) / 100,
      });
    });
    next();
  };
}