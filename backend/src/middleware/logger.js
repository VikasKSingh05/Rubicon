// Structured JSON logger + per-request correlation IDs. Never logs secrets,
// tokens, passwords, or request bodies of untrusted payloads.

import crypto from "node:crypto";
import { performance } from "node:perf_hooks";

function write(level, service, fields) {
  const line = {
    level,
    time: new Date().toISOString(),
    service,
    ...fields,
  };
  const str = JSON.stringify(line);
  if (level === "error") process.stderr.write(`${str}\n`);
  else process.stdout.write(`${str}\n`);
}

export function makeLogger(service) {
  return {
    info: (msg, fields = {}) => write("info", service, { msg, ...fields }),
    warn: (msg, fields = {}) => write("warn", service, { msg, ...fields }),
    error: (msg, fields = {}) => write("error", service, { msg, ...fields }),
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