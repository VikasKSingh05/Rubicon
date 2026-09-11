import express from "express";
import cors from "cors";
import helmet from "helmet";
import cookieParser from "cookie-parser";
import rateLimit from "express-rate-limit";
import mongoose from "mongoose";
import authRoutes from "./routes/auth.js";
import uploadRoutes from "./routes/upload.js";
import assessmentRoutes from "./routes/assessments.js";
import chainRoutes from "./routes/chain.js";
import { config } from "./config.js";
import { makeLogger, requestId, requestLogger } from "./middleware/logger.js";

const logger = makeLogger("backend");

const standardHandler = (_req, res) =>
  res.status(429).json({ error: "too many requests" });

export default function createApp() {
  const app = express();
  app.disable("x-powered-by");
  app.use(requestId);
  app.use(requestLogger(logger));
  app.use(
    helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https://*.tile.openstreetmap.org"],
          connectSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          fontSrc: ["'self'", "data:"],
        },
      },
    }),
  );
  app.use(cors({ origin: config.corsOrigins }));
  app.use(express.json());
  app.use(cookieParser());

  if (config.rateLimitEnabled) {
    app.use(
      "/auth",
      rateLimit({ windowMs: 15 * 60 * 1000, limit: 10, standardHeaders: true, legacyHeaders: false, handler: standardHandler }),
    );
    app.use(
      "/upload",
      rateLimit({ windowMs: 15 * 60 * 1000, limit: 20, standardHeaders: true, legacyHeaders: false, handler: standardHandler }),
    );
  }

  app.get("/health", (_req, res) => {
    const mongoOk = mongoose.connection.readyState === 1;
    res.status(mongoOk ? 200 : 503).json({
      status: mongoOk ? "ok" : "degraded",
      service: "backend",
      mongo: mongoOk ? "connected" : "disconnected",
    });
  });

  // Basic startup-time counters for ops dashboards (no deps).
  const metrics = { startedAt: Date.now(), uploads: 0, proofs: 0, errors: 0 };
  app.locals.metrics = metrics;
  app.get("/metrics", (_req, res) => {
    res.type("text/plain").end([
      "# HELP process_uptime_seconds App uptime.",
      "# TYPE process_uptime_seconds gauge",
      `process_uptime_seconds ${(Date.now() - metrics.startedAt) / 1000}`,
      "# HELP rubicon_uploads_total Uploads accepted.",
      "# TYPE rubicon_uploads_total counter",
      `rubicon_uploads_total ${metrics.uploads}`,
      "# HELP rubicon_proofs_total Proof pipelines completed.",
      "# TYPE rubicon_proofs_total counter",
      `rubicon_proofs_total ${metrics.proofs}`,
      "# HELP rubicon_errors_total Internal errors.",
      "# TYPE rubicon_errors_total counter",
      `rubicon_errors_total ${metrics.errors}`,
    ].join("\n"));
  });

  app.get("/", (_req, res) => {
    res.json({ service: "backend", message: "Rubicon backend" });
  });

  app.use("/auth", authRoutes);
  app.use("/upload", uploadRoutes);
  app.use("/assessments", assessmentRoutes);
  app.use("/chain", chainRoutes);

  // 404 for unknown routes.
  app.use((_req, res) => {
    res.status(404).json({ error: "not found" });
  });

  // Central error handler — asyncHandler-wrapped routes land here.
  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, next) => {
    if (res.headersSent) return next(err);
    metrics.errors += 1;
    logger.error("unhandled error", {
      requestId: req.requestId,
      message: err?.message || String(err),
    });
    const status = err?.status || err?.statusCode || 500;
    return res.status(status).json({ error: status >= 500 ? "internal server error" : err?.message || "request failed" });
  });

  return app;
}