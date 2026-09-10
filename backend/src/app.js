import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import authRoutes from "./routes/auth.js";
import uploadRoutes from "./routes/upload.js";
import assessmentRoutes from "./routes/assessments.js";
import chainRoutes from "./routes/chain.js";

export default function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/health", (_req, res) => {
    const mongoOk = mongoose.connection.readyState === 1;
    res.status(mongoOk ? 200 : 503).json({
      status: mongoOk ? "ok" : "degraded",
      service: "backend",
      mongo: mongoOk ? "connected" : "disconnected",
    });
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
  app.use((err, _req, res, next) => {
    if (res.headersSent) return next(err);
    console.error("[backend] unhandled error:", err?.message || err);
    const status = err?.status || err?.statusCode || 500;
    return res.status(status).json({ error: status >= 500 ? "internal server error" : err?.message || "request failed" });
  });

  return app;
}