import express from "express";
import cors from "cors";
import authRoutes from "./routes/auth.js";
import uploadRoutes from "./routes/upload.js";
import assessmentRoutes from "./routes/assessments.js";
import chainRoutes from "./routes/chain.js";

export default function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json());

  app.get("/health", (_req, res) => {
    res.json({ status: "ok", service: "backend" });
  });

  app.get("/", (_req, res) => {
    res.json({ service: "backend", message: "Rubicon backend" });
  });

  app.use("/auth", authRoutes);
  app.use("/upload", uploadRoutes);
  app.use("/assessments", assessmentRoutes);
  app.use("/chain", chainRoutes);

  return app;
}