import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PORT = process.env.FRONTEND_PORT || 3000;
const app = express();
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(__dirname, "dist");

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "frontend" });
});

app.use(express.static(dist));
app.get("*", (_req, res) => {
  res.sendFile(path.join(dist, "index.html"));
});

app.listen(PORT, () => {
  console.log(`[frontend] serving on port ${PORT}`);
});