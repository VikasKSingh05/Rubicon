import { pathToFileURL } from "node:url";
import express from "express";

const PORT = process.env.BACKEND_PORT || 4000;
const app = express();

app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "backend" });
});

app.get("/", (_req, res) => {
  res.json({ service: "backend", message: "Rubicon backend stub" });
});

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  app.listen(PORT, () => {
    console.log(`[backend] listening on port ${PORT}`);
  });
}

export default app;
