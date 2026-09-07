import { pathToFileURL } from "node:url";
import express from "express";

const PORT = process.env.AGENT_SERVICE_PORT || 8001;
const app = express();

app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "agent-service" });
});

app.get("/", (_req, res) => {
  res.json({ service: "agent-service", message: "Rubicon agent-service stub" });
});

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  app.listen(PORT, () => {
    console.log(`[agent-service] listening on port ${PORT}`);
  });
}

export default app;
