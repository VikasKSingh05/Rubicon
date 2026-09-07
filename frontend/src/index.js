import { pathToFileURL } from "node:url";
import express from "express";

const PORT = process.env.FRONTEND_PORT || 3000;
const app = express();

app.use(express.static("public"));

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "frontend" });
});

app.get("/", (_req, res) => {
  res.json({ service: "frontend", message: "Rubicon frontend stub" });
});

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  app.listen(PORT, () => {
    console.log(`[frontend] listening on port ${PORT}`);
  });
}

export default app;
