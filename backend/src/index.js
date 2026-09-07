import { pathToFileURL } from "node:url";
import "dotenv/config";
import createApp from "./app.js";
import { connectDb } from "./db.js";
import { config } from "./config.js";

const app = createApp();

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  await connectDb(config.mongoUri);
  app.listen(config.port, () => {
    console.log(`[backend] listening on port ${config.port}`);
  });
}

export default app;