import { pathToFileURL } from "node:url";
import "dotenv/config";
import createApp from "./app.js";
import { connectDb, disconnectDb } from "./db.js";
import { config } from "./config.js";

const app = createApp();

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  await connectDb(config.mongoUri);
  const server = app.listen(config.port, () => {
    console.log(`[backend] listening on port ${config.port}`);
  });

  const shutdown = async (signal) => {
    console.log(`[backend] ${signal} received — shutting down`);
    server.close(async () => {
      await disconnectDb();
      process.exit(0);
    });
    // Safety net if connections never drain.
    setTimeout(() => process.exit(1), 10_000).unref();
  };
  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT", () => shutdown("SIGINT"));
}

export default app;