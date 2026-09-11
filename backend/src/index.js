import { pathToFileURL } from "node:url";
import "dotenv/config";
import createApp from "./app.js";
import { connectDb, disconnectDb } from "./db.js";
import { config, validateProductionSecrets } from "./config.js";
import { makeLogger } from "./middleware/logger.js";

const logger = makeLogger("backend");

/**
 * Start the HTTP server and wire graceful shutdown so in-flight connections
 * drain before we drop Mongo. Extracted for a testable shutdown path.
 */
export async function startServer({ port = config.port, mongoUri = config.mongoUri } = {}) {
  validateProductionSecrets();
  await connectDb(mongoUri);
  const app = createApp();
  const server = await new Promise((resolve) => {
    const s = app.listen(port, () => {
      logger.info("listening", { port });
      resolve(s);
    });
  });

  let shuttingDown = false;
  const shutdown = async (signal) => {
    if (shuttingDown) return;
    shuttingDown = true;
    logger.info("shutting down", { signal });
    await new Promise((resolve) => server.close(resolve));
    await disconnectDb();
    logger.info("shutdown complete", { signal });
  };
  process.on("SIGTERM", () => {
    shutdown("SIGTERM").finally(() => process.exit(0));
  });
  process.on("SIGINT", () => {
    shutdown("SIGINT").finally(() => process.exit(0));
  });

  return { server, shutdown };
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  await startServer();
}

export default { startServer };