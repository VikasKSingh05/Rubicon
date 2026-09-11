export const config = {
  nodeEnv: process.env.NODE_ENV || "development",
  isProd: process.env.NODE_ENV === "production",
  port: Number(process.env.BACKEND_PORT) || 4000,
  mongoUri: process.env.MONGO_URI || "mongodb://localhost:27017/rubicon",
  jwtSecret: process.env.JWT_SECRET || "dev-secret-change-me",
  accessTokenTtl: process.env.ACCESS_TOKEN_TTL || "15m",
  aiEngineUrl: process.env.AI_ENGINE_URL || "http://localhost:8000",
  maxUploadMb: Number(process.env.MAX_UPLOAD_MB) || 100,
  storageDir: process.env.STORAGE_DIR || "storage",
  corsOrigins: (process.env.CORS_ORIGINS || "http://localhost:3000")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
  rateLimitEnabled: process.env.RATE_LIMIT_DISABLED !== "1",
  // ---- Phase 5: IPFS + Polygon Amoy ----
  pinataJwt: process.env.PINATA_JWT || "",
  amoyRpcUrl: process.env.AMOY_RPC_URL || "",
  deployerPrivateKey: process.env.DEPLOYER_PRIVATE_KEY || "",
  contractAddress: process.env.CONTRACT_ADDRESS || "",
  chainId: Number(process.env.CHAIN_ID) || 80002,
};

const DEFAULT_SECRET = "dev-secret-change-me";

/**
 * Refuse to boot on an insecure secret outside development. Call from the
 * server entrypoint (not tests) so local dev with the default still works.
 */
export function validateProductionSecrets() {
  if (config.isProd) {
    const problems = [];
    if (!process.env.JWT_SECRET || config.jwtSecret === DEFAULT_SECRET) {
      problems.push("JWT_SECRET must be set to a strong secret in production");
    }
    if (problems.length > 0) throw new Error(problems.join("; "));
  }
}