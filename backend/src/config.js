export const config = {
  port: Number(process.env.BACKEND_PORT) || 4000,
  mongoUri: process.env.MONGO_URI || "mongodb://localhost:27017/rubicon",
  jwtSecret: process.env.JWT_SECRET || "dev-secret-change-me",
  aiEngineUrl: process.env.AI_ENGINE_URL || "http://localhost:8000",
  maxUploadMb: 100,
  // ---- Phase 5: IPFS + Polygon Amoy ----
  pinataJwt: process.env.PINATA_JWT || "",
  amoyRpcUrl: process.env.AMOY_RPC_URL || "",
  deployerPrivateKey: process.env.DEPLOYER_PRIVATE_KEY || "",
  contractAddress: process.env.CONTRACT_ADDRESS || "",
  chainId: Number(process.env.CHAIN_ID) || 80002,
};