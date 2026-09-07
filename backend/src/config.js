export const config = {
  port: Number(process.env.BACKEND_PORT) || 4000,
  mongoUri: process.env.MONGO_URI || "mongodb://localhost:27017/rubicon",
  jwtSecret: process.env.JWT_SECRET || "dev-secret-change-me",
  aiEngineUrl: process.env.AI_ENGINE_URL || "http://localhost:8000",
  maxUploadMb: 100,
};