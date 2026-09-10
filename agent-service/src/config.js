export const config = {
  port: Number(process.env.AGENT_SERVICE_PORT) || 8001,
  backendUrl: process.env.BACKEND_URL || "http://localhost:4000",
  llm: {
    provider: String(process.env.LLM_PROVIDER || "").toLowerCase(),
    anthropicModel: process.env.AGENT_MODEL || "claude-sonnet-4-5",
    openaiModel: process.env.OPENAI_MODEL || "gpt-4o",
    // OpenRouter is OpenAI-compatible; model is any OpenRouter slug, e.g.
    // "anthropic/claude-3.5-sonnet" or "openai/gpt-4o".
    openrouterModel: process.env.OPENROUTER_MODEL || "anthropic/claude-3.5-sonnet",
  },
};

export function llmConfigured() {
  if (!config.llm.provider) return false;
  if (config.llm.provider === "anthropic") return Boolean(process.env.ANTHROPIC_API_KEY);
  if (config.llm.provider === "openai") return Boolean(process.env.OPENAI_API_KEY);
  // OpenRouter accepts either its own key or (for convenience) the OpenAI one.
  if (config.llm.provider === "openrouter")
    return Boolean(process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY);
  return false;
}