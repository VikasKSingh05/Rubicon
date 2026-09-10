// Tool-calling agent loops for Anthropic and OpenAI. Both answer ONLY from
// tool results; if the provider errors (bad key, model, network) we raise and
// the orchestrator degrades to the deterministic fallback agent.

import { config, llmConfigured } from "./config.js";
import { execTool, manifest } from "./tools.js";

const ITERATION_LIMIT = 5;
const TIMEOUT_MS = 20_000;
const SYSTEM_PROMPT = [
  "You are the Rubicon disaster-assessment assistant.",
  "You answer ONLY from the results of the provided tools — never from your own knowledge.",
  "Rules:",
  "- If a tool call fails or returns no data, say so plainly. Never invent ids, numbers, or on-chain status.",
  "- Answer in clear, concise plain language. When reporting counts, show the breakdown by severity.",
  "- Use get_recent_assessments for anything about counts/zones/recent activity.",
  "- Use get_assessment for a single assessment's details (confidence, chain status).",
  "- Use get_chain_status only when the question is about on-chain verification or proof.",
].join("\n");

async function postJson(url, headers, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(`${data?.error?.message || res.status} ${data?.error?.type || ""}`.trim());
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

// ---- Anthropic ------------------------------------------------------------

async function anthropicCall(model, messages) {
  const data = await postJson(
    "https://api.anthropic.com/v1/messages",
    {
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    {
      model,
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      tools: manifest().map((t) => ({ name: t.name, description: t.description, input_schema: t.input_schema })),
      messages,
    },
  );
  const text = data.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  const toolUses = data.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ id: b.id, name: b.name, input: b.input }));
  return { text, toolUses, assistantMessage: { role: "assistant", content: data.content } };
}

function anthropicAppendResults(messages, assistantMessage, toolUses, results) {
  const next = [...messages, assistantMessage];
  toolUses.forEach((tu, i) => {
    next.push({
      role: "user",
      content: [{ type: "tool_result", tool_use_id: tu.id, content: results[i].summary }],
    });
  });
  return next;
}

// ---- OpenAI / OpenRouter (OpenAI-compatible) -----------------------------------

function bearer(keyNames) {
  for (const name of keyNames) {
    if (process.env[name]) return process.env[name];
  }
  throw new Error(`missing API key (tried ${keyNames.join(", ")})`);
}

async function openaiCompatibleCall({ url, model, keyNames, messages }) {
  const data = await postJson(
    url,
    { Authorization: `Bearer ${bearer(keyNames)}`, "content-type": "application/json" },
    {
      model,
      messages,
      tools: manifest().map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.input_schema },
      })),
    },
  );
  const message = data.choices?.[0]?.message || {};
  const toolCalls = message.tool_calls || [];
  const toolUses = toolCalls.map((tc) => {
    let input = {};
    try {
      input = JSON.parse(tc.function.arguments || "{}");
    } catch {
      input = {};
    }
    return { id: tc.id, name: tc.function.name, input };
  });
  return {
    text: message.content || "",
    toolUses,
    assistantMessage: { role: "assistant", content: message.content || "", tool_calls: toolCalls },
  };
}

function openaiCall(model, messages) {
  return openaiCompatibleCall({
    url: "https://api.openai.com/v1/chat/completions",
    model,
    keyNames: ["OPENAI_API_KEY"],
    messages,
  });
}

function openrouterCall(model, messages) {
  return openaiCompatibleCall({
    url: "https://openrouter.ai/api/v1/chat/completions",
    model,
    keyNames: ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
    messages,
  });
}

function openaiAppendResults(messages, assistantMessage, toolUses, results) {
  const next = [...messages, assistantMessage];
  toolUses.forEach((tu, i) => {
    next.push({ role: "tool", tool_call_id: tu.id, content: results[i].summary });
  });
  return next;
}

const PROVIDERS = {
  anthropic: {
    call: (messages) => anthropicCall(config.llm.anthropicModel, messages),
    appendResults: anthropicAppendResults,
  },
  openai: {
    call: (messages) => openaiCall(config.llm.openaiModel, messages),
    appendResults: openaiAppendResults,
  },
  openrouter: {
    call: (messages) => openrouterCall(config.llm.openrouterModel, messages),
    appendResults: openaiAppendResults,
  },
};

// Runs the tool-calling loop. opts: { query, assessmentId, token, backendUrl, fetch }.
export async function runAgent(opts) {
  if (!llmConfigured()) throw new Error("no LLM provider configured");

  const provider = PROVIDERS[config.llm.provider];
  if (!provider) throw new Error(`unknown LLM_PROVIDER: ${config.llm.provider}`);

  const toolOpts = { backendUrl: opts.backendUrl, token: opts.token, fetchImpl: opts.fetch };
  let messages = [{ role: "user", content: opts.query }];
  const toolCalls = [];
  const textParts = [];

  for (let i = 0; i < ITERATION_LIMIT; i += 1) {
    const { text, toolUses, assistantMessage } = await provider.call(messages);
    if (text) textParts.push(text);

    if (!toolUses?.length) break;

    const results = [];
    for (const tu of toolUses) {
      const r = await execTool(tu.name, tu.input, toolOpts);
      toolCalls.push({ tool: tu.name, input: tu.input, ok: r.ok });
      results.push(r);
    }
    messages = provider.appendResults(messages, assistantMessage, toolUses, results);
  }

  const answer = textParts.join("\n").trim() || "I don't have an answer from my tools yet.";
  return { answer, toolCalls };
}