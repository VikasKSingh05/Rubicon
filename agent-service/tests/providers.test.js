import test from "node:test";
import assert from "node:assert/strict";

// Tests the LLM tool-calling loop against the OpenRouter (OpenAI-compatible)
// endpoint using a stubbed fetch — no real API calls, no keys required.

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

test("openrouter tool-calling loop answers from tool results", async () => {
  process.env.LLM_PROVIDER = "openrouter";
  process.env.OPENROUTER_API_KEY = "sk-or-v1-test";
  process.env.OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet";

  const { config } = await import("../src/config.js");
  assert.equal(config.llm.provider, "openrouter");

  let openrouterCalls = 0;
  const seenUrls = [];
  globalThis.fetch = async (url, init) => {
    seenUrls.push(url);
    if (url.startsWith("https://openrouter.ai")) {
      const auth = init.headers.Authorization;
      openrouterCalls += 1;
      const first = openrouterCalls === 1;
      return {
        ok: true,
        json: async () => ({
          choices: [
            {
              message: first
                ? {
                    content: null,
                    tool_calls: [
                      {
                        id: "call_1",
                        type: "function",
                        function: { name: "get_recent_assessments", arguments: "{}" },
                      },
                    ],
                  }
                : { content: "Severe zones: Severe Collapse (a01).", tool_calls: null },
            },
          ],
        }),
      };
    }
    if (url.endsWith("/assessments")) {
      return {
        ok: true,
        json: async () => ({
          assessments: [
            { id: "a01", prediction: "Severe Collapse", severity: "severe", confidence: 0.9, createdAt: "2026-09-08T00:00:00Z" },
          ],
        }),
      };
    }
    throw new Error(`unexpected fetch url: ${url}`);
  };

  const { runAgent } = await import("../src/providers.js");
  const res = await runAgent({ query: "Which zones are severe?", token: null, backendUrl: "http://agent-backend.local" });

  assert.ok(seenUrls.includes(OPENROUTER_URL));
  assert.match(res.answer, /Severe Collapse/);
  assert.equal(res.toolCalls.length, 1);
  assert.equal(res.toolCalls[0].tool, "get_recent_assessments");
  assert.equal(res.toolCalls[0].ok, true);
  assert.deepEqual(res.toolCalls[0].input, {});
});

test("openrouter falls back to OPENAI_API_KEY when its own key is missing", async () => {
  process.env.LLM_PROVIDER = "openrouter";
  delete process.env.OPENROUTER_API_KEY;
  process.env.OPENAI_API_KEY = "sk-openai-test";

  const { config } = await import("../src/config.js");
  assert.equal(config.llm.provider, "openrouter");

  const seenAuth = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (url.startsWith("https://openrouter.ai")) {
      seenAuth.push(init.headers.Authorization);
      return {
        ok: true,
        json: async () => ({
          choices: [{ message: { content: "No severe zones right now.", tool_calls: null } }],
        }),
      };
    }
    throw new Error(`unexpected fetch url: ${url}`);
  };

  try {
    const { runAgent } = await import("../src/providers.js");
    const res = await runAgent({
      query: "how many assessments?",
      token: null,
      backendUrl: "http://agent-backend.local",
    });
    assert.match(res.answer, /No severe zones/);
    assert.equal(seenAuth[0], "Bearer sk-openai-test");
  } finally {
    globalThis.fetch = originalFetch;
  }
});