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

test("anthropic tool-calling loop answers from tool results", async () => {
  const { config } = await import("../src/config.js");
  const previousProvider = config.llm.provider;
  config.llm.provider = "anthropic";
  process.env.ANTHROPIC_API_KEY = "sk-ant-test";

  let calls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (!url.startsWith("https://api.anthropic.com")) {
      return { ok: true, json: async () => ({}) }; // backend tool URL
    }
    calls += 1;
    const first = calls === 1;
    return {
      ok: true,
      json: async () => ({
        content: first
          ? [
              {
                type: "tool_use",
                id: "toolu_1",
                name: "get_assessment",
                input: { assessmentId: "a01" },
              },
            ]
          : [{ type: "text", text: "Zone a01 was assessed: Severe Collapse (91%)." }],
      }),
    };
  };

  try {
    const { runAgent } = await import("../src/providers.js");
    const res = await runAgent({
      query: "how was zone a01 assessed?",
      token: null,
      backendUrl: "http://agent-backend.local",
    });
    assert.match(res.answer, /Severe Collapse/);
    assert.equal(res.toolCalls.length, 1);
    assert.equal(res.toolCalls[0].tool, "get_assessment");
  } finally {
    globalThis.fetch = originalFetch;
    delete process.env.ANTHROPIC_API_KEY;
    config.llm.provider = previousProvider;
  }
});

test("tool-calling loop stops after the iteration limit instead of spinning", async () => {
  const { config } = await import("../src/config.js");
  const previousProvider = config.llm.provider;
  config.llm.provider = "openai";
  process.env.OPENAI_API_KEY = "sk-openai-test";

  let calls = 0;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (!url.startsWith("https://api.openai.com")) {
      return { ok: true, json: async () => ({}) }; // backend tool URL
    }
    calls += 1;
    return {
      ok: true,
      json: async () => ({
        choices: [
          // The model always requests a tool, never producing text.
          {
            message: {
              content: null,
              tool_calls: [
                {
                  id: "call_n",
                  type: "function",
                  function: { name: "get_recent_assessments", arguments: "{}" },
                },
              ],
            },
          },
        ],
      }),
    };
  };

  try {
    const { ITERATION_LIMIT, runAgent } = await import("../src/providers.js");
    const res = await runAgent({
      query: "keep looping please",
      token: null,
      backendUrl: "http://agent-backend.local",
    });
    assert.equal(calls, ITERATION_LIMIT, "provider calls must be capped");
    assert.equal(res.toolCalls.length, ITERATION_LIMIT);
    assert.match(res.answer, /I don't have an answer from my tools yet/);
  } finally {
    globalThis.fetch = originalFetch;
    delete process.env.OPENAI_API_KEY;
    config.llm.provider = previousProvider;
  }
});