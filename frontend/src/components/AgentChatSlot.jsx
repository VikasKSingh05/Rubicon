import { useRef, useState } from "react";
import { agentQuery } from "../lib/agent.js";

// Phase 4 — tool-grounded agent chat. The agent answers only from backend tool
// results; each reply shows the tools it checked (🔧 chips) via tool_calls.
const SUGGESTIONS = [
  "Show severe zones from today",
  "Is the last upload verified on-chain?",
  "How many assessments do we have?",
];

function ToolChips({ calls }) {
  if (!calls || calls.length === 0) return null;
  return (
    <div className="mt-1.5 flex flex-wrap gap-1">
      {calls.map((tc, i) => (
        <span
          key={`${tc.tool}-${i}`}
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
            tc.ok
              ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200"
              : "bg-red-50 text-red-600 ring-1 ring-red-200"
          }`}
          title={tc.ok ? "tool returned data" : "tool failed — agent refused to guess"}
        >
          🔧 {tc.tool} {tc.ok ? "✓" : "✗"}
        </span>
      ))}
    </div>
  );
}

export default function AgentChatSlot({ assessmentId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef(null);

  async function ask(text) {
    const query = text.trim();
    if (!query || busy) return;
    setMessages((m) => [...m, { role: "user", text: query }]);
    setInput("");
    setBusy(true);
    try {
      const res = await agentQuery(query, { assessmentId });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          toolCalls: res.tool_calls,
          createdAt: res.createdAt,
        },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `I couldn't reach the agent service: ${err.message}`, error: true },
      ]);
    } finally {
      setBusy(false);
      scrollRef.current?.scrollIntoView?.({ block: "end" });
    }
  }

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-2.5">
        <h3 className="text-sm font-semibold text-slate-700">
          Agent Assistant{" "}
          <span className="ml-1 rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-500">
            Phase 4
          </span>
        </h3>
        <span className="flex items-center gap-1 text-[10px] text-slate-400">
          <span className="inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
          tool-grounded
        </span>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 && (
          <div className="space-y-2 pt-1">
            <p className="text-sm text-slate-500">
              Ask about assessments, zones, or on-chain verification. I answer only from live backend
              data.
            </p>
            {SUGGESTIONS.map((q) => (
              <button
                key={q}
                disabled={busy}
                onClick={() => ask(q)}
                className="block w-full rounded-full border border-slate-300 px-3 py-1.5 text-xs text-slate-600 transition hover:border-slate-400 hover:bg-slate-50 disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            {m.role === "user" ? (
              <span className="inline-block max-w-[85%] rounded-2xl rounded-tr-sm bg-slate-800 px-3 py-1.5 text-sm text-white">
                {m.text}
              </span>
            ) : (
              <div
                className={`inline-block max-w-[95%] rounded-2xl rounded-tl-sm border px-3 py-2 text-left text-sm ${
                  m.error
                    ? "border-red-200 bg-red-50 text-red-700"
                    : "border-slate-200 bg-slate-50 text-slate-700"
                }`}
              >
                <p className="whitespace-pre-line">{m.text}</p>
                {!m.error && <ToolChips calls={m.toolCalls} />}
                {m.createdAt && (
                  <p className="mt-1 text-[10px] text-slate-400">
                    {new Date(m.createdAt).toLocaleTimeString()}
                  </p>
                )}
                {m.error && (
                  <button
                    disabled={busy}
                    onClick={() => {
                      const lastUser = [...messages].reverse().find((x) => x.role === "user");
                      if (lastUser) ask(lastUser.text);
                    }}
                    className="mt-1.5 rounded-full border border-red-300 px-2.5 py-0.5 text-[10px] font-medium text-red-600 transition hover:bg-red-100 disabled:opacity-50"
                  >
                    Retry
                  </button>
                )}
              </div>
            )}
          </div>
        ))}

        {busy && (
          <div className="text-left">
            <span className="inline-block rounded-2xl rounded-tl-sm border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm text-slate-400">
              Checking assessments…
            </span>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <form
        className="flex gap-2 border-t border-slate-200 p-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
          placeholder="Ask the agent…"
          className="min-w-0 flex-1 rounded-full border border-slate-300 px-3 py-1.5 text-sm text-slate-700 placeholder:text-slate-400 focus:border-slate-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="shrink-0 rounded-full bg-slate-800 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-slate-700 disabled:opacity-50"
        >
          Ask
        </button>
      </form>
    </div>
  );
}