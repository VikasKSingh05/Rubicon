// Reserved layout slot for the Agent Chat (wired for real in Phase 4).
// Building this slot now avoids a layout reshuffle later.
const SUGGESTIONS = [
  "Show severe zones from today",
  "Is the last upload verified on-chain?",
];

export default function AgentChatSlot() {
  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-4 py-3">
        <h3 className="text-sm font-semibold text-slate-700">
          Agent Assistant{" "}
          <span className="ml-1 rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-500">
            Phase 4
          </span>
        </h3>
      </div>
      <div className="flex flex-1 flex-col items-center justify-center p-4 text-center">
        <p className="text-sm text-slate-500">
          Ask questions about your assessments in plain language.
        </p>
        <div className="mt-3 space-y-2">
          {SUGGESTIONS.map((q) => (
            <button
              key={q}
              disabled
              className="block w-full rounded-full border border-slate-300 px-3 py-1.5 text-xs text-slate-500"
              title="Coming in Phase 4"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}