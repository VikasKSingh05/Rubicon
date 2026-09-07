import { SEVERITY, VERIFY_BADGE } from "../lib/severity.js";

export default function SeverityLegend() {
  return (
    <div className="space-y-1 rounded-lg border border-slate-200 bg-white/95 p-3 text-xs shadow-sm backdrop-blur">
      <p className="font-semibold text-slate-700">Severity</p>
      {Object.entries(SEVERITY).map(([key, s]) => (
        <div key={key} className="flex items-center gap-2">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: s.color }}
          />
          <span className="text-slate-600">
            {s.icon && <span aria-hidden>{s.icon} </span>}
            {s.label}
          </span>
        </div>
      ))}
      <div className="flex items-center gap-2 border-t border-slate-200 pt-1 mt-1">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: VERIFY_BADGE.color }}
        />
        <span className="text-slate-600">Verified on-chain</span>
      </div>
    </div>
  );
}