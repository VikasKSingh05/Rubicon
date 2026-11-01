import React from "react";
import { SEVERITY, VERIFY_BADGE } from "../lib/severity.js";

export default function SeverityLegend() {
  return (
    <div className="space-y-1.5 rounded-lg border border-slate-200 bg-white/95 p-3 text-xs shadow-md backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-slate-100 pb-1">
        <p className="font-semibold text-slate-800">xBD Damage & Heatmap</p>
      </div>
      <div className="space-y-1 pt-0.5">
        {Object.entries(SEVERITY).map(([key, s]) => (
          <div key={key} className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 shrink-0 rounded-full shadow-sm"
              style={{ backgroundColor: s.color }}
            />
            <span className="text-slate-700 font-medium">
              {s.label}
            </span>
          </div>
        ))}
        <div className="flex items-center gap-2 border-t border-slate-100 pt-1.5 mt-1">
          <span
            className="inline-block h-2.5 w-2.5 shrink-0 rounded-full shadow-sm"
            style={{ backgroundColor: VERIFY_BADGE.color }}
          />
          <span className="text-slate-600 font-medium">Verified on Amoy</span>
        </div>
      </div>
    </div>
  );
}