import React, { useRef, useState } from "react";
import { api } from "../lib/api.js";

const ACCEPT_DRONE = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

export default function DroneInspectorModal({ isOpen, onClose, parentCellId, assessmentId, onInspectionComplete }) {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  if (!isOpen) return null;

  async function handleAnalyze() {
    if (!file) {
      setError("Please select a drone inspection image.");
      return;
    }
    setError(null);
    setAnalyzing(true);
    setResult(null);

    try {
      const form = new FormData();
      form.append("image", file);
      if (parentCellId) form.append("parent_cell_id", parentCellId);
      if (assessmentId) form.append("assessmentId", assessmentId);

      // Using api() helper with FormData
      const res = await api("/drone", {
        method: "POST",
        body: form,
      });

      setResult(res);
      if (onInspectionComplete) {
        onInspectionComplete(res);
      }
    } catch (err) {
      setError(err?.message || "Failed to analyze drone image.");
    } finally {
      setAnalyzing(false);
    }
  }

  function handleReset() {
    setFile(null);
    setResult(null);
    setError(null);
  }

  const severityColorMap = {
    RED: "bg-red-500/15 text-red-700 border-red-500/30",
    ORANGE: "bg-amber-500/15 text-amber-700 border-amber-500/30",
    YELLOW: "bg-yellow-500/15 text-yellow-800 border-yellow-500/30",
    GREEN: "bg-emerald-500/15 text-emerald-700 border-emerald-500/30",
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget && !analyzing) onClose();
      }}
    >
      <div className="relative w-full max-w-xl rounded-xl border border-slate-200 bg-white p-6 shadow-2xl transition-all">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div>
            <h2 className="text-lg font-bold text-slate-800">Stage 2: Drone Triage Inspector</h2>
            <p className="text-xs text-slate-500">
              High-resolution structural failure analysis & failure mode breakdown
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={analyzing}
            className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
          >
            ✕
          </button>
        </div>

        {/* Parent cell context */}
        {parentCellId && (
          <div className="my-3 flex items-center gap-2 rounded-lg bg-blue-50/70 px-3 py-2 text-xs text-blue-800 border border-blue-200/50">
            <span className="font-semibold">Parent H3 Cell:</span>
            <code className="rounded bg-white px-1.5 py-0.5 font-mono text-[11px] text-blue-900 shadow-sm">
              {parentCellId}
            </code>
          </div>
        )}

        {/* Upload area */}
        {!result && (
          <div className="mt-4 space-y-4">
            <div
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current?.click()}
              className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 p-8 text-center transition hover:border-blue-500 hover:bg-blue-50/20"
            >
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT_DRONE.join(",")}
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setFile(f);
                }}
              />
              <div className="mb-2 text-3xl">🚁</div>
              <p className="text-sm font-semibold text-slate-700">
                {file ? file.name : "Select or drag drone inspection imagery"}
              </p>
              <p className="mt-1 text-xs text-slate-400">
                PNG, JPEG, GeoTIFF up to 100 MB
              </p>
            </div>

            {error && (
              <div className="rounded-lg bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
                {error}
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={onClose}
                disabled={analyzing}
                className="rounded-lg px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={!file || analyzing}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {analyzing ? (
                  <>
                    <span className="animate-spin text-sm">↻</span>
                    <span>Analyzing Damage…</span>
                  </>
                ) : (
                  "Run Drone Triage"
                )}
              </button>
            </div>
          </div>
        )}

        {/* Results View */}
        {result && (
          <div className="mt-4 space-y-4">
            <div className="flex items-center justify-between rounded-lg bg-slate-50 p-3 border border-slate-200/80">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Damage Severity
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xl font-bold text-slate-800">
                    {result.damage_percentage}% Area Damaged
                  </span>
                </div>
              </div>
              <span
                className={`rounded-full border px-3 py-1 text-xs font-bold ${
                  severityColorMap[result.severity] || "bg-slate-100 text-slate-800"
                }`}
              >
                {result.severity || "UNKNOWN"} ({result.damage_level?.toUpperCase()})
              </span>
            </div>

            {/* Failure Modes Breakdown */}
            <div className="rounded-lg border border-slate-200 p-3">
              <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-slate-500">
                Detected Failure Modes
              </h4>
              <div className="space-y-2.5">
                {result.failure_modes &&
                  Object.entries(result.failure_modes).map(([key, data]) => {
                    const label =
                      key === "roof_collapse"
                        ? "Roof Structural Collapse"
                        : key === "facade_failure"
                        ? "Facade / Wall Failure"
                        : "Peripheral Debris Field";
                    return (
                      <div key={key} className="rounded-md border border-slate-100 bg-white p-2.5 shadow-sm">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-semibold text-slate-700">{label}</span>
                          <span
                            className={`font-bold ${
                              data.detected ? "text-red-600" : "text-emerald-600"
                            }`}
                          >
                            {data.detected ? "DETECTED" : "CLEAR"} ({(data.confidence * 100).toFixed(0)}% conf)
                          </span>
                        </div>
                        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              data.detected ? "bg-red-500" : "bg-emerald-500"
                            }`}
                            style={{ width: `${Math.min(data.area_pct || data.confidence * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Metrics & Hash */}
            <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-500">
              <div className="truncate">
                <span className="font-semibold">SHA-256: </span>
                <span className="font-mono">{result.image_sha256?.slice(0, 12)}…</span>
              </div>
              <div className="text-right">
                <span className="font-semibold">Execution: </span>
                <span>{result.exec_ms} ms</span>
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex justify-between pt-2">
              <button
                type="button"
                onClick={handleReset}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50"
              >
                Inspect Another Image
              </button>
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-700"
              >
                Done
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
