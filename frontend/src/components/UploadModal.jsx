import React, { useRef, useState } from "react";
import { api } from "../lib/api.js";

const STEPS = [
  { key: "uploading", label: "Uploading" },
  { key: "analyzing", label: "Analyzing" },
  { key: "verifying", label: "Verifying on-chain" },
  { key: "done", label: "Done" },
];

export function FileDropzone({ label, accept, file, onFile }) {
  const inputRef = useRef(null);
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-slate-700">{label}</label>
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="flex min-h-24 w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 text-center transition-colors hover:border-primary hover:bg-slate-100"
      >
        {file ? (
          <span className="px-2 text-sm font-medium text-primary">{file.name}</span>
        ) : (
          <>
            <span className="text-xl">📄</span>
            <span className="mt-1 text-xs text-slate-500">
              Click to choose ({accept})
            </span>
          </>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => onFile(e.target.files?.[0] || null)}
      />
    </div>
  );
}

function StepRow({ step, index, activeStep }) {
  const done = activeStep > index;
  const current = activeStep === index;
  return (
    <li className="flex items-center gap-3">
      <span
        className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold ${
          done
            ? "bg-severity-none text-white"
            : current
              ? "bg-primary text-white"
              : "bg-slate-200 text-slate-500"
        }`}
      >
        {done ? "✓" : index + 1}
      </span>
      <span className={`text-sm ${current ? "font-semibold text-primary" : "text-slate-600"}`}>
        {step.label}
      </span>
    </li>
  );
}

export function ProgressStepper({ activeStep }) {
  return (
    <ol className="space-y-3">
      {STEPS.map((step, i) => (
        <StepRow key={step.key} step={step} index={i} activeStep={activeStep} />
      ))}
    </ol>
  );
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function UploadModal({ onClose, onCreated }) {
  const [hsi, setHsi] = useState(null);
  const [lidar, setLidar] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | in-progress | done | error
  const [activeStep, setActiveStep] = useState(0);
  const [error, setError] = useState(null);

  const reset = () => {
    setHsi(null);
    setLidar(null);
    setPhase("idle");
    setActiveStep(0);
    setError(null);
  };

  const canSubmit = hsi && lidar && phase !== "in-progress";

  async function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setPhase("in-progress");
    setActiveStep(0);

    // Steps 2–3 are simulated with delays in Phase 1 so the UX pattern exists
    // before the real pipeline (AI + chain) lands in Phases 3/5.
    try {
      setActiveStep(1); // Analyzing (simulated)
      await sleep(1500);
      setActiveStep(2); // Verifying on-chain (simulated)
      await sleep(1800);

      const form = new FormData();
      form.append("hsi", hsi);
      form.append("lidar", lidar);
      const data = await api("/upload", { method: "POST", body: form });

      setActiveStep(3);
      await sleep(400);
      onCreated(data);
    } catch (err) {
      setError(err.message);
      setPhase("error");
    }
  }

  function close() {
    if (phase === "in-progress") return; // don't lose an in-flight upload
    reset();
    onClose();
  }

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/40 p-4">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-primary">Upload New Scan</h2>
          <button
            onClick={close}
            className="rounded-md px-2 py-1 text-slate-500 hover:bg-slate-100"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {phase === "in-progress" && activeStep < 3 ? (
          <ProgressStepper activeStep={activeStep} />
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <FileDropzone
                label="Hyperspectral (HSI)"
                accept=".tiff,.tif"
                file={hsi}
                onFile={setHsi}
              />
              <FileDropzone
                label="LiDAR"
                accept=".las"
                file={lidar}
                onFile={setLidar}
              />
            </div>
            {error && <p className="mt-3 text-sm text-severity-severe">{error}</p>}
            <button
              type="submit"
              disabled={!canSubmit}
              className="mt-5 w-full rounded-md bg-primary px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Run Assessment
            </button>
          </form>
        )}
      </div>
    </div>
  );
}