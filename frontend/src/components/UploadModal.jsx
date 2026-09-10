import React, { useRef, useState } from "react";
import { api } from "../lib/api.js";

const STEPS = [
  { key: "uploading", label: "Uploading" },
  { key: "analyzing", label: "Analyzing" },
  { key: "verifying", label: "Verifying on-chain" },
  { key: "done", label: "Done" },
];

const ACCEPT = {
  hsi: [".tif", ".tiff"],
  lidar: [".las", ".laz"],
};

const MAX_BYTES = 100 * 1024 * 1024;

function fileError(file, kind) {
  if (!file) return null;
  const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;
  if (!ACCEPT[kind].includes(ext)) {
    return `Only ${ACCEPT[kind].join(", ")} files are supported.`;
  }
  if (file.size > MAX_BYTES) return "File exceeds the 100 MB limit.";
  return null;
}

export function FileDropzone({ label, kind, file, error, onFile }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const accept = ACCEPT[kind].join(",");

  function pick(f) {
    setDragging(false);
    if (f) onFile(f);
  }

  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-slate-700">{label}</label>
      <div
        role="button"
        tabIndex={0}
        aria-label={`Choose ${label} file`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          pick(e.dataTransfer.files?.[0] || null);
        }}
        className={`flex min-h-24 w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed text-center transition-colors ${
          dragging
            ? "border-primary bg-primary/10"
            : "border-slate-300 bg-slate-50 hover:border-primary hover:bg-slate-100"
        }`}
      >
        {file ? (
          <span className="px-2 text-sm font-medium text-primary">
            {file.name}{" "}
            <span className="text-slate-400">({(file.size / 1024).toFixed(0)} KB)</span>
          </span>
        ) : (
          <>
            <span className="text-xl">📄</span>
            <span className="mt-1 px-2 text-xs text-slate-500">
              Click or drop ({accept})
            </span>
          </>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => pick(e.target.files?.[0] || null)}
      />
      {error && (
        <p className="mt-1 text-xs text-severity-severe" role="alert">
          {error}
        </p>
      )}
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

  const hsiFileError = fileError(hsi, "hsi");
  const lidarFileError = fileError(lidar, "lidar");
  const canSubmit =
    hsi && lidar && !hsiFileError && !lidarFileError && phase !== "in-progress";

  async function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setPhase("in-progress");
    setActiveStep(0); // Uploading — the server runs analyze + verify in the same request

    try {
      const form = new FormData();
      form.append("hsi", hsi);
      form.append("lidar", lidar);
      const data = await api("/upload", { method: "POST", body: form });

      setActiveStep(STEPS.length); // all steps complete server-side
      await new Promise((r) => setTimeout(r, 450)); // let "Done" register before closing
      onCreated(data);
    } catch (err) {
      setActiveStep(0);
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

        {phase === "in-progress" && activeStep < STEPS.length ? (
          <ProgressStepper activeStep={activeStep} />
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <FileDropzone
                label="Hyperspectral (HSI)"
                kind="hsi"
                file={hsi}
                error={phase === "error" || hsi ? hsiFileError : null}
                onFile={setHsi}
              />
              <FileDropzone
                label="LiDAR"
                kind="lidar"
                file={lidar}
                error={phase === "error" || lidar ? lidarFileError : null}
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