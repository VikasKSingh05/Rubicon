import React from "react";
import SeverityBadge from "./SeverityBadge.jsx";
import ChainBadge from "./ChainBadge.jsx";
import { downloadBlob } from "../lib/api.js";

function Row({ label, value }) {
  return (
    <div className="flex justify-between gap-2 py-1 text-sm">
      <span className="text-slate-500">{label}</span>
      <span className="text-right font-medium text-slate-800">{value || "—"}</span>
    </div>
  );
}

function DownloadRow({ assessmentId, kind, filename }) {
  const defaultName = kind === "hsi" ? "HSI data" : "LiDAR data";
  return (
    <button
      onClick={() => downloadBlob(`/files/${assessmentId}/${kind}`, filename || defaultName)}
      className="flex w-full items-center justify-between rounded-md border border-slate-200 px-2.5 py-1.5 text-xs transition hover:bg-slate-50"
    >
      <span className="text-slate-700">{kind === "hsi" ? "HSI" : "LiDAR"}</span>
      <span className="font-medium text-primary">{filename || defaultName}</span>
    </button>
  );
}

export function ClassProbBars({ class_probs }) {
  const entries = class_probs ? Object.entries(class_probs) : [];
  const max = Math.max(0.01, ...entries.map(([, v]) => v));
  return (
    <div className="space-y-2">
      {entries.map(([cls, prob]) => (
        <div key={cls}>
          <div className="mb-0.5 flex justify-between text-xs">
            <span className="text-slate-600">{cls}</span>
            <span className="font-medium text-slate-800">{(prob * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${(prob / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ChainInfo({ a }) {
  const hasChain = a.hsiCid || a.lidarCid || a.txHash;
  if (!hasChain) return <p className="text-sm text-slate-500">Pending on-chain proof…</p>;
  return (
    <div>
      {a.hsiCid && <Row label="HSI CID" value={a.hsiCid} />}
      {a.lidarCid && <Row label="LiDAR CID" value={a.lidarCid} />}
      {a.txHash && <Row label="Tx hash" value={`${a.txHash.slice(0, 12)}…`} />}
    </div>
  );
}

export default function DataPanel({ assessment, onReverify, reverifyPending }) {
  if (!assessment) return null;
  const a = assessment;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <SeverityBadge prediction={a.prediction} />
        <ChainBadge
          chainVerified={a.chainVerified}
          href={a.txHash ? `https://amoy.polygonscan.com/tx/${a.txHash}` : null}
        />
      </div>

      <section className="rounded-lg border border-slate-200 p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Prediction
        </h3>
        <p className="text-lg font-semibold text-primary">{a.prediction || "Unknown"}</p>
        <Row label="Confidence" value={`${((a.confidence ?? 0) * 100).toFixed(0)}%`} />
        <Row label="Model" value={a.model_version} />
      </section>

      <section className="rounded-lg border border-slate-200 p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Class probabilities
        </h3>
        <ClassProbBars class_probs={a.class_probs} />
      </section>

      <section className="rounded-lg border border-slate-200 p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Evidence & chain
        </h3>
        <ChainInfo a={a} />
        <Row label="Uploaded" value={a.timestamps?.uploaded ? new Date(a.timestamps.uploaded).toLocaleString() : null} />
        <Row label="Analyzed" value={a.timestamps?.analyzed ? new Date(a.timestamps.analyzed).toLocaleString() : null} />
        {a.timestamps?.chainLogged && (
          <Row label="Chain log" value={new Date(a.timestamps.chainLogged).toLocaleString()} />
        )}
        {a.hsiCid && onReverify && (
          <button
            onClick={onReverify}
            disabled={reverifyPending}
            className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
          >
            {reverifyPending ? "Re-verifying…" : "Re-verify on chain"}
          </button>
        )}
      </section>

      {(a.storage?.hsi || a.storage?.lidar) && (
        <section className="rounded-lg border border-slate-200 p-3">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Original files
          </h3>
          <div className="space-y-1.5">
            {a.storage?.hsi && (
              <DownloadRow assessmentId={a.id} kind="hsi" filename={a.filename?.hsi} />
            )}
            {a.storage?.lidar && (
              <DownloadRow assessmentId={a.id} kind="lidar" filename={a.filename?.lidar} />
            )}
          </div>
        </section>
      )}
    </div>
  );
}