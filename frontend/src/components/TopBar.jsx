import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { severityOf } from "../lib/severity.js";
import { useAuth } from "../context/AuthContext.jsx";
import ChangePasswordModal from "./ChangePasswordModal.jsx";

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function TopBar({ onUpload, onRefresh, refreshing }) {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [changeOpen, setChangeOpen] = useState(false);
  return (
    <>
    <header className="flex h-14 items-center justify-between border-b border-primary/20 bg-primary px-4 text-white">
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-left"
        title="Back to dashboard"
      >
        <span className="text-lg font-bold">Rubicon</span>
        <span className="hidden text-xs text-white/70 sm:inline">
          Disaster Assessment
        </span>
      </button>
      <div className="flex items-center gap-3">
        <span className="hidden max-w-48 truncate text-xs text-white/80 sm:inline">
          {user?.email}
        </span>
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={refreshing}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
            title="Refresh"
          >
            {refreshing ? "↻" : "⟳"}
          </button>
        )}
        <button
          onClick={onUpload}
          className="rounded-md bg-verify px-3 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-verify/90"
        >
          + Upload New Scan
        </button>
        <button
          onClick={() => setChangeOpen(true)}
          className="rounded-md px-2 py-1 text-xs text-white/80 transition-colors hover:bg-white/10"
        >
          Password
        </button>
        <button
          onClick={() => {
            logout();
            navigate("/login");
          }}
          className="rounded-md px-2 py-1 text-xs text-white/80 transition-colors hover:bg-white/10"
        >
          Logout
        </button>
      </div>
    </header>
    {changeOpen && <ChangePasswordModal onClose={() => setChangeOpen(false)} />}
    </>
  );
}

export function AssessmentList({ assessments, activeId, onSelect, loading = false }) {
  if (loading) {
    return (
      <div className="p-4 text-sm text-slate-400" role="status">
        Loading assessments…
      </div>
    );
  }
  if (assessments.length === 0) {
    return (
      <p className="p-4 text-sm text-slate-500">
        No assessments yet. Upload a scan to get started.
      </p>
    );
  }
  return (
    <ul className="divide-y divide-slate-200">
      {assessments.map((a) => {
        const active = a.id === activeId;
        return (
          <li key={a.id}>
            <button
              onClick={() => onSelect(a)}
              className={`flex w-full items-start gap-2 px-4 py-3 text-left transition-colors ${
                active ? "bg-primary/10" : "hover:bg-slate-100"
              }`}
            >
              <span
                className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: severityOf(a.prediction).color }}
              />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-slate-800">
                  {a.prediction || "Unknown"}
                </p>
                <p className="mt-1 truncate text-xs text-slate-500">
                  {a.center ? `${a.center.lat.toFixed(4)}, ${a.center.lng.toFixed(4)}` : "no coords"}
                </p>
                <p className="mt-0.5 text-xs text-slate-400">{formatTime(a.createdAt)}</p>
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export function Sidebar({
  assessments,
  activeId,
  onSelect,
  loading = false,
  canLoadMore = false,
  loadingMore = false,
  onLoadMore,
}) {
  return (
    <aside className="w-72 shrink-0 overflow-y-auto border-r border-slate-300 bg-white">
      <div className="border-b border-slate-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-700">Past Assessments</h2>
      </div>
      <AssessmentList
        assessments={assessments}
        activeId={activeId}
        onSelect={onSelect}
        loading={loading}
      />
      {canLoadMore && !loading && (
        <div className="border-t border-slate-200 p-3">
          <button
            onClick={onLoadMore}
            disabled={loadingMore}
            className="w-full rounded-md border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-600 transition-colors hover:bg-slate-50 disabled:opacity-60"
          >
            {loadingMore ? "Loading…" : "Load more"}
          </button>
        </div>
      )}
    </aside>
  );
}