import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../lib/api.js";
import TopBar from "../components/TopBar.jsx";
import MapView from "../components/MapView.jsx";
import DataPanel from "../components/DataPanel.jsx";
import AgentChatSlot from "../components/AgentChatSlot.jsx";

export default function AssessmentPage() {
  const { id } = useParams();
  const [assessment, setAssessment] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api(`/assessments/${id}`)
      .then((data) => {
        if (!cancelled) setAssessment(data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setStatus(err.status || 0);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    const notFound = status === 404;
    return (
      <div className="flex h-full flex-col">
        <TopBar onUpload={() => {}} />
        <div className="flex flex-1 items-center justify-center">
          <div className="text-center">
            <p
              className={`${notFound ? "text-4xl font-bold text-slate-400" : "text-severity-severe"}`}
            >
              {notFound ? "404" : ""}
            </p>
            <p className="mt-1 text-slate-600">
              {notFound ? "Assessment not found." : error}
            </p>
            <Link to="/" className="mt-2 inline-block text-sm underline">
              Back to dashboard
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!assessment) {
    return (
      <div className="flex h-full flex-col">
        <TopBar onUpload={() => {}} />
        <div className="flex flex-1 items-center justify-center text-slate-400">
          Loading assessment…
        </div>
      </div>
    );
  }

  const center = (() => {
    const ring = assessment.geojson_polygon?.coordinates?.[0] || [];
    if (ring.length === 0) return null;
    const lngs = ring.map((c) => c[0]);
    const lats = ring.map((c) => c[1]);
    return { lng: (Math.min(...lngs) + Math.max(...lngs)) / 2, lat: (Math.min(...lats) + Math.max(...lats)) / 2 };
  })();

  return (
    <div className="flex h-full flex-col">
      <TopBar onUpload={() => {}} />
      <div className="flex min-h-0 flex-1">
        <main className="min-w-0 flex-1">
          <MapView polygons={[assessment]} center={center} />
        </main>
        <aside className="flex w-[380px] shrink-0 flex-col border-l border-slate-300 bg-white">
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            <DataPanel assessment={assessment} />
          </div>
          <div className="h-72 shrink-0 border-t border-slate-200 p-3">
            <AgentChatSlot assessmentId={assessment.id} />
          </div>
        </aside>
      </div>
    </div>
  );
}