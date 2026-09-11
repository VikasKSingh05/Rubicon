import React, { useCallback, useEffect, useState } from "react";
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
  const [reverifyPending, setReverifyPending] = useState(false);

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

  const handleReverify = useCallback(async () => {
    if (!assessment?.hsiCid || reverifyPending) return;
    setReverifyPending(true);
    try {
      const res = await api(`/chain/verify/${assessment.hsiCid}`);
      if (res.assessmentId) {
        const fresh = await api(`/assessments/${id}`);
        setAssessment(fresh);
      }
    } catch {
      // ignore — badge will remain stale but harmless
    } finally {
      setReverifyPending(false);
    }
  }, [assessment?.hsiCid, id, reverifyPending]);

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
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <main className="min-h-[50vh] min-w-0 flex-1 lg:min-h-0">
          <MapView polygons={[assessment]} center={center} />
        </main>
        <aside className="flex w-full shrink-0 flex-col border-t border-slate-300 bg-white sm:w-full lg:w-[380px] lg:border-l lg:border-t-0">
          <div className="min-h-0 flex-1 overflow-y-auto p-4">
            <DataPanel
              assessment={assessment}
              onReverify={assessment.hsiCid ? handleReverify : null}
              reverifyPending={reverifyPending}
            />
          </div>
          <div className="h-72 shrink-0 border-t border-slate-200 p-3">
            <AgentChatSlot assessmentId={assessment.id} />
          </div>
        </aside>
      </div>
    </div>
  );
}
