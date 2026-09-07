import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import TopBar, { Sidebar } from "../components/TopBar.jsx";
import MapView from "../components/MapView.jsx";
import SeverityLegend from "../components/SeverityLegend.jsx";
import UploadModal from "../components/UploadModal.jsx";

export default function DashboardPage() {
  const navigate = useNavigate();
  const [assessments, setAssessments] = useState([]);
  const [active, setActive] = useState(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [loadError, setLoadError] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await api("/assessments");
      setAssessments(data.assessments);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function handleSelect(item) {
    setActive(item);
  }

  function handleCreated(data) {
    setUploadOpen(false);
    navigate(data.links.detail);
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar onUpload={() => setUploadOpen(true)} />
      <div className="flex min-h-0 flex-1">
        <Sidebar
          assessments={assessments}
          activeId={active?.id}
          onSelect={handleSelect}
        />
        <main className="relative min-w-0 flex-1">
          <MapView
            polygons={assessments}
            center={active?.center}
          />
          <div className="absolute bottom-4 left-4 z-[999]">
            <SeverityLegend />
          </div>
          {loadError && (
            <div className="absolute left-4 top-4 z-[999] rounded-lg bg-white/95 px-3 py-2 text-sm text-severity-severe shadow">
              {loadError}
            </div>
          )}
        </main>
      </div>

      {uploadOpen && (
        <UploadModal
          onClose={() => setUploadOpen(false)}
          onCreated={handleCreated}
        />
      )}
    </div>
  );
}