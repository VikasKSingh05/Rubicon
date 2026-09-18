import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api.js";
import { useAuth } from "../context/AuthContext.jsx";
import TopBar, { Sidebar } from "../components/TopBar.jsx";
import MapView from "../components/MapView.jsx";
import SeverityLegend from "../components/SeverityLegend.jsx";
import UploadModal from "../components/UploadModal.jsx";

const PAGE_SIZE = 50;

// Mock data for demo mode — realistic Houston-area disaster assessments
const DEMO_ASSESSMENTS = [
  {
    id: "demo-1",
    title: "Downtown Houston — Flood Zone A",
    severity: "severe",
    severity_score: 0.91,
    created_at: new Date().toISOString(),
    center: { lat: 29.7604, lng: -95.3698 },
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[
        [-95.375, 29.755], [-95.365, 29.755], [-95.365, 29.765],
        [-95.375, 29.765], [-95.375, 29.755],
      ]],
    },
  },
  {
    id: "demo-2",
    title: "Memorial Park — Structural Damage",
    severity: "moderate",
    severity_score: 0.62,
    created_at: new Date(Date.now() - 3600000).toISOString(),
    center: { lat: 29.7645, lng: -95.4344 },
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[
        [-95.44, 29.76], [-95.43, 29.76], [-95.43, 29.77],
        [-95.44, 29.77], [-95.44, 29.76],
      ]],
    },
  },
  {
    id: "demo-3",
    title: "Galveston Bay — Coastal Erosion",
    severity: "minor",
    severity_score: 0.28,
    created_at: new Date(Date.now() - 7200000).toISOString(),
    center: { lat: 29.3013, lng: -94.7977 },
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[
        [-94.81, 29.29], [-94.79, 29.29], [-94.79, 29.31],
        [-94.81, 29.31], [-94.81, 29.29],
      ]],
    },
  },
  {
    id: "demo-4",
    title: "Energy Corridor — Infrastructure Impact",
    severity: "severe",
    severity_score: 0.85,
    created_at: new Date(Date.now() - 10800000).toISOString(),
    center: { lat: 29.7786, lng: -95.5619 },
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[
        [-95.57, 29.77], [-95.55, 29.77], [-95.55, 29.79],
        [-95.57, 29.79], [-95.57, 29.77],
      ]],
    },
  },
  {
    id: "demo-5",
    title: "Texas Medical Center — Minimal Damage",
    severity: "none",
    severity_score: 0.05,
    created_at: new Date(Date.now() - 14400000).toISOString(),
    center: { lat: 29.7066, lng: -95.3988 },
    geojson_polygon: {
      type: "Polygon",
      coordinates: [[
        [-95.405, 29.700], [-95.393, 29.700], [-95.393, 29.713],
        [-95.405, 29.713], [-95.405, 29.700],
      ]],
    },
  },
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const { demoMode } = useAuth();
  const [assessments, setAssessments] = useState([]);
  const [total, setTotal] = useState(0);
  const [active, setActive] = useState(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    if (demoMode) {
      // Use mock data in demo mode
      setAssessments(DEMO_ASSESSMENTS);
      setTotal(DEMO_ASSESSMENTS.length);
      setLoading(false);
      return;
    }
    try {
      const data = await api(`/assessments?limit=${PAGE_SIZE}&offset=0`);
      setAssessments(data.assessments);
      setTotal(data.total);
    } catch (err) {
      setLoadError(err.message);
    } finally {
      setLoading(false);
    }
  }, [demoMode]);

  useEffect(() => {
    load();
  }, [load]);

  async function handleLoadMore() {
    if (demoMode) return; // No pagination in demo mode
    setLoadError(null);
    setLoadingMore(true);
    try {
      const data = await api(`/assessments?limit=${PAGE_SIZE}&offset=${assessments.length}`);
      setAssessments((prev) => [...prev, ...data.assessments]);
      setTotal(data.total);
    } catch (err) {
      setLoadError(err.message);
    } finally {
      setLoadingMore(false);
    }
  }

  function handleSelect(item) {
    setActive(item);
  }

  function handleCreated(data) {
    setUploadOpen(false);
    navigate(data.links.detail);
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar
          onUpload={() => setUploadOpen(true)}
          onRefresh={loading ? null : load}
          refreshing={loading}
        />
      <div className="flex min-h-0 flex-1">
        <Sidebar
          assessments={assessments}
          activeId={active?.id}
          onSelect={handleSelect}
          loading={loading}
          canLoadMore={!loading && assessments.length < total}
          loadingMore={loadingMore}
          onLoadMore={handleLoadMore}
        />
        <main className="relative min-w-0 flex-1">
          <MapView
            polygons={assessments}
            center={active?.center}
          />
          <div className="absolute bottom-4 left-4 z-[999]">
            <SeverityLegend />
          </div>
          {demoMode && (
            <div className="absolute left-1/2 top-4 z-[999] -translate-x-1/2 rounded-full bg-primary/90 px-4 py-1.5 text-xs font-semibold text-white shadow-lg backdrop-blur-sm">
              🚀 Demo Mode — Viewing mock assessment data
            </div>
          )}
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