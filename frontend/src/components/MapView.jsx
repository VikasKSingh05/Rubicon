import React, { useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON, Popup, useMap } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import { severityOf } from "../lib/severity.js";
import SeverityBadge from "./SeverityBadge.jsx";
import ChainBadge from "./ChainBadge.jsx";

const AMOY_EXPLORER = "https://amoy.polygonscan.com";

function FlyTo({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center) {
      map.flyTo([center.lat, center.lng], 13, { duration: 0.8 });
    }
  }, [map, center]);
  return null;
}

function polygonStyle(feature) {
  const s = severityOf(feature.properties?.prediction);
  return {
    color: s.color,
    weight: 2,
    fillColor: s.color,
    fillOpacity: 0.25,
  };
}

const CELL_COLORS = {
  RED: "#DC2626",
  ORANGE: "#F59E0B",
  YELLOW: "#EAB308",
  GREEN: "#22C55E",
};

function cellStyle(feature) {
  const sev = feature.properties?.severity || "GREEN";
  const color = CELL_COLORS[sev] || "#22C55E";
  return {
    color,
    weight: 1.5,
    fillColor: color,
    fillOpacity: 0.35,
  };
}

export default function MapView({ polygons = [], cells = null, center, onInspectCell = null }) {
  const navigate = useNavigate();
  const defaultCenter = center ? [center.lat, center.lng] : [29.76, -95.36];
  const defaultZoom = center ? 13 : 4;

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={defaultCenter}
        zoom={defaultZoom}
        className="h-full w-full"
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FlyTo center={center} />

        {/* Polygons (Assessments) */}
        {polygons.map((p) =>
          p.geojson_polygon ? (
            <GeoJSON
              key={p.id}
              data={{
                type: "Feature",
                properties: { prediction: p.prediction },
                geometry: p.geojson_polygon,
              }}
              style={polygonStyle}
            >
              <Popup>
                <div className="min-w-[180px]">
                  <SeverityBadge prediction={p.prediction} />
                  <div className="mt-1 text-xs font-medium text-slate-700">
                    Confidence: {(p.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="mt-2 flex flex-col gap-1.5">
                    <button
                      onClick={() => navigate(`/assessments/${p.id}`)}
                      className="rounded-md bg-primary px-2 py-1 text-xs font-semibold text-white hover:bg-primary/90"
                    >
                      View Details
                    </button>
                    <ChainBadge
                      chainVerified={p.chainVerified}
                      href={
                        p.txHash
                          ? `${AMOY_EXPLORER}/tx/${p.txHash}`
                          : undefined
                      }
                    />
                  </div>
                </div>
              </Popup>
            </GeoJSON>
          ) : null
        )}

        {/* H3 Hexagonal Grid Cells (Stage 1 Satellite Triage Aggregation) */}
        {cells?.features?.map((f, i) => (
          <GeoJSON
            key={f.id || `h3_cell_${i}`}
            data={f}
            style={cellStyle}
          >
            <Popup>
              <div className="min-w-[200px] text-xs">
                <div className="flex items-center justify-between border-b border-slate-100 pb-1 font-bold">
                  <span className="text-slate-800">H3 Cell</span>
                  <span
                    className="rounded px-1.5 py-0.5 text-[10px] font-bold text-white"
                    style={{ backgroundColor: CELL_COLORS[f.properties?.severity] || "#22C55E" }}
                  >
                    {f.properties?.severity}
                  </span>
                </div>
                <div className="mt-1 font-mono text-[11px] text-slate-600">
                  {f.properties?.cell_id}
                </div>
                {f.properties?.building_count !== undefined && (
                  <div className="mt-1 text-slate-600">
                    Buildings: <span className="font-semibold text-slate-800">{f.properties.building_count}</span>
                  </div>
                )}
                {f.properties?.damage_ratio !== undefined && (
                  <div className="text-slate-600">
                    Damage ratio: <span className="font-semibold text-slate-800">{((f.properties.damage_ratio || 0) * 100).toFixed(0)}%</span>
                  </div>
                )}
                {onInspectCell && (
                  <button
                    onClick={() => onInspectCell(f.properties?.cell_id)}
                    className="mt-2.5 w-full rounded bg-blue-600 px-2 py-1.5 font-semibold text-white shadow-sm hover:bg-blue-700"
                  >
                    🚁 Inspect with Drone (Stage 2)
                  </button>
                )}
              </div>
            </Popup>
          </GeoJSON>
        ))}
      </MapContainer>

      {polygons.length === 0 && (!cells || cells.features?.length === 0) && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="rounded-lg bg-white/95 px-4 py-3 text-center shadow">
            <p className="text-sm font-semibold text-slate-700">
              No assessments mapped yet
            </p>
            <p className="mt-0.5 text-xs text-slate-500">
              Upload a scan to start assessing damage.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}