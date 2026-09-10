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

export default function MapView({ polygons = [], center }) {
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
      </MapContainer>

      {polygons.length === 0 && (
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