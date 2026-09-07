import React, { useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON, Popup, useMap } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import { severityOf } from "../lib/severity.js";
import SeverityBadge from "./SeverityBadge.jsx";
import ChainBadge from "./ChainBadge.jsx";

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
      {polygons.map((p) => (
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
                <ChainBadge disabled />
              </div>
            </div>
          </Popup>
        </GeoJSON>
      ))}
    </MapContainer>
  );
}