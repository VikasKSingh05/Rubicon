import React, { useEffect, useState } from "react";
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

// Generate circular radial geometry approximation for smooth GIS heatmap rendering
function generateCirclePolygon(centerLng, centerLat, radiusMeters, steps = 24) {
  const coords = [];
  const degLat = radiusMeters / 111139.0;
  const degLng = radiusMeters / (111139.0 * Math.cos((centerLat * Math.PI) / 180.0));
  for (let i = 0; i < steps; i++) {
    const angle = (Math.PI * 2 * i) / steps;
    const lng = Number((centerLng + degLng * Math.cos(angle)).toFixed(7));
    const lat = Number((centerLat + degLat * Math.sin(angle)).toFixed(7));
    coords.push([lng, lat]);
  }
  coords.push(coords[0]);
  return coords;
}

function makeTileBBox(centerLng, centerLat, sizeDeg = 0.012) {
  const half = sizeDeg / 2;
  return [
    [
      [Number((centerLng - half).toFixed(6)), Number((centerLat - half).toFixed(6))],
      [Number((centerLng + half).toFixed(6)), Number((centerLat - half).toFixed(6))],
      [Number((centerLng + half).toFixed(6)), Number((centerLat + half).toFixed(6))],
      [Number((centerLng - half).toFixed(6)), Number((centerLat + half).toFixed(6))],
      [Number((centerLng - half).toFixed(6)), Number((centerLat - half).toFixed(6))],
    ],
  ];
}

// Build multi-ring radial heat blobs from an assessment center point
function generateHeatmapFeatures(polygonDoc) {
  const ring = polygonDoc?.geojson_polygon?.coordinates?.[0] || [];
  if (ring.length === 0) return [];
  const lngs = ring.map((c) => c[0]);
  const lats = ring.map((c) => c[1]);
  const centerLng = (Math.min(...lngs) + Math.max(...lngs)) / 2;
  const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;

  const pred = polygonDoc.prediction || "";
  const s = severityOf(pred);
  const color = s.color;

  // Concentric radial gradient rings
  const layers = [
    { radius: 750, opacity: 0.12, weight: 0 },
    { radius: 480, opacity: 0.25, weight: 0 },
    { radius: 260, opacity: 0.45, weight: 1 },
    { radius: 110, opacity: 0.75, weight: 1.5 },
  ];

  return layers.map((lvl, idx) => ({
    type: "Feature",
    id: `heat_${polygonDoc.id}_${idx}`,
    geometry: {
      type: "Polygon",
      coordinates: [generateCirclePolygon(centerLng, centerLat, lvl.radius)],
    },
    properties: {
      color,
      fillOpacity: lvl.opacity,
      weight: lvl.weight,
      prediction: pred,
      confidence: polygonDoc.confidence,
      heatLevel: idx,
    },
  }));
}

// Global trained xBD satellite disaster imagery clusters
export const XBD_TRAINED_DISASTER_CLUSTERS = [
  {
    id: "xbd-santa-rosa",
    name: "Santa Rosa Wildfire",
    region: "California, USA",
    emoji: "🌲",
    lat: 38.4404,
    lng: -122.7141,
    prediction: "Destroyed",
    count: 480,
    confidence: 0.942,
    preFile: "santa-rosa-wildfire_00000001_pre_disaster.png",
    postFile: "santa-rosa-wildfire_00000001_post_disaster.png",
    distribution: { destroyed: 70, major: 17, minor: 8, none: 5 },
  },
  {
    id: "xbd-hurricane-michael",
    name: "Hurricane Michael",
    region: "Mexico Beach, Florida, USA",
    emoji: "🌀",
    lat: 29.9438,
    lng: -85.4216,
    prediction: "Destroyed",
    count: 620,
    confidence: 0.965,
    preFile: "hurricane-michael_00000014_pre_disaster.png",
    postFile: "hurricane-michael_00000014_post_disaster.png",
    distribution: { destroyed: 78, major: 15, minor: 5, none: 2 },
  },
  {
    id: "xbd-mexico-earthquake",
    name: "Mexico City Earthquake",
    region: "Puebla, Mexico",
    emoji: "🏢",
    lat: 19.3400,
    lng: -99.1500,
    prediction: "Major Damage",
    count: 340,
    confidence: 0.895,
    preFile: "mexico-earthquake_00000021_pre_disaster.png",
    postFile: "mexico-earthquake_00000021_post_disaster.png",
    distribution: { destroyed: 12, major: 51, minor: 22, none: 15 },
  },
  {
    id: "xbd-volcano-guatemala",
    name: "Volcán de Fuego Eruption",
    region: "Escuintla, Guatemala",
    emoji: "🌋",
    lat: 14.4747,
    lng: -90.8808,
    prediction: "Destroyed",
    count: 290,
    confidence: 0.938,
    preFile: "volcano-fuego_00000030_pre_disaster.png",
    postFile: "volcano-fuego_00000030_post_disaster.png",
    distribution: { destroyed: 70, major: 20, minor: 6, none: 4 },
  },
  {
    id: "xbd-palu-tsunami",
    name: "Palu Earthquake & Tsunami",
    region: "Sulawesi, Indonesia",
    emoji: "🌊",
    lat: -0.8917,
    lng: 119.8707,
    prediction: "Destroyed",
    count: 510,
    confidence: 0.951,
    preFile: "palu-tsunami_00000045_pre_disaster.png",
    postFile: "palu-tsunami_00000045_post_disaster.png",
    distribution: { destroyed: 68, major: 22, minor: 7, none: 3 },
  },
  {
    id: "xbd-midwest-flood",
    name: "Midwest River Basin Floods",
    region: "Nebraska / Iowa, USA",
    emoji: "🌾",
    lat: 41.2565,
    lng: -95.9345,
    prediction: "Minor Damage",
    count: 180,
    confidence: 0.862,
    preFile: "midwest-floods_00000050_pre_disaster.png",
    postFile: "midwest-floods_00000050_post_disaster.png",
    distribution: { destroyed: 6, major: 14, minor: 48, none: 32 },
  },
  {
    id: "xbd-joplin-tornado",
    name: "Joplin EF5 Tornado",
    region: "Missouri, USA",
    emoji: "🌪️",
    lat: 37.0842,
    lng: -94.5133,
    prediction: "Destroyed",
    count: 420,
    confidence: 0.971,
    preFile: "joplin-tornado_00000080_pre_disaster.png",
    postFile: "joplin-tornado_00000080_post_disaster.png",
    distribution: { destroyed: 77, major: 18, minor: 4, none: 1 },
  },
  {
    id: "xbd-woolsey-fire",
    name: "Woolsey Wildfire",
    region: "Malibu / LA, California",
    emoji: "🔥",
    lat: 34.0259,
    lng: -118.7798,
    prediction: "Major Damage",
    count: 310,
    confidence: 0.912,
    preFile: "woolsey-fire_00000060_pre_disaster.png",
    postFile: "woolsey-fire_00000060_post_disaster.png",
    distribution: { destroyed: 18, major: 52, minor: 18, none: 12 },
  },
  {
    id: "xbd-pinery-bushfire",
    name: "Pinery Bushfire",
    region: "South Australia",
    emoji: "🇦🇺",
    lat: -34.3317,
    lng: 138.5667,
    prediction: "Major Damage",
    count: 210,
    confidence: 0.889,
    preFile: "pinery-bushfire_00000070_pre_disaster.png",
    postFile: "pinery-bushfire_00000070_post_disaster.png",
    distribution: { destroyed: 12, major: 46, minor: 24, none: 18 },
  },
  {
    id: "xbd-portugal-wildfire",
    name: "Pedrógão Grande Wildfire",
    region: "Portugal",
    emoji: "🇵🇹",
    lat: 39.9167,
    lng: -8.1444,
    prediction: "Major Damage",
    count: 260,
    confidence: 0.903,
    preFile: "portugal-wildfire_00000090_pre_disaster.png",
    postFile: "portugal-wildfire_00000090_post_disaster.png",
    distribution: { destroyed: 16, major: 49, minor: 21, none: 14 },
  },
];

export default function MapView({ polygons = [], cells = null, center, onInspectCell = null }) {
  const navigate = useNavigate();
  const [internalCenter, setInternalCenter] = useState(null);
  const [layerMode, setLayerMode] = useState("all"); // 'all' | 'heatmap' | 'cells' | 'footprints'
  const [showTrainedXbd, setShowTrainedXbd] = useState(false);
  const [activeDisasterId, setActiveDisasterId] = useState(null);

  const effectiveCenter = internalCenter || (center ? [center.lat, center.lng] : [38.44, -122.71]);
  const defaultZoom = center || internalCenter ? 13 : 4;

  const showHeatmap = layerMode === "all" || layerMode === "heatmap";
  const showFootprints = layerMode === "all" || layerMode === "footprints";
  const showCells = layerMode === "all" || layerMode === "cells";

  function handleJumpToDisaster(cluster) {
    setInternalCenter({ lat: cluster.lat, lng: cluster.lng });
    setShowTrainedXbd(true);
    setActiveDisasterId(cluster.id);
  }

  // Aggregate heatmap features from all mapped polygons
  const assessmentHeatmaps = polygons.flatMap((p) => generateHeatmapFeatures(p));

  // Generate heatmap features from trained xBD dataset clusters if enabled
  const trainedHeatmaps = showTrainedXbd
    ? XBD_TRAINED_DISASTER_CLUSTERS.flatMap((c) => {
        const dummyDoc = {
          id: c.id,
          prediction: c.prediction,
          confidence: c.confidence,
          geojson_polygon: {
            type: "Polygon",
            coordinates: [generateCirclePolygon(c.lng, c.lat, 2500, 20)],
          },
        };
        return generateHeatmapFeatures(dummyDoc);
      })
    : [];

  const heatmapFeatures = [...assessmentHeatmaps, ...trainedHeatmaps];

  // Satellite Footprint Bounding Boxes for trained xBD tiles
  const trainedTileFootprints = showTrainedXbd
    ? XBD_TRAINED_DISASTER_CLUSTERS.map((c) => ({
        type: "Feature",
        id: `tile_${c.id}`,
        geometry: {
          type: "Polygon",
          coordinates: makeTileBBox(c.lng, c.lat, 0.012),
        },
        properties: {
          ...c,
          isTrainedTile: true,
        },
      }))
    : [];

  return (
    <div className="relative h-full w-full">
      {/* Unified Top Navigation & Controls Overlay */}
      <div className="absolute top-3 left-3 right-3 z-[1000] flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        {/* Left Side: xBD Trained Scenes Quick-Jump Navigator */}
        <div className="pointer-events-auto flex items-center gap-1.5 overflow-x-auto rounded-lg border border-slate-200 bg-white/95 p-1.5 shadow-md backdrop-blur-sm scrollbar-none max-w-full lg:max-w-[62%]">
          <span className="shrink-0 px-1 text-[11px] font-bold text-slate-700 flex items-center gap-1">
            <span>🛰️</span>
            <span className="hidden sm:inline">xBD Scenes:</span>
          </span>
          {XBD_TRAINED_DISASTER_CLUSTERS.map((c) => {
            const isSelected = activeDisasterId === c.id;
            return (
              <button
                key={c.id}
                type="button"
                onClick={() => handleJumpToDisaster(c)}
                className={`shrink-0 flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium transition ${
                  isSelected
                    ? "bg-slate-900 text-white shadow-sm ring-1 ring-slate-700"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                }`}
                title={`Jump to ${c.name} (${c.region})`}
              >
                <span>{c.emoji}</span>
                <span>{c.name.split(" ")[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Right Side: GIS Layer Controls */}
        <div className="pointer-events-auto flex items-center gap-1 rounded-lg border border-slate-200 bg-white/95 p-1.5 shadow-md backdrop-blur-sm">
          <span className="hidden md:inline px-1 text-xs font-semibold text-slate-600">Layers:</span>
          <button
            type="button"
            onClick={() => setLayerMode("all")}
            className={`rounded px-2 py-0.5 text-xs font-medium transition ${
              layerMode === "all"
                ? "bg-primary text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            All
          </button>
          <button
            type="button"
            onClick={() => setLayerMode("heatmap")}
            className={`rounded px-2 py-0.5 text-xs font-medium transition ${
              layerMode === "heatmap"
                ? "bg-red-600 text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            🔥 Heatmap
          </button>
          <button
            type="button"
            onClick={() => setLayerMode("cells")}
            className={`rounded px-2 py-0.5 text-xs font-medium transition ${
              layerMode === "cells"
                ? "bg-blue-600 text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            ⬡ H3
          </button>
          <button
            type="button"
            onClick={() => setLayerMode("footprints")}
            className={`rounded px-2 py-0.5 text-xs font-medium transition ${
              layerMode === "footprints"
                ? "bg-emerald-600 text-white shadow-sm"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            🏢 Footprints
          </button>

          <div className="h-4 w-[1px] bg-slate-300 mx-1" />

          <label className="flex items-center gap-1 px-1 text-xs font-medium text-slate-700 cursor-pointer">
            <input
              type="checkbox"
              checked={showTrainedXbd}
              onChange={(e) => setShowTrainedXbd(e.target.checked)}
              className="rounded border-slate-300 text-primary focus:ring-primary h-3.5 w-3.5"
            />
            <span className="text-[11px]">xBD Data</span>
          </label>
        </div>
      </div>

      <MapContainer
        center={effectiveCenter}
        zoom={defaultZoom}
        className="h-full w-full"
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FlyTo center={internalCenter || center} />

        {/* GIS Radial Damage Heatmaps */}
        {showHeatmap &&
          heatmapFeatures.map((f) => (
            <GeoJSON
              key={f.id}
              data={f}
              style={{
                color: f.properties?.color,
                fillColor: f.properties?.color,
                fillOpacity: f.properties?.fillOpacity,
                weight: f.properties?.weight || 0,
                stroke: (f.properties?.weight || 0) > 0,
              }}
            >
              <Popup>
                <div className="text-xs">
                  <div className="font-bold text-slate-800 mb-0.5">
                    Heatmap Intensity Zone
                  </div>
                  <SeverityBadge prediction={f.properties?.prediction} />
                  {f.properties?.confidence && (
                    <div className="mt-1 text-slate-600">
                      Confidence: {(f.properties.confidence * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              </Popup>
            </GeoJSON>
          ))}

        {/* Trained xBD Satellite Tile Footprints */}
        {showFootprints &&
          trainedTileFootprints.map((tile) => {
            const props = tile.properties;
            const s = severityOf(props.prediction);
            return (
              <GeoJSON
                key={tile.id}
                data={tile}
                style={{
                  color: s.color,
                  weight: 2,
                  dashArray: "4, 4",
                  fillColor: s.color,
                  fillOpacity: 0.12,
                }}
              >
                <Popup>
                  <div className="min-w-[240px] text-xs">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-1.5 mb-1.5">
                      <span className="font-bold text-slate-800 flex items-center gap-1">
                        <span>{props.emoji}</span>
                        <span>{props.name}</span>
                      </span>
                      <SeverityBadge prediction={props.prediction} />
                    </div>

                    <div className="space-y-1 text-slate-600 text-[11px]">
                      <div>
                        <span className="text-slate-400">Region:</span> {props.region}
                      </div>
                      <div>
                        <span className="text-slate-400">Sensor:</span> Maxar WorldView (0.5m GSD)
                      </div>
                      <div className="font-mono text-[10px] bg-slate-50 p-1 rounded border border-slate-200">
                        <div className="text-emerald-700">PRE: {props.preFile}</div>
                        <div className="text-rose-700">POST: {props.postFile}</div>
                      </div>
                      <div>
                        <span className="text-slate-400">Model Confidence:</span>{" "}
                        <span className="font-semibold text-slate-800">
                          {(props.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400">Building Footprints:</span>{" "}
                        <span className="font-semibold text-slate-800">{props.count} detected</span>
                      </div>
                    </div>

                    {props.distribution && (
                      <div className="mt-2 border-t border-slate-100 pt-1.5">
                        <div className="text-[10px] font-semibold text-slate-500 mb-1">
                          xBD Damage Severity Breakdown:
                        </div>
                        <div className="flex h-2 w-full overflow-hidden rounded bg-slate-100">
                          <div style={{ width: `${props.distribution.destroyed}%` }} className="bg-[#DC2626]" title={`Destroyed: ${props.distribution.destroyed}%`} />
                          <div style={{ width: `${props.distribution.major}%` }} className="bg-[#F59E0B]" title={`Major: ${props.distribution.major}%`} />
                          <div style={{ width: `${props.distribution.minor}%` }} className="bg-[#EAB308]" title={`Minor: ${props.distribution.minor}%`} />
                          <div style={{ width: `${props.distribution.none}%` }} className="bg-[#22C55E]" title={`No Damage: ${props.distribution.none}%`} />
                        </div>
                      </div>
                    )}
                  </div>
                </Popup>
              </GeoJSON>
            );
          })}

        {/* Polygons (Assessments from Database) */}
        {showFootprints &&
          polygons.map((p) =>
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
                  <div className="min-w-[200px]">
                    <SeverityBadge prediction={p.prediction} />
                    <div className="mt-1 text-xs font-medium text-slate-700">
                      Confidence: {(p.confidence * 100).toFixed(0)}%
                    </div>
                    <div className="mt-2 flex flex-col gap-1.5">
                      <button
                        onClick={() => navigate(`/assessments/${p.id}`)}
                        className="rounded-md bg-primary px-2 py-1 text-xs font-semibold text-white hover:bg-primary/90"
                      >
                        View Full Details
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
        {showCells &&
          cells?.features?.map((f, i) => (
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

      {/* Empty State Banner (preserving test selector) */}
      {polygons.length === 0 && (!cells || cells.features?.length === 0) && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="pointer-events-auto rounded-lg bg-white/95 p-4 text-center shadow-lg border border-slate-200 max-w-sm">
            <p className="text-sm font-semibold text-slate-700">
              No assessments mapped yet
            </p>
            <p className="mt-0.5 text-xs text-slate-500">
              Upload a scan to start assessing damage.
            </p>
            <div className="mt-3 border-t border-slate-100 pt-2.5">
              <button
                type="button"
                onClick={() => handleJumpToDisaster(XBD_TRAINED_DISASTER_CLUSTERS[0])}
                className="w-full rounded bg-primary px-3 py-1.5 text-xs font-semibold text-white shadow transition hover:bg-primary/90"
              >
                🗺️ Explore Trained xBD Scenes on Map
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}