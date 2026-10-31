// Global Design System — severity palette + verified-on-chain accent.
// Supports both xBD dual-view damage classification (No Damage, Minor, Major, Destroyed)
// and legacy disaster classifications.

export const SEVERITY = {
  none: { label: "No Damage", color: "#22C55E", icon: null },
  minor: { label: "Minor Damage", color: "#EAB308", icon: "⚡" },
  moderate: { label: "Major Damage", color: "#F59E0B", icon: "⚠" },
  severe: { label: "Destroyed", color: "#DC2626", icon: "⚠" },
};

export const VERIFY_BADGE = {
  color: "#0EA5E9",
  label: "Verified on-chain",
  icon: "✓",
};

export function severityOf(prediction) {
  if (!prediction) return SEVERITY.none;
  const p = String(prediction).toLowerCase();
  if (p === "severe collapse" || p === "destroyed" || p.includes("destroy")) return SEVERITY.severe;
  if (p === "moderate" || p === "major damage" || p.includes("major")) return SEVERITY.moderate;
  if (p === "minor damage" || p.includes("minor")) return SEVERITY.minor;
  return SEVERITY.none;
}