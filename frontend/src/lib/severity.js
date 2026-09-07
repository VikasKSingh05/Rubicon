// Global Design System — severity palette + verified-on-chain accent.
// Never rely on color alone: every badge pairs a color with an icon and a text label.

export const SEVERITY = {
  none: { label: "None / Minimal", color: "#22C55E", icon: null },
  moderate: { label: "Moderate", color: "#F59E0B", icon: "⚠" },
  severe: { label: "Severe", color: "#DC2626", icon: "⚠" },
};

export const VERIFY_BADGE = {
  color: "#0EA5E9",
  label: "Verified on-chain",
  icon: "✓",
};

export function severityOf(prediction) {
  if (prediction === "Severe Collapse") return SEVERITY.severe;
  if (prediction === "Moderate") return SEVERITY.moderate;
  return SEVERITY.none;
}