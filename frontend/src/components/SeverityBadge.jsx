import { severityOf } from "../lib/severity.js";

export default function SeverityBadge({ prediction, className = "" }) {
  const s = severityOf(prediction);
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white ${className}`}
      style={{ backgroundColor: s.color }}
      title={prediction}
    >
      {s.icon && <span aria-hidden>{s.icon}</span>}
      {s.label}
    </span>
  );
}