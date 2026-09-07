// Phase 1: placeholder — disabled until Phase 5 wires real IPFS + on-chain verification.
export default function ChainBadge({ chainVerified, disabled = true, href }) {
  const base =
    "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold";
  if (disabled) {
    return (
      <span
        className={`${base} bg-slate-200 text-slate-400 cursor-not-allowed`}
        title="On-chain verification arrives in Phase 5"
      >
        🔗 Verify on-chain
      </span>
    );
  }
  const content = (
    <>
      <span aria-hidden>{chainVerified ? "✓" : ""}</span>
      {chainVerified ? "Verified on-chain" : "Verify on-chain"}
    </>
  );
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`${base} text-white`}
      style={{ backgroundColor: "#0EA5E9" }}
    >
      {content}
    </a>
  );
}