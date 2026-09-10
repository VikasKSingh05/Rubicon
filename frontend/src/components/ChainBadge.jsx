// Phase 5 — live on-chain verification badge. Points at the Amoy block explorer
// when a transaction hash exists; renders as a plain state chip otherwise.
export default function ChainBadge({ chainVerified, disabled = false, href }) {
  const base =
    "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-semibold";

  if (disabled) {
    return (
      <span
        className={`${base} bg-slate-200 text-slate-400 cursor-not-allowed`}
        title="On-chain verification is disabled for this view"
      >
        🔗 Verify on-chain
      </span>
    );
  }

  const content = (
    <>
      <span aria-hidden>{chainVerified ? "✓" : "🔗"}</span>
      {chainVerified ? "Verified on-chain" : "Verify on-chain"}
    </>
  );

  if (!href) {
    return <span className={`${base} bg-slate-200 text-slate-500`}>{content}</span>;
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`${base} text-white`}
      style={{ backgroundColor: "#0EA5E9" }}
      title="Open transaction on Polygon Amoy explorer"
    >
      {content}
    </a>
  );
}