// 4UWHAt — Wordmark
// PrismMark + "4UWHAt" lockup. Used in nav rail and footer.
// Source: assets/app/atoms.jsx (function Wordmark).

import { PrismMark } from "./PrismMark";

interface WordmarkProps {
  size?: number;
  color?: string;
  className?: string;
}

export function Wordmark({
  size = 14,
  color = "#fff",
  className,
}: WordmarkProps) {
  return (
    <div
      className={className}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        color,
      }}
    >
      <PrismMark size={size + 4} color="var(--accent)" />
      <span
        style={{
          fontFamily: "var(--font-display)",
          fontWeight: 700,
          fontSize: size,
          letterSpacing: "-0.02em",
        }}
      >
        4UWHAt
      </span>
    </div>
  );
}
