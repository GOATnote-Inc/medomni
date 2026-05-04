// 4UWHAt — Stripe
// Signature pink hairline. Used as section accent / decorative motif.
// Source: assets/app/atoms.jsx (function Stripe).

import type { CSSProperties } from "react";

interface StripeProps {
  width?: number | string;
  height?: number | string;
  style?: CSSProperties;
  className?: string;
}

export function Stripe({
  width = 32,
  height = 3,
  style,
  className,
}: StripeProps) {
  return (
    <div
      aria-hidden="true"
      className={className}
      style={{
        width,
        height,
        background: "var(--accent)",
        ...style,
      }}
    />
  );
}
