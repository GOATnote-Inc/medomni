// 4UWHAt — Mono
// Inline mono micro label. Used for ranges, MRN, dates, ICD codes, deltas.
// Source: assets/app/atoms.jsx (function Mono).

import type { CSSProperties, ReactNode } from "react";

interface MonoProps {
  children: ReactNode;
  color?: string;
  size?: number;
  style?: CSSProperties;
  className?: string;
}

export function Mono({
  children,
  color = "rgba(255,255,255,0.5)",
  size = 10,
  style,
  className,
}: MonoProps) {
  return (
    <span
      className={className}
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: size,
        fontWeight: 500,
        letterSpacing: "0.06em",
        color,
        ...style,
      }}
    >
      {children}
    </span>
  );
}
