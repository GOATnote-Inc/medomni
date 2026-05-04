// 4UWHAt — Eyebrow
// Small uppercase mono label, magenta by default. Section labels everywhere.
// Source: assets/app/atoms.jsx (function Eyebrow).

import type { CSSProperties, ReactNode } from "react";

interface EyebrowProps {
  children: ReactNode;
  color?: string;
  style?: CSSProperties;
  className?: string;
}

export function Eyebrow({ children, color, style, className }: EyebrowProps) {
  return (
    <div
      className={className}
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        fontWeight: 700,
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        color: color ?? "var(--accent)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}
