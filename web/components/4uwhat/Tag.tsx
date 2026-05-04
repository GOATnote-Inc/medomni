// 4UWHAt — Tag
// Mono uppercase pill with a 1px border. Variants: default / accent / color-prop.
// Source: assets/app/atoms.jsx (function Tag).

import type { CSSProperties, ReactNode } from "react";

interface TagProps {
  children: ReactNode;
  accent?: boolean;
  color?: string;
  style?: CSSProperties;
  className?: string;
}

export function Tag({
  children,
  accent = false,
  color,
  style,
  className,
}: TagProps) {
  const borderColor = accent
    ? "var(--accent)"
    : color ?? "rgba(255,255,255,0.18)";
  const fg = accent ? "var(--accent)" : color ?? "rgba(255,255,255,0.7)";
  const bg = accent ? "rgba(255,0,150,0.08)" : "transparent";

  return (
    <span
      className={className}
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: 9.5,
        fontWeight: 700,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        padding: "3px 7px",
        borderRadius: 2,
        border: "1px solid",
        borderColor,
        color: fg,
        background: bg,
        lineHeight: 1,
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        ...style,
      }}
    >
      {children}
    </span>
  );
}
