// 4UWHAt — PrismMark
// Inline SVG brand icon. Triangle with a vertical refraction line.
// Source: assets/app/atoms.jsx (function PrismMark).

import type { CSSProperties } from "react";

interface PrismMarkProps {
  size?: number;
  color?: string;
  style?: CSSProperties;
  className?: string;
}

export function PrismMark({
  size = 18,
  color = "currentColor",
  style,
  className,
}: PrismMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth={2}
      strokeLinecap="square"
      strokeLinejoin="miter"
      style={style}
      className={className}
      aria-hidden="true"
    >
      <path d="M3 20 L12 4 L21 20 Z" />
      <path d="M12 4 L12 20" opacity={0.5} />
    </svg>
  );
}
