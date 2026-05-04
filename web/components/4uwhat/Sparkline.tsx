// 4UWHAt — Sparkline
// Compact SVG line chart with optional last-point dot. Used in vitals/lab rows.
// Source: assets/app/atoms.jsx (function Sparkline).

interface SparklineProps {
  data: number[];
  w?: number;
  h?: number;
  stroke?: string;
  fill?: string;
  dot?: boolean;
  strokeWidth?: number;
}

export function Sparkline({
  data,
  w = 100,
  h = 28,
  stroke = "currentColor",
  fill = "none",
  dot = true,
  strokeWidth = 1.5,
}: SparklineProps) {
  if (!data || data.length === 0) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = w / (data.length - 1 || 1);
  const pts: [number, number][] = data.map((v, i) => [
    i * stepX,
    h - ((v - min) / range) * (h - 4) - 2,
  ]);
  const d = pts
    .map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`))
    .join(" ");
  const last = pts[pts.length - 1];

  return (
    <svg
      width={w}
      height={h}
      style={{ display: "block", overflow: "visible" }}
      aria-hidden="true"
    >
      {fill !== "none" && (
        <path d={`${d} L${w},${h} L0,${h} Z`} fill={fill} opacity={0.18} />
      )}
      <path
        d={d}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {dot && <circle cx={last[0]} cy={last[1]} r={2} fill={stroke} />}
    </svg>
  );
}
