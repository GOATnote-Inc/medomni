// 4UWHAt — TrendChart
// SVG line chart with gridlines (0/25/50/75/100% at rgba(255,255,255,0.06)),
// per-point dots, and an emphasized last-point dot.
// Used in the mobile lab-detail 24-month trend card.
// Source: assets/app/atoms.jsx (function TrendChart).

interface TrendChartProps {
  data: number[];
  dates?: string[];
  /** Optional reference range (currently unused — preserved for parity with source). */
  range?: { low?: number; high?: number };
  w?: number;
  h?: number;
  accent?: string;
}

export function TrendChart({
  data,
  dates,
  w = 360,
  h = 100,
  accent = "#ff0096",
}: TrendChartProps) {
  if (!data || data.length === 0) return null;

  const min = Math.min(...data) * 0.9;
  const max = Math.max(...data) * 1.1;
  const r = max - min || 1;
  const stepX = (w - 24) / (data.length - 1 || 1);
  const Y = (v: number) => h - ((v - min) / r) * (h - 16) - 10;
  const pts: [number, number][] = data.map((v, i) => [12 + i * stepX, Y(v)]);
  const d = pts
    .map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`))
    .join(" ");

  return (
    <svg
      width={w}
      height={h + 24}
      style={{ display: "block" }}
      aria-hidden="true"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
        <line
          key={i}
          x1={0}
          x2={w}
          y1={h * t + 0.5}
          y2={h * t + 0.5}
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={1}
        />
      ))}
      <path
        d={d}
        fill="none"
        stroke={accent}
        strokeWidth={1.75}
        strokeLinecap="round"
      />
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={p[1]}
          r={i === pts.length - 1 ? 3.5 : 2.25}
          fill={i === pts.length - 1 ? accent : "#000"}
          stroke={accent}
          strokeWidth={1.5}
        />
      ))}
      {dates &&
        dates.map((label, i) => (
          <text
            key={i}
            x={pts[i][0]}
            y={h + 16}
            fontFamily="var(--font-mono)"
            fontSize={9}
            fill="rgba(255,255,255,0.4)"
            textAnchor="middle"
            letterSpacing="0.08em"
          >
            {label}
          </text>
        ))}
    </svg>
  );
}
