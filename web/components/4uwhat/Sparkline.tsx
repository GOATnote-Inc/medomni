// 4UWHAt — Sparkline
// Compact SVG line chart with optional last-point dot. Used in vitals/lab rows.
// Source: assets/app/atoms.jsx (function Sparkline).
//
// Auto-width: the `w` prop now feeds the SVG `viewBox` only; the rendered
// element fills 100% of its parent's content box. This prevents fixed-pixel
// SVGs from escaping (or overlapping siblings in) compressed grid cells —
// e.g. when the AI rail is widened and the 6-column vitals strip narrows.
// `preserveAspectRatio="none"` lets the line stretch horizontally to fit.

interface SparklineProps {
  data: number[];
  /** ViewBox max-x. Sets the chart's intrinsic aspect ratio; render width is 100% of parent. */
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
      viewBox={`0 0 ${w} ${h}`}
      width="100%"
      height={h}
      preserveAspectRatio="none"
      style={{ display: "block", maxWidth: "100%", height: "auto", overflow: "hidden" }}
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
        // Compensate for non-uniform scaling under preserveAspectRatio="none"
        // so the stroke doesn't appear stretched horizontally.
        vectorEffect="non-scaling-stroke"
      />
      {dot && (
        <circle
          cx={last[0]}
          cy={last[1]}
          r={2}
          fill={stroke}
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}
