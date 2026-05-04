// 4UWHAt — RangeBar
// Horizontal bar showing a value within a reference range and hard min/max.
// Used in the mobile lab-detail screen (LDL 92, target 50–100).
// Source: assets/app/atoms.jsx (function RangeBar).

interface RangeBarProps {
  value: number;
  low: number;
  high: number;
  hardMin?: number;
  hardMax?: number;
  w?: number;
  accent?: string;
}

export function RangeBar({
  value,
  low,
  high,
  hardMin,
  hardMax,
  w = 120,
  accent = "#ff0096",
}: RangeBarProps) {
  const min = hardMin ?? low - (high - low) * 0.5;
  const max = hardMax ?? high + (high - low) * 0.5;
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const lowT = (low - min) / (max - min);
  const highT = (high - min) / (max - min);

  return (
    <div style={{ width: w, position: "relative", height: 18 }}>
      <div
        style={{
          position: "absolute",
          top: 8,
          left: 0,
          right: 0,
          height: 2,
          background: "rgba(255,255,255,0.1)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 8,
          left: `${lowT * 100}%`,
          width: `${(highT - lowT) * 100}%`,
          height: 2,
          background: "rgba(255,255,255,0.35)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: 4,
          left: `calc(${t * 100}% - 5px)`,
          width: 10,
          height: 10,
          background: accent,
          boxShadow: `0 0 12px ${accent}`,
        }}
      />
    </div>
  );
}
