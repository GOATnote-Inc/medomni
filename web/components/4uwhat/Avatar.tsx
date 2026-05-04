// 4UWHAt — Avatar
// Square (no border-radius) initials avatar. Variants: outline / accent.
// Optional online indicator: 6px magenta dot + glow at lower-right.
// Source: assets/app/atoms.jsx (function Avatar).

interface AvatarProps {
  initials: string;
  size?: number;
  accent?: boolean;
  online?: boolean;
  className?: string;
}

export function Avatar({
  initials,
  size = 28,
  accent = false,
  online = false,
  className,
}: AvatarProps) {
  return (
    <div
      className={className}
      style={{
        width: size,
        height: size,
        position: "relative",
        background: accent ? "var(--accent)" : "transparent",
        border: "1px solid",
        borderColor: accent ? "var(--accent)" : "rgba(255,255,255,0.2)",
        color: accent ? "#000" : "rgba(255,255,255,0.85)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "var(--font-mono)",
        fontWeight: 700,
        fontSize: size * 0.36,
        letterSpacing: "0.04em",
      }}
    >
      {initials}
      {online && (
        <span
          aria-hidden="true"
          style={{
            position: "absolute",
            bottom: -1,
            right: -1,
            width: 6,
            height: 6,
            borderRadius: 999,
            background: "var(--accent)",
            boxShadow: "0 0 8px var(--accent)",
            border: "1px solid #000",
          }}
        />
      )}
    </div>
  );
}
