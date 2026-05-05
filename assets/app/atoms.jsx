// Shared atomic components for 4UWHAt medical records UI.
// All component & helper names are u4w-prefixed where ambiguous.

// ─── Tooltip ───────────────────────────────────────────────────────────────
// Hover-revealed reference info. Tasteful + on-brand: black surface, hairline
// pink top stripe, mono body. Anchored to the wrapped element.
function Tooltip({ children, label, range, source, hint, side = 'top', delay = 80 }) {
  const [open, setOpen] = React.useState(false);
  const [pos, setPos] = React.useState({ x: 0, y: 0 });
  const ref = React.useRef(null);
  const timer = React.useRef(null);

  const show = () => {
    timer.current = setTimeout(() => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      setPos({ x: r.left + r.width / 2, y: side === 'top' ? r.top : r.bottom });
      setOpen(true);
    }, delay);
  };
  const hide = () => {
    clearTimeout(timer.current);
    setOpen(false);
  };

  return (
    <React.Fragment>
      <span ref={ref}
            onMouseEnter={show} onMouseLeave={hide}
            onFocus={show} onBlur={hide}
            style={{
              display: 'inline-flex', alignItems: 'baseline',
              borderBottom: '1px dotted rgba(255,255,255,0.18)',
              cursor: 'help', gap: 'inherit',
            }}>
        {children}
      </span>
      {open && ReactDOM.createPortal(
        <div style={{
          position: 'fixed', left: pos.x, top: pos.y,
          transform: side === 'top' ? 'translate(-50%, calc(-100% - 8px))' : 'translate(-50%, 8px)',
          zIndex: 9999, pointerEvents: 'none',
          background: '#000',
          border: '1px solid var(--accent)',
          boxShadow: '0 8px 24px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,0,150,0.2)',
          padding: '10px 12px', minWidth: 180, maxWidth: 260,
          fontFamily: 'JetBrains Mono, monospace',
        }}>
          {/* top pink hairline */}
          <div style={{
            position: 'absolute', top: -1, left: 0, right: 0, height: 2,
            background: 'var(--accent)',
          }}/>
          {label && (
            <div style={{
              fontSize: 9, fontWeight: 700, letterSpacing: '0.16em',
              color: 'var(--accent)', textTransform: 'uppercase',
              marginBottom: 6,
            }}>{label}</div>
          )}
          {range && (
            <div style={{ fontSize: 12, color: '#fff', fontWeight: 500, marginBottom: hint ? 4 : 0 }}>
              {range}
            </div>
          )}
          {hint && (
            <div style={{ fontSize: 10.5, color: 'rgba(255,255,255,0.65)', lineHeight: 1.4 }}>
              {hint}
            </div>
          )}
          {source && (
            <div style={{
              marginTop: 6, paddingTop: 6,
              borderTop: '1px solid rgba(255,255,255,0.08)',
              fontSize: 9, letterSpacing: '0.12em', textTransform: 'uppercase',
              color: 'rgba(255,255,255,0.4)',
            }}>{source}</div>
          )}
        </div>,
        document.body
      )}
    </React.Fragment>
  );
}

// ─── Sparkline ──────────────────────────────────────────────────────────────
function Sparkline({ data, w = 100, h = 28, stroke = 'currentColor', fill = 'none', dot = true, strokeWidth = 1.5 }) {
  if (!data || data.length === 0) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const stepX = w / (data.length - 1 || 1);
  const pts = data.map((v, i) => [i * stepX, h - ((v - min) / range) * (h - 4) - 2]);
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const last = pts[pts.length - 1];
  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      {fill !== 'none' && (
        <path d={`${d} L${w},${h} L0,${h} Z`} fill={fill} opacity="0.18"/>
      )}
      <path d={d} fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"/>
      {dot && <circle cx={last[0]} cy={last[1]} r="2" fill={stroke}/>}
    </svg>
  );
}

// ─── Trend chart with dots + range band (for lab detail) ────────────────────
function TrendChart({ data, dates, range, w = 360, h = 100, accent = '#ff0096' }) {
  if (!data || data.length === 0) return null;
  const min = Math.min(...data) * 0.9;
  const max = Math.max(...data) * 1.1;
  const r = max - min || 1;
  const stepX = (w - 24) / (data.length - 1 || 1);
  const Y = (v) => h - ((v - min) / r) * (h - 16) - 10;
  const pts = data.map((v, i) => [12 + i * stepX, Y(v)]);
  const d = pts.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');

  return (
    <svg width={w} height={h + 24} style={{ display: 'block' }}>
      {/* gridlines */}
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
        <line key={i} x1="0" x2={w} y1={h * t + 0.5} y2={h * t + 0.5}
              stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      ))}
      {/* trend */}
      <path d={d} fill="none" stroke={accent} strokeWidth="1.75" strokeLinecap="round"/>
      {pts.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={i === pts.length - 1 ? 3.5 : 2.25}
                fill={i === pts.length - 1 ? accent : '#000'}
                stroke={accent} strokeWidth="1.5"/>
      ))}
      {/* dates */}
      {dates && dates.map((d, i) => (
        <text key={i} x={pts[i][0]} y={h + 16} fontFamily="JetBrains Mono, monospace"
              fontSize="9" fill="rgba(255,255,255,0.4)" textAnchor="middle"
              letterSpacing="0.08em" textTransform="uppercase">
          {d}
        </text>
      ))}
    </svg>
  );
}

// ─── Eyebrow ───────────────────────────────────────────────────────────────
function Eyebrow({ children, color, style }) {
  return (
    <div style={{
      fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fontWeight: 700,
      letterSpacing: '0.16em', textTransform: 'uppercase',
      color: color || 'var(--accent)', ...style,
    }}>{children}</div>
  );
}

// ─── Mono micro label ──────────────────────────────────────────────────────
function Mono({ children, color = 'rgba(255,255,255,0.5)', size = 10, style }) {
  return (
    <span style={{
      fontFamily: 'JetBrains Mono, monospace', fontSize: size, fontWeight: 500,
      letterSpacing: '0.06em', color, ...style,
    }}>{children}</span>
  );
}

// ─── Pink stripe (signature motif) ─────────────────────────────────────────
function Stripe({ width = 32, height = 3, style }) {
  return <div style={{ width, height, background: 'var(--accent)', ...style }}/>;
}

// ─── Status dot ─────────────────────────────────────────────────────────────
function Dot({ color = 'var(--accent)', size = 6, glow = false, style }) {
  return (
    <span style={{
      display: 'inline-block', width: size, height: size, borderRadius: 999,
      background: color,
      boxShadow: glow ? `0 0 10px ${color}` : 'none',
      ...style,
    }}/>
  );
}

// ─── Pill / tag ─────────────────────────────────────────────────────────────
function Tag({ children, accent = false, color, style }) {
  return (
    <span style={{
      fontFamily: 'JetBrains Mono, monospace', fontSize: 9.5, fontWeight: 700,
      letterSpacing: '0.12em', textTransform: 'uppercase',
      padding: '3px 7px', borderRadius: 2,
      border: '1px solid',
      borderColor: accent ? 'var(--accent)' : (color || 'rgba(255,255,255,0.18)'),
      color: accent ? 'var(--accent)' : (color || 'rgba(255,255,255,0.7)'),
      background: accent ? 'rgba(255,0,150,0.08)' : 'transparent',
      lineHeight: 1, display: 'inline-flex', alignItems: 'center', gap: 5,
      ...style,
    }}>{children}</span>
  );
}

// ─── Avatar (initials) ──────────────────────────────────────────────────────
function Avatar({ initials, size = 28, accent = false, online = false }) {
  return (
    <div style={{
      width: size, height: size, position: 'relative',
      background: accent ? 'var(--accent)' : 'transparent',
      border: '1px solid', borderColor: accent ? 'var(--accent)' : 'rgba(255,255,255,0.2)',
      color: accent ? '#000' : 'rgba(255,255,255,0.85)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: size * 0.36,
      letterSpacing: '0.04em',
    }}>
      {initials}
      {online && (
        <span style={{
          position: 'absolute', bottom: -1, right: -1, width: 6, height: 6, borderRadius: 999,
          background: 'var(--accent)', boxShadow: '0 0 8px var(--accent)',
          border: '1px solid #000',
        }}/>
      )}
    </div>
  );
}

// ─── Range bar (for lab values w/ reference range) ──────────────────────────
function RangeBar({ value, low, high, hardMin, hardMax, w = 120, accent = '#ff0096' }) {
  const min = hardMin ?? (low - (high - low) * 0.5);
  const max = hardMax ?? (high + (high - low) * 0.5);
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const lowT = (low - min) / (max - min);
  const highT = (high - min) / (max - min);
  return (
    <div style={{ width: w, position: 'relative', height: 18 }}>
      <div style={{
        position: 'absolute', top: 8, left: 0, right: 0, height: 2,
        background: 'rgba(255,255,255,0.1)',
      }}/>
      <div style={{
        position: 'absolute', top: 8, left: `${lowT * 100}%`, width: `${(highT - lowT) * 100}%`,
        height: 2, background: 'rgba(255,255,255,0.35)',
      }}/>
      <div style={{
        position: 'absolute', top: 4, left: `calc(${t * 100}% - 5px)`,
        width: 10, height: 10, background: accent,
        boxShadow: `0 0 12px ${accent}`,
      }}/>
    </div>
  );
}

// ─── Big numeric stat ──────────────────────────────────────────────────────
function Stat({ value, unit, label, delta, deltaGood = true, big = false }) {
  return (
    <div>
      <div style={{
        fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700,
        fontSize: big ? 64 : 40, lineHeight: 1, letterSpacing: '-0.03em',
        color: '#fff', display: 'flex', alignItems: 'baseline', gap: 6,
      }}>
        <span>{value}</span>
        {unit && <Mono size={big ? 16 : 12} color="rgba(255,255,255,0.45)" style={{ fontWeight: 500 }}>{unit}</Mono>}
      </div>
      {(label || delta) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
          {label && <Mono>{label}</Mono>}
          {delta && (
            <Mono color={deltaGood ? 'var(--accent)' : '#ffaa00'}>
              {delta}
            </Mono>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Prism mark (small inline SVG icon) ────────────────────────────────────
function PrismMark({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="square" strokeLinejoin="miter">
      <path d="M3 20 L12 4 L21 20 Z"/>
      <path d="M12 4 L12 20" opacity="0.5"/>
    </svg>
  );
}

// ─── Wordmark ──────────────────────────────────────────────────────────────
function Wordmark({ size = 14, color = '#fff' }) {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, color }}>
      <PrismMark size={size + 4} color="var(--accent)"/>
      <span style={{
        fontFamily: 'Space Grotesk, sans-serif', fontWeight: 700,
        fontSize: size, letterSpacing: '-0.02em',
      }}>4UWHAt</span>
    </div>
  );
}

Object.assign(window, {
  Sparkline, TrendChart, Eyebrow, Mono, Stripe, Dot, Tag, Avatar,
  RangeBar, Stat, PrismMark, Wordmark, Tooltip,
});
