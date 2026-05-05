// Variation B — BODY CONSOLE
// Bolder, more cinematic. A giant prism refraction mark on the left, with
// the body's "spectrum" of signals fanning out. Single hero metric with a
// large display number. AI ribbon under the hero. Right side: a cleaner
// vertical timeline + collapsed modules. Records-as-spectrum, not as table.

const BodyConsole = () => {
  const [tab, setTab] = React.useState('signals');

  const TABS = [
    { id: 'signals',   label: 'Signals' },
    { id: 'timeline',  label: 'Timeline' },
    { id: 'systems',   label: 'Systems' },
    { id: 'genome',    label: 'Genome' },
    { id: 'sharing',   label: 'Sharing' },
  ];

  // 6 signal "wavelengths" — the spectrum metaphor
  const SPECTRUM = [
    { k: 'CARDIO',   label: 'Cardiovascular',  status: 'good',   value: 'LDL 92 · HRV 62',   detail: 'Statin working. HRV up 8ms in 30d.' },
    { k: 'METAB',    label: 'Metabolic',       status: 'good',   value: 'A1c 5.2',            detail: 'Stable for 18 months.' },
    { k: 'RESP',     label: 'Respiratory',     status: 'watch',  value: 'PEF 480 L/min',      detail: 'Asthma controlled. 0 rescue uses in 60d.' },
    { k: 'IMMUNE',   label: 'Immune',          status: 'good',   value: 'hs-CRP 0.6',         detail: 'Inflammation low.' },
    { k: 'NUTRI',    label: 'Nutritional',     status: 'low',    value: 'Vit D 28',           detail: 'Below 30 ng/mL. Supplement up to 4000 IU?' },
    { k: 'SLEEP',    label: 'Sleep · Recovery',status: 'good',   value: '7h 14m · 88 score',  detail: 'Best 30 days in 2 years.' },
  ];

  return (
    <div style={{
      width: 1440, height: 900, background: '#000', color: '#fff',
      display: 'grid', gridTemplateColumns: '1fr 420px',
      fontFamily: 'Space Grotesk, sans-serif', overflow: 'hidden',
      border: '1px solid #1f1f1f', position: 'relative',
    }}>
      {/* ── HEADER ──────────────────────────────────────────────────── */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 56,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 32px', borderBottom: '1px solid #1f1f1f', zIndex: 5, background: '#000',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
          <Wordmark size={13}/>
          <span style={{ width: 1, height: 18, background: '#1f1f1f' }}/>
          <Mono color="rgba(255,255,255,0.55)">{PATIENT.name.toUpperCase()} · {PATIENT.mrn}</Mono>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: 'transparent', border: 'none', cursor: 'pointer',
              padding: '8px 14px',
              fontFamily: 'JetBrains Mono', fontSize: 10.5, fontWeight: 700,
              letterSpacing: '0.14em', textTransform: 'uppercase',
              color: tab === t.id ? 'var(--accent)' : 'rgba(255,255,255,0.4)',
              borderBottom: '2px solid', borderBottomColor: tab === t.id ? 'var(--accent)' : 'transparent',
            }}>{t.label}</button>
          ))}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <Mono size={9}><Dot color="var(--accent)" glow={true} size={5}/> &nbsp;LIVE · APR 22</Mono>
          <Avatar initials="MO" accent={true} size={28}/>
        </div>
      </div>

      {/* ── MAIN ─────────────────────────────────────────────────────── */}
      <main style={{ padding: '88px 32px 24px 32px', overflow: 'auto', position: 'relative' }}>
        {/* HERO — giant prism + display number */}
        <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 32, marginBottom: 28 }}>
          {/* Left: BIG prism mark with refraction lines */}
          <div style={{ position: 'relative', height: 360 }}>
            <PrismHero/>
          </div>

          {/* Right: hero stat + system summary */}
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <Eyebrow style={{ marginBottom: 14 }}>YOUR BODY · APR 22, 2026</Eyebrow>
              <div style={{
                fontFamily: 'Space Grotesk', fontWeight: 700,
                fontSize: 96, lineHeight: 0.92, letterSpacing: '-0.045em',
                color: '#fff',
              }}>
                Mostly<br/>
                <span style={{ color: 'var(--accent)' }}>resolved.</span>
              </div>
              <div style={{
                marginTop: 24, maxWidth: 540,
                fontSize: 18, lineHeight: 1.5, color: 'rgba(255,255,255,0.7)',
              }}>
                5 of 6 systems are in range. Vitamin D is low — first time
                trending up in 18 months. Statin keeps working. Resting HR
                hit a 5-year minimum on Tuesday.
              </div>
            </div>

            {/* Hero stats row */}
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 0,
              marginTop: 28, borderTop: '1px solid rgba(255,255,255,0.08)',
              paddingTop: 18,
            }}>
              <HeroStat n="42" u="DAYS" l="STREAK · MEDS" tip={{ label: 'ADHERENCE STREAK', range: '≥95% of doses on time', hint: '42 consecutive days hitting all scheduled meds.', source: 'Pillpack reminders · Apple Health' }}/>
              <HeroStat n="58" u="BPM"  l="RESTING HR" accent tip={{ label: 'RESTING HR', range: VITALS.hr.range, hint: VITALS.hr.hint, source: VITALS.hr.source }}/>
              <HeroStat n="92" u="MG/DL" l="LDL · TARGET" tip={{ label: 'LDL CHOLESTEROL', range: 'Reference: <100 mg/dL · on statin <70', hint: 'Down 50 points in 24 months. Statin working.', source: 'Quest · Apr 22' }}/>
              <HeroStat n="0"  u=""     l="RESCUE INHALER" tip={{ label: 'ALBUTEROL USES', range: 'Goal: <2 puffs/week', hint: 'Zero rescue uses in 60 days. Asthma well-controlled.', source: 'Inhaler tracker · 60d' }}/>
            </div>
          </div>
        </div>

        {/* AI RIBBON — full-bleed, command-bar feel */}
        <div style={{
          background: '#0a0a0a',
          border: '1px solid rgba(255,255,255,0.08)',
          borderLeft: '3px solid var(--accent)',
          padding: '14px 20px', marginBottom: 28,
          display: 'flex', alignItems: 'center', gap: 18,
        }}>
          <PrismMark size={20} color="var(--accent)"/>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 18 }}>
            <Mono color="var(--accent)" size={10}>ASK YOUR RECORD</Mono>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
              "why is my LDL trending down?"<span className="caret" style={{ color: 'var(--accent)' }}>_</span>
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            {AI_SUGGESTIONS.slice(0, 2).map((s, i) => (
              <button key={i} style={{
                background: 'transparent', border: '1px solid rgba(255,255,255,0.12)',
                color: 'rgba(255,255,255,0.7)',
                padding: '6px 12px', fontSize: 11, fontFamily: 'inherit',
                cursor: 'pointer', whiteSpace: 'nowrap',
              }}>
                {s}
              </button>
            ))}
          </div>
          <Mono size={9} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <span style={kbdB}>⌘</span><span style={kbdB}>K</span>
          </Mono>
        </div>

        {/* SPECTRUM — 6 system bars */}
        <Eyebrow style={{ marginBottom: 12 }}>SYSTEMS · 6 WAVELENGTHS</Eyebrow>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {SPECTRUM.map((s, i) => <SpectrumRow key={i} s={s} index={i}/>)}
        </div>
      </main>

      {/* ── RIGHT: TIMELINE + DENSE MODULES ─────────────────────────── */}
      <aside style={{
        borderLeft: '1px solid #1f1f1f',
        padding: '88px 28px 24px 28px',
        overflow: 'auto', background: '#040404',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <Eyebrow>TIMELINE</Eyebrow>
          <Mono>{TIMELINE.length} EVENTS · 6 MO</Mono>
        </div>

        {/* Timeline */}
        <div style={{ position: 'relative', paddingLeft: 16, marginBottom: 32 }}>
          <div style={{
            position: 'absolute', left: 4, top: 8, bottom: 8, width: 1,
            background: 'rgba(255,255,255,0.08)',
          }}/>
          {TIMELINE.map((e, i) => (
            <div key={i} style={{ position: 'relative', paddingBottom: 18 }}>
              <div style={{
                position: 'absolute', left: -16, top: 4,
                width: 9, height: 9, background: e.kind === 'visit' ? 'var(--accent)' : '#000',
                border: '1px solid', borderColor: e.kind === 'visit' ? 'var(--accent)' : 'rgba(255,255,255,0.3)',
                boxShadow: e.kind === 'visit' ? '0 0 10px rgba(255,0,150,0.6)' : 'none',
              }}/>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 3 }}>
                <Mono size={9} color="rgba(255,255,255,0.45)">{e.date}</Mono>
                <Tag color={e.kind === 'visit' ? 'var(--accent)' : 'rgba(255,255,255,0.25)'}
                     style={{ fontSize: 8.5, padding: '2px 5px' }}>
                  {e.tag}
                </Tag>
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, lineHeight: 1.3 }}>{e.title}</div>
              <Mono size={9} style={{ marginTop: 2, display: 'block' }}>{e.who} · {e.loc}</Mono>
            </div>
          ))}
        </div>

        {/* Quick modules — collapsed */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <CollapsedModule title="MEDICATIONS" count="4 ACTIVE">
            {MEDS.slice(0, 4).map(m => (
              <div key={m.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                <span style={{ fontSize: 11.5 }}>{m.name}</span>
                <Mono size={9}>{m.dose} · {m.freq.toUpperCase()}</Mono>
              </div>
            ))}
          </CollapsedModule>

          <CollapsedModule title="IMAGING" count="3 STUDIES">
            {IMAGING.map(i => (
              <div key={i.id} style={{ padding: '4px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 11.5, fontWeight: 500 }}>{i.kind} · {i.region}</span>
                  <Mono size={9}>{i.date.slice(0, 7)}</Mono>
                </div>
                <Mono size={9} color="rgba(255,255,255,0.5)" style={{ marginTop: 2 }}>{i.read}</Mono>
              </div>
            ))}
          </CollapsedModule>

          <CollapsedModule title="CARE TEAM" count={`${CARE_TEAM.length} PEOPLE`}>
            <div style={{ display: 'flex', gap: 8, paddingTop: 4, flexWrap: 'wrap' }}>
              {CARE_TEAM.map(c => (
                <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <Avatar initials={c.avatar} size={22} online={c.online}/>
                  <Mono size={9} color="rgba(255,255,255,0.7)">{c.name.replace('Dr. ', '')}</Mono>
                </div>
              ))}
            </div>
          </CollapsedModule>
        </div>
      </aside>
    </div>
  );
};

// ── Big prism hero ──────────────────────────────────────────────────────────
const PrismHero = () => {
  // a chunky outline prism on the left, with 6 spectral lines refracting
  // out to the right (like a real prism splitting light).
  return (
    <svg viewBox="0 0 380 360" style={{ width: '100%', height: '100%' }}>
      {/* Beam in — white */}
      <line x1="0" y1="180" x2="100" y2="180" stroke="rgba(255,255,255,0.65)" strokeWidth="2"/>

      {/* Prism triangle */}
      <path d="M 110 60 L 110 300 L 270 180 Z"
            fill="none" stroke="#fff" strokeWidth="1.5" strokeLinejoin="miter"/>
      {/* prism inner shimmer */}
      <path d="M 110 60 L 110 300 L 270 180 Z"
            fill="rgba(255,255,255,0.025)"/>

      {/* center axis through prism */}
      <line x1="110" y1="180" x2="270" y2="180"
            stroke="rgba(255,0,150,0.18)" strokeWidth="1" strokeDasharray="2 4"/>

      {/* refracted spectrum out, in pink shades */}
      {[
        { y: 80,  color: '#ff66c4', label: 'CARDIO' },
        { y: 130, color: '#ff44b3', label: 'METAB' },
        { y: 170, color: '#ff0096', label: 'RESP' },
        { y: 210, color: '#ff0096', label: 'IMMUNE' },
        { y: 250, color: '#cc0078', label: 'NUTRI' },
        { y: 290, color: '#99005a', label: 'SLEEP' },
      ].map((b, i) => (
        <g key={i}>
          <line x1="270" y1="180" x2="380" y2={b.y}
                stroke={b.color} strokeWidth="1.5" opacity={0.85}/>
          <circle cx="380" cy={b.y} r="2.5" fill={b.color}/>
          <text x="372" y={b.y - 5}
                fontFamily="JetBrains Mono" fontSize="8" fontWeight="700"
                letterSpacing="0.16em" fill={b.color} textAnchor="end">
            {b.label}
          </text>
        </g>
      ))}

      {/* mono label, top */}
      <text x="0" y="20" fontFamily="JetBrains Mono" fontSize="9" fontWeight="700"
            letterSpacing="0.18em" fill="rgba(255,255,255,0.4)">
        INPUT · UNIFIED
      </text>
      <text x="380" y="20" fontFamily="JetBrains Mono" fontSize="9" fontWeight="700"
            letterSpacing="0.18em" fill="rgba(255,255,255,0.4)" textAnchor="end">
        OUTPUT · 6 SYSTEMS
      </text>
    </svg>
  );
};

// ── Hero stat (under big headline) ──────────────────────────────────────────
const HeroStat = ({ n, u, l, accent, tip }) => {
  const inner = (
    <span style={{
      fontFamily: 'Space Grotesk', fontSize: 40, fontWeight: 700,
      letterSpacing: '-0.03em', lineHeight: 1,
      color: accent ? 'var(--accent)' : '#fff',
      display: 'inline-flex', alignItems: 'baseline', gap: 4,
    }}>
      {n}{u && <Mono size={11} style={{ fontWeight: 500 }}>{u}</Mono>}
    </span>
  );
  return (
    <div style={{ paddingRight: 16 }}>
      {tip ? <Tooltip {...tip}>{inner}</Tooltip> : inner}
      <Mono size={9} style={{ marginTop: 6, display: 'block' }}>{l}</Mono>
    </div>
  );
};

// ── Spectrum row (one of 6 system cards) ────────────────────────────────────
const SpectrumRow = ({ s, index }) => {
  const color =
    s.status === 'low'   ? '#ffaa00' :
    s.status === 'watch' ? '#66aaff' :
                           'var(--accent)';
  return (
    <div style={{
      background: '#0e0e0e', border: '1px solid rgba(255,255,255,0.07)',
      padding: 16, position: 'relative', overflow: 'hidden',
    }}>
      {/* index */}
      <Mono size={9} style={{ position: 'absolute', top: 12, right: 14, color: 'rgba(255,255,255,0.25)' }}>
        0{index + 1}/06
      </Mono>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <Dot color={color} glow={true} size={6}/>
        <Mono size={10} color={color}>{s.k}</Mono>
        <span style={{ color: 'rgba(255,255,255,0.2)' }}>·</span>
        <Mono size={10}>{s.label.toUpperCase()}</Mono>
      </div>
      <div style={{
        fontFamily: 'Space Grotesk', fontSize: 22, fontWeight: 600,
        letterSpacing: '-0.015em', lineHeight: 1.2,
      }}>
        <Tooltip label={s.label} range={s.detail} hint={`Status: ${s.status.toUpperCase()}`}>
          <span>{s.value}</span>
        </Tooltip>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: 'rgba(255,255,255,0.55)', lineHeight: 1.4 }}>
        {s.detail}
      </div>
      {/* bottom mini-spectrum */}
      <div style={{ display: 'flex', gap: 2, marginTop: 14, height: 4 }}>
        {Array.from({ length: 24 }).map((_, i) => (
          <div key={i} style={{
            flex: 1,
            background: i < (s.status === 'good' ? 22 : s.status === 'watch' ? 16 : 11) ? color : 'rgba(255,255,255,0.06)',
            opacity: i < (s.status === 'good' ? 22 : s.status === 'watch' ? 16 : 11) ? (0.4 + (i / 24) * 0.6) : 1,
          }}/>
        ))}
      </div>
    </div>
  );
};

// ── Right rail collapsed module ─────────────────────────────────────────────
const CollapsedModule = ({ title, count, children }) => (
  <div style={{ background: '#0a0a0a', border: '1px solid rgba(255,255,255,0.07)', padding: 14 }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
      <Eyebrow>{title}</Eyebrow>
      <Mono>{count}</Mono>
    </div>
    {children}
  </div>
);

const kbdB = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  fontFamily: 'JetBrains Mono', fontSize: 9, fontWeight: 700,
  padding: '2px 5px', border: '1px solid rgba(255,255,255,0.18)',
  color: 'rgba(255,255,255,0.7)', minWidth: 14,
};

window.BodyConsole = BodyConsole;
