// Mobile — iOS home + lab detail in 4UWHAt style.

const MobileHome = () => (
  <IOSDevice dark={true} width={390} height={812}>
    <div style={{
      width: '100%', height: '100%', background: '#000', color: '#fff',
      fontFamily: 'Space Grotesk, sans-serif', overflow: 'hidden',
      display: 'flex', flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{ padding: '8px 20px 14px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Wordmark size={12}/>
        <Avatar initials="MO" accent={true} size={28}/>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '0 20px 20px' }}>
        {/* Greeting + signal */}
        <Eyebrow style={{ marginBottom: 8 }}>TODAY · APR 22</Eyebrow>
        <div style={{
          fontFamily: 'Space Grotesk', fontSize: 32, fontWeight: 700,
          letterSpacing: '-0.025em', lineHeight: 1.05, marginBottom: 6,
        }}>
          Hi Maya. <span style={{ color: 'var(--accent)' }}>Your LDL is 92.</span>
        </div>
        <div style={{ fontSize: 14, color: 'rgba(255,255,255,0.6)', lineHeight: 1.45 }}>
          50 points lower than 24 months ago. The statin is working.
        </div>

        {/* AI bar */}
        <button style={{
          marginTop: 18, width: '100%',
          background: '#0a0a0a', border: '2px solid var(--accent)',
          padding: '14px 14px', display: 'flex', alignItems: 'center', gap: 10,
          color: '#fff', fontFamily: 'inherit', textAlign: 'left',
          boxShadow: 'var(--glow-pink)',
        }}>
          <PrismMark size={18} color="var(--accent)"/>
          <Mono size={11} color="rgba(255,255,255,0.85)">ASK YOUR RECORD</Mono>
          <span style={{ marginLeft: 'auto', color: 'var(--accent)' }}>→</span>
        </button>

        {/* Vitals 2x2 */}
        <Eyebrow style={{ marginTop: 24, marginBottom: 10 }}>VITALS · 7D</Eyebrow>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {[VITALS.hr, VITALS.bp, VITALS.hrv, VITALS.sleep].map((v, i) => (
            <div key={i} style={{
              background: '#0e0e0e', border: '1px solid rgba(255,255,255,0.07)', padding: 12,
            }}>
              <Mono size={9}>{v.label.toUpperCase()}</Mono>
              <Tooltip label={v.label} range={v.range} hint={v.hint} source={v.source}>
                <span style={{ display: 'inline-flex', alignItems: 'baseline', gap: 4, marginTop: 4 }}>
                  <span style={{ fontFamily: 'Space Grotesk', fontSize: 22, fontWeight: 700, letterSpacing: '-0.02em' }}>
                    {v.value}
                  </span>
                  <Mono size={9}>{v.unit}</Mono>
                </span>
              </Tooltip>
              <div style={{ marginTop: 6, color: 'var(--accent)' }}>
                <Sparkline data={v.spark} w={140} h={20} stroke="var(--accent)" strokeWidth={1.25}/>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                <Mono size={9} color="var(--accent)">{v.delta}</Mono>
                <Mono size={9} color="rgba(255,255,255,0.3)">{v.range.split(' · ')[0].split(' typical')[0].split(' (')[0]}</Mono>
              </div>
            </div>
          ))}
        </div>

        {/* Latest labs card */}
        <Eyebrow style={{ marginTop: 24, marginBottom: 10 }}>LATEST LABS · APR 22</Eyebrow>
        <div style={{ background: '#0e0e0e', border: '1px solid rgba(255,255,255,0.07)' }}>
          {LABS.slice(0, 5).map((l, i) => (
            <div key={l.id} style={{
              padding: '12px 14px',
              borderBottom: i < 4 ? '1px solid rgba(255,255,255,0.05)' : 'none',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <Tooltip label={l.name} range={`Reference: ${l.range} ${l.unit}`} hint={l.hint} source={l.source}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{l.name}</span>
                </Tooltip>
                <Mono size={9} style={{ marginTop: 2, display: 'block' }}>RANGE {l.range}</Mono>
              </div>
              <div style={{ color: l.flag === 'low' ? '#ffaa00' : 'var(--accent)' }}>
                <Sparkline data={l.trend} w={50} h={16} stroke="currentColor" strokeWidth={1.2} dot={false}/>
              </div>
              <div style={{ textAlign: 'right', minWidth: 60 }}>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 13, fontWeight: 700, color: l.flag === 'low' ? '#ffaa00' : '#fff' }}>
                  {l.value}
                </span>
                <Mono size={9} style={{ marginLeft: 3 }}>{l.unit}</Mono>
              </div>
            </div>
          ))}
        </div>

        {/* Meds */}
        <Eyebrow style={{ marginTop: 24, marginBottom: 10 }}>MEDS · 4 ACTIVE</Eyebrow>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {MEDS.slice(0, 3).map(m => (
            <div key={m.id} style={{
              padding: '10px 12px', background: '#0e0e0e',
              border: '1px solid rgba(255,255,255,0.07)',
              display: 'flex', alignItems: 'center', gap: 10,
            }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 600 }}>{m.name}</div>
                <Mono size={9}>{m.dose} · {m.freq.toUpperCase()}</Mono>
              </div>
              {m.refills != null && (
                <Mono size={9} color={m.refills <= 1 ? '#ffaa00' : 'rgba(255,255,255,0.5)'}>
                  {m.refills} REFILL{m.refills !== 1 ? 'S' : ''}
                </Mono>
              )}
            </div>
          ))}
        </div>

        {/* Pink stripe footer */}
        <div style={{ marginTop: 28, height: 3, background: 'var(--accent)', width: 48 }}/>
        <Mono size={9} style={{ marginTop: 8, display: 'block' }}>END OF RECORD · v0.42</Mono>
      </div>

      {/* Tab bar */}
      <div style={{
        height: 56, borderTop: '1px solid #1f1f1f',
        display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)',
        background: '#000',
      }}>
        {[
          { l: 'HOME',   on: true },
          { l: 'LABS',   on: false },
          { l: 'MEDS',   on: false },
          { l: 'TIME',   on: false },
          { l: 'ASK',    on: false, accent: true },
        ].map((t, i) => (
          <div key={i} style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 4,
            color: t.on ? 'var(--accent)' : 'rgba(255,255,255,0.4)',
            borderTop: t.on ? '2px solid var(--accent)' : '2px solid transparent',
            marginTop: -1,
          }}>
            {t.accent
              ? <PrismMark size={16} color="var(--accent)"/>
              : <div style={{ width: 14, height: 14, border: '1.5px solid currentColor' }}/>}
            <Mono size={8.5} color="currentColor">{t.l}</Mono>
          </div>
        ))}
      </div>
    </div>
  </IOSDevice>
);

const MobileLabDetail = () => {
  const lab = LABS[0]; // LDL
  return (
    <IOSDevice dark={true} width={390} height={812}>
      <div style={{
        width: '100%', height: '100%', background: '#000', color: '#fff',
        fontFamily: 'Space Grotesk, sans-serif', overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
      }}>
        {/* Top nav */}
        <div style={{
          padding: '6px 16px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: '1px solid #1f1f1f',
        }}>
          <button style={{ background: 'transparent', border: 'none', color: 'var(--accent)', fontFamily: 'inherit', fontSize: 14, fontWeight: 600 }}>
            ← Labs
          </button>
          <Mono size={9}>APR 22, 2026</Mono>
          <div style={{ width: 38 }}/>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: '20px' }}>
          {/* Hero value */}
          <Eyebrow>LDL CHOLESTEROL · NORMAL</Eyebrow>
          <div style={{
            marginTop: 12,
            fontFamily: 'Space Grotesk', fontSize: 96, fontWeight: 700,
            letterSpacing: '-0.04em', lineHeight: 0.92, color: 'var(--accent)',
            display: 'flex', alignItems: 'baseline', gap: 6,
          }}>
            92<Mono size={16} color="rgba(255,255,255,0.5)" style={{ fontWeight: 500 }}>mg/dL</Mono>
          </div>
          <Mono size={10} style={{ marginTop: 8, display: 'block' }}>TARGET &lt; 100 · YOU'RE 8 BELOW</Mono>

          {/* Range bar */}
          <div style={{ marginTop: 18 }}>
            <RangeBar value={92} low={50} high={100} hardMin={40} hardMax={200} w={350} accent="#ff0096"/>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6 }}>
              <Mono size={9}>40</Mono>
              <Mono size={9} color="rgba(255,255,255,0.7)">100 · TARGET</Mono>
              <Mono size={9}>200</Mono>
            </div>
          </div>

          {/* Trend chart */}
          <div style={{
            marginTop: 24, padding: '18px 14px',
            background: '#0e0e0e', border: '1px solid rgba(255,255,255,0.07)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <Eyebrow>24 MONTH TREND</Eyebrow>
              <Mono>−50 PTS</Mono>
            </div>
            <TrendChart data={lab.trend} dates={['JAN24', 'JUN24', 'NOV24', 'APR25', 'OCT25', 'APR26']} w={324} h={120}/>
          </div>

          {/* AI insight */}
          <div style={{
            marginTop: 16, padding: 16,
            background: '#0a0a0a', borderLeft: '3px solid var(--accent)',
            border: '1px solid rgba(255,255,255,0.08)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <PrismMark size={14} color="var(--accent)"/>
              <Mono size={9} color="var(--accent)">4UWHAt · INSIGHT</Mono>
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.5, color: 'rgba(255,255,255,0.9)' }}>
              The biggest drop happened 6 months after starting Rosuvastatin. Your trend is still
              going down — at the current rate you'll hit 85 by next visit.
            </div>
          </div>

          {/* Related */}
          <Eyebrow style={{ marginTop: 24, marginBottom: 10 }}>RELATED</Eyebrow>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <RelatedRow icon="Rx"   title="Rosuvastatin 5mg"  meta="STARTED MAR 2024" accent/>
            <RelatedRow icon="Lab"  title="HDL · 64 mg/dL"    meta="DRAWN APR 22"/>
            <RelatedRow icon="Note" title="Annual physical"   meta="DR. ADEBAYO · APR 22"/>
            <RelatedRow icon="Plan" title="Mediterranean diet" meta="LOGGED 142 DAYS"/>
          </div>
        </div>
      </div>
    </IOSDevice>
  );
};

const RelatedRow = ({ icon, title, meta, accent }) => (
  <div style={{
    padding: '12px 14px', background: '#0e0e0e',
    border: '1px solid', borderColor: accent ? 'var(--accent)' : 'rgba(255,255,255,0.07)',
    display: 'flex', alignItems: 'center', gap: 12,
  }}>
    <Mono size={9} color={accent ? 'var(--accent)' : 'rgba(255,255,255,0.5)'} style={{ width: 32 }}>
      {icon.toUpperCase()}
    </Mono>
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 13, fontWeight: 500 }}>{title}</div>
      <Mono size={9}>{meta}</Mono>
    </div>
    <span style={{ color: 'rgba(255,255,255,0.4)' }}>→</span>
  </div>
);

Object.assign(window, { MobileHome, MobileLabDetail });
