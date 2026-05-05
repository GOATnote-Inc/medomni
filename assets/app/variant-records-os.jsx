// Variation A — RECORDS OS
// Dense desktop dashboard, terminal-flavored. Left rail nav, main grid of
// modules, right rail = AI command bar + recent activity. Records-as-data.

const RecordsOS = () => {
  const [activeModule, setActiveModule] = React.useState('overview');
  const [aiOpen, setAiOpen] = React.useState(true);

  // ── Left nav ────────────────────────────────────────────────────────────
  const NAV = [
    { id: 'overview',   label: 'Overview',     count: null,   k: 'O' },
    { id: 'timeline',   label: 'Timeline',     count: 142,    k: 'T' },
    { id: 'labs',       label: 'Labs',         count: 8,      k: 'L' },
    { id: 'meds',       label: 'Medications',  count: 4,      k: 'M' },
    { id: 'cond',       label: 'Conditions',   count: 4,      k: 'C' },
    { id: 'vitals',     label: 'Vitals',       count: null,   k: 'V' },
    { id: 'imaging',    label: 'Imaging',      count: 3,      k: 'I' },
    { id: 'wear',       label: 'Wearables',    count: null,   k: 'W' },
    { id: 'notes',      label: 'Visit notes',  count: 12,     k: 'N' },
    { id: 'team',       label: 'Care team',    count: 4,      k: 'P' },
    { id: 'genome',     label: 'Genome',       count: null,   k: 'G' },
    { id: 'shares',     label: 'Sharing',      count: 4,      k: 'S' },
  ];

  return (
    <div style={{
      width: 1440, height: 900, background: '#000', color: '#fff',
      display: 'grid', gridTemplateColumns: '220px 1fr 360px',
      fontFamily: 'Space Grotesk, sans-serif', overflow: 'hidden',
      border: '1px solid #1f1f1f',
    }}>
      {/* ── LEFT RAIL ─────────────────────────────────────────────────── */}
      <aside style={{
        borderRight: '1px solid #1f1f1f', display: 'flex', flexDirection: 'column',
        background: '#000',
      }}>
        <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid #1f1f1f' }}>
          <Wordmark size={13}/>
          <Mono color="rgba(255,255,255,0.35)" size={9} style={{ marginTop: 10, display: 'block' }}>
            HEALTH / v4.2
          </Mono>
        </div>

        {/* Patient identity */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #1f1f1f' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Avatar initials="MO" size={32} accent={true}/>
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, lineHeight: 1.2 }}>{PATIENT.name}</div>
              <Mono size={9} color="rgba(255,255,255,0.4)">{PATIENT.mrn}</Mono>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: '8px 8px', flex: 1, overflowY: 'auto' }}>
          {NAV.map(n => {
            const active = n.id === activeModule;
            return (
              <button key={n.id} onClick={() => setActiveModule(n.id)} style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                width: '100%', padding: '8px 12px', border: 'none', cursor: 'pointer',
                background: active ? 'rgba(255,0,150,0.08)' : 'transparent',
                color: active ? 'var(--accent)' : 'rgba(255,255,255,0.7)',
                borderLeft: '2px solid', borderLeftColor: active ? 'var(--accent)' : 'transparent',
                fontFamily: 'inherit', textAlign: 'left',
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Mono size={9} color={active ? 'var(--accent)' : 'rgba(255,255,255,0.3)'}>{n.k}</Mono>
                  <span style={{ fontSize: 13, fontWeight: active ? 600 : 500 }}>{n.label}</span>
                </span>
                {n.count != null && (
                  <Mono size={9} color={active ? 'var(--accent)' : 'rgba(255,255,255,0.35)'}>
                    {String(n.count).padStart(3, '0')}
                  </Mono>
                )}
              </button>
            );
          })}
        </nav>

        {/* Footer status */}
        <div style={{ padding: '12px 20px', borderTop: '1px solid #1f1f1f' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Dot color="var(--accent)" glow={true} size={6}/>
            <Mono size={9}>SYNCED · 2 MIN AGO</Mono>
          </div>
        </div>
      </aside>

      {/* ── MAIN ─────────────────────────────────────────────────────── */}
      <main style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Top bar */}
        <div style={{
          height: 56, borderBottom: '1px solid #1f1f1f', padding: '0 28px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <Eyebrow>OVERVIEW</Eyebrow>
            <span style={{ color: 'rgba(255,255,255,0.2)' }}>/</span>
            <Mono color="rgba(255,255,255,0.55)">YOUR RECORD · APR 22, 2026</Mono>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button style={btnGhost}>Export</button>
            <button style={btnGhost}>Share</button>
            <button style={btnPrimary}>+ New entry</button>
          </div>
        </div>

        {/* Content scroll area */}
        <div style={{ overflow: 'auto', padding: '24px 28px', flex: 1 }}>
          {/* Hero row: identity + signal-of-the-day */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 16, marginBottom: 16 }}>
            <div style={cardBase}>
              <Eyebrow style={{ marginBottom: 12 }}>PATIENT · {PATIENT.pronouns.toUpperCase()}</Eyebrow>
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
                <div>
                  <div style={{ fontFamily: 'Space Grotesk', fontSize: 36, fontWeight: 700, letterSpacing: '-0.025em', lineHeight: 1.05 }}>
                    {PATIENT.name}
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 18, flexWrap: 'wrap' }}>
                    <KeyVal k="AGE"   v={`${PATIENT.age} yr`}/>
                    <KeyVal k="DOB"   v={PATIENT.dob}/>
                    <KeyVal k="BLOOD" v={PATIENT.bloodType} tip={{ label: 'BLOOD TYPE', range: 'O+ · universal red-cell donor', hint: 'Compatible with all positive recipients.', source: 'TYPE & SCREEN · 2018' }}/>
                    <KeyVal k="HT"    v={PATIENT.height}/>
                    <KeyVal k="WT"    v={PATIENT.weight} tip={{ label: 'WEIGHT', range: VITALS.weight.range, hint: VITALS.weight.hint, source: VITALS.weight.source }}/>
                    <KeyVal k="PCP"   v={PATIENT.primaryCare}/>
                  </div>
                </div>
                <Stripe width={48} height={3}/>
              </div>
              {/* Active flags */}
              <div style={{ marginTop: 18, paddingTop: 16, borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <Tag accent={true}><Dot size={5} color="var(--accent)" glow={true}/> 2 ACTIVE CONDITIONS</Tag>
                <Tag><Dot size={5} color="rgba(255,255,255,0.5)"/> 3 RX ACTIVE</Tag>
                <Tag><Dot size={5} color="rgba(255,255,255,0.5)"/> NKA · NO ALLERGIES</Tag>
                <Tag color="#ffaa00"><Dot size={5} color="#ffaa00"/> VIT D LOW · 28 ng/mL</Tag>
              </div>
            </div>

            {/* Signal of the day — the big, single insight */}
            <div style={{ ...cardBase, position: 'relative', overflow: 'hidden' }}>
              <Eyebrow style={{ marginBottom: 12 }}>SIGNAL · TODAY</Eyebrow>
              <div style={{ fontFamily: 'Space Grotesk', fontSize: 22, fontWeight: 600, lineHeight: 1.25, letterSpacing: '-0.015em' }}>
                Your LDL dropped <span style={{ color: 'var(--accent)' }}>50 points</span> in 24 months. The statin is working.
              </div>
              <div style={{ marginTop: 16, display: 'flex', alignItems: 'flex-end', gap: 16 }}>
                <div>
                  <div style={{ fontFamily: 'Space Grotesk', fontSize: 48, fontWeight: 700, letterSpacing: '-0.03em', lineHeight: 1, color: 'var(--accent)' }}>
                    92
                  </div>
                  <Mono size={9} style={{ marginTop: 4 }}>MG/DL · TARGET &lt;100</Mono>
                </div>
                <div style={{ flex: 1, paddingBottom: 4, color: 'var(--accent)' }}>
                  <Sparkline data={LABS[0].trend} w={180} h={56} stroke="var(--accent)" fill="var(--accent)" strokeWidth={1.75}/>
                </div>
              </div>
              <Mono size={9} style={{ marginTop: 12, display: 'block', color: 'rgba(255,255,255,0.35)' }}>
                ↳ FLAGGED BY 4UWHAt · 4H AGO
              </Mono>
            </div>
          </div>

          {/* Vitals strip */}
          <div style={{ ...cardBase, marginBottom: 16, padding: 0 }}>
            <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Eyebrow>VITALS · LAST 12 READINGS</Eyebrow>
              <Mono>FROM APPLE WATCH · OURA · CLINIC</Mono>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)' }}>
              {Object.values(VITALS).map((v, i) => (
                <div key={i} style={{
                  padding: '16px 20px',
                  borderRight: i < 5 ? '1px solid rgba(255,255,255,0.06)' : 'none',
                }}>
                  <Mono size={9}>{v.label.toUpperCase()}</Mono>
                  <Tooltip label={v.label} range={v.range} hint={v.hint} source={v.source}>
                    <span style={{ display: 'flex', alignItems: 'baseline', gap: 5, marginTop: 6 }}>
                      <span style={{ fontFamily: 'Space Grotesk', fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em' }}>
                        {v.value}
                      </span>
                      <Mono size={9} style={{ fontWeight: 500 }}>{v.unit}</Mono>
                    </span>
                  </Tooltip>
                  <div style={{ marginTop: 8, height: 22, color: 'var(--accent)' }}>
                    <Sparkline data={v.spark} w={120} h={22} stroke="var(--accent)" strokeWidth={1.25}/>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                    <Mono size={9} color="var(--accent)">{v.delta}</Mono>
                    <Mono size={9} color="rgba(255,255,255,0.3)">{v.range.split(' · ')[0].split(' typical')[0]}</Mono>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Labs + Meds + Conditions row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16, marginBottom: 16 }}>
            {/* Labs table */}
            <div style={cardBase}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <Eyebrow>LATEST LABS · APR 22</Eyebrow>
                <Mono color="rgba(255,255,255,0.4)">8 RESULTS · 1 OUT OF RANGE</Mono>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {/* header */}
                <div style={tableHead}>
                  <span>ANALYTE</span>
                  <span style={{ textAlign: 'right' }}>VALUE</span>
                  <span>RANGE</span>
                  <span>TREND</span>
                  <span style={{ textAlign: 'right' }}>FLAG</span>
                </div>
                {LABS.map(l => (
                  <div key={l.id} style={tableRow}>
                    <Tooltip label={l.name} range={`Reference: ${l.range} ${l.unit}`} hint={l.hint} source={l.source}>
                      <span style={{ fontWeight: 500 }}>{l.name}</span>
                    </Tooltip>
                    <span style={{ textAlign: 'right', fontFamily: 'JetBrains Mono', fontVariantNumeric: 'tabular-nums', color: l.flag === 'low' ? '#ffaa00' : '#fff' }}>
                      {l.value} <Mono size={9}>{l.unit}</Mono>
                    </span>
                    <Mono>{l.range}</Mono>
                    <span style={{ color: l.flag === 'low' ? '#ffaa00' : 'var(--accent)' }}>
                      <Sparkline data={l.trend} w={70} h={18} stroke="currentColor" strokeWidth={1.25} dot={false}/>
                    </span>
                    <span style={{ textAlign: 'right' }}>
                      {l.flag === 'normal'
                        ? <Mono size={9} color="rgba(255,255,255,0.45)">OK</Mono>
                        : <Tag color="#ffaa00" style={{ fontSize: 8.5, padding: '2px 5px' }}>LOW</Tag>}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Meds + conditions stacked */}
            <div style={{ display: 'grid', gridTemplateRows: '1fr 1fr', gap: 16 }}>
              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>MEDICATIONS · 4 ACTIVE</Eyebrow>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {MEDS.slice(0, 4).map(m => (
                    <div key={m.id} style={medRow}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <Tooltip label={m.name} range={`${m.dose} · ${m.freq}`} hint={`Prescriber: ${m.prescriber} · since ${m.since}`} source={m.refills != null ? `${m.refills} refills remaining` : 'OTC'}>
                          <span style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</span>
                        </Tooltip>
                        <Mono size={9} style={{ marginTop: 2, display: 'block' }}>{m.dose} · {m.freq}</Mono>
                      </div>
                      {m.adherence != null && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <Mono size={9}>{Math.round(m.adherence * 100)}%</Mono>
                          <div style={{ width: 36, height: 3, background: 'rgba(255,255,255,0.08)' }}>
                            <div style={{ width: `${m.adherence * 100}%`, height: '100%', background: 'var(--accent)' }}/>
                          </div>
                        </div>
                      )}
                      {m.refills != null && (
                        <Mono size={9} color={m.refills <= 1 ? '#ffaa00' : 'rgba(255,255,255,0.45)'}>
                          {m.refills} REFILL{m.refills !== 1 ? 'S' : ''}
                        </Mono>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div style={cardBase}>
                <Eyebrow style={{ marginBottom: 12 }}>PROBLEM LIST</Eyebrow>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  {CONDITIONS.map(c => (
                    <div key={c.id} style={{
                      display: 'grid', gridTemplateColumns: '12px 1fr auto auto', alignItems: 'center', gap: 10,
                      padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)',
                    }}>
                      <Dot color={c.status === 'active' ? 'var(--accent)' : 'rgba(255,255,255,0.3)'} size={6} glow={c.status === 'active'}/>
                      <span style={{ fontSize: 13, color: c.status === 'active' ? '#fff' : 'rgba(255,255,255,0.5)' }}>
                        {c.name}
                      </span>
                      <Mono size={9}>{c.icd}</Mono>
                      <Mono size={9} color={c.status === 'active' ? 'var(--accent)' : 'rgba(255,255,255,0.4)'}>
                        SINCE {c.onset}
                      </Mono>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Bottom row: timeline preview + sharing */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16 }}>
            <div style={cardBase}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <Eyebrow>TIMELINE · LAST 6 MONTHS</Eyebrow>
                <Mono>SCROLL FOR FULL HISTORY ↓</Mono>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {TIMELINE.slice(0, 6).map((e, i) => (
                  <div key={i} style={timelineRow}>
                    <Mono size={10} color="rgba(255,255,255,0.4)" style={{ width: 78, flexShrink: 0 }}>
                      {e.date.slice(5).replace('-', '/')}
                    </Mono>
                    <Tag color={e.kind === 'visit' ? 'var(--accent)' : 'rgba(255,255,255,0.3)'}
                         accent={e.kind === 'visit'}
                         style={{ width: 50, justifyContent: 'center' }}>
                      {e.tag}
                    </Tag>
                    <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>{e.title}</span>
                    <Mono color="rgba(255,255,255,0.45)">{e.who}</Mono>
                  </div>
                ))}
              </div>
            </div>

            <div style={cardBase}>
              <Eyebrow style={{ marginBottom: 12 }}>WHO HAS ACCESS · {SHARES.length}</Eyebrow>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {SHARES.map((s, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '10px 12px', border: '1px solid rgba(255,255,255,0.06)',
                  }}>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 500 }}>{s.who}</div>
                      <Mono size={9}>{s.scope.toUpperCase()}</Mono>
                    </div>
                    <button style={{ ...btnGhost, padding: '4px 8px', fontSize: 10 }}>REVOKE</button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ── RIGHT RAIL — AI command bar + activity ──────────────────── */}
      <aside style={{
        borderLeft: '1px solid #1f1f1f', display: 'flex', flexDirection: 'column',
        background: '#000',
      }}>
        {/* AI command */}
        <div style={{ padding: '20px 22px', borderBottom: '1px solid #1f1f1f' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <Eyebrow>ASK YOUR RECORD</Eyebrow>
            <Mono size={9} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ ...kbd }}>⌘</span><span style={{ ...kbd }}>K</span>
            </Mono>
          </div>
          <div style={{
            background: '#0a0a0a', border: '2px solid var(--accent)',
            padding: '12px 14px', display: 'flex', alignItems: 'center', gap: 10,
            boxShadow: 'var(--glow-pink)',
          }}>
            <PrismMark size={16} color="var(--accent)"/>
            <span style={{ flex: 1, fontFamily: 'JetBrains Mono', fontSize: 12, color: 'rgba(255,255,255,0.95)' }}>
              ask anything<span className="caret">_</span>
            </span>
          </div>
          {/* Suggestions */}
          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
            {AI_SUGGESTIONS.map((s, i) => (
              <button key={i} style={suggestionBtn}>
                <span style={{ color: 'var(--accent)', marginRight: 8 }}>→</span>
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Care team */}
        <div style={{ padding: '18px 22px', borderBottom: '1px solid #1f1f1f' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
            <Eyebrow>CARE TEAM</Eyebrow>
            <Mono>{CARE_TEAM.filter(c => c.online).length} ONLINE</Mono>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {CARE_TEAM.map(c => (
              <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Avatar initials={c.avatar} online={c.online} size={28}/>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.3 }}>{c.name}</div>
                  <Mono size={9}>{c.role.toUpperCase()}</Mono>
                </div>
                <button style={{ ...btnGhost, padding: '3px 7px', fontSize: 10 }}>MSG</button>
              </div>
            ))}
          </div>
        </div>

        {/* Recent activity / log */}
        <div style={{ padding: '18px 22px', flex: 1, overflowY: 'auto' }}>
          <Eyebrow style={{ marginBottom: 12 }}>RECENT ACTIVITY</Eyebrow>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <ActivityRow time="2 min" actor="4UWHAt" verb="synced" obj="Apple Health · 47 entries"/>
            <ActivityRow time="11 min" actor="Dr. Patel" verb="signed" obj="Visit note · Feb 19" accent/>
            <ActivityRow time="4h" actor="4UWHAt" verb="flagged" obj="Vitamin D · 28 ng/mL · low"/>
            <ActivityRow time="1d" actor="Quest" verb="released" obj="Lipid panel + CBC + CMP"/>
            <ActivityRow time="2d" actor="You" verb="shared" obj="Pulmonary record → Dr. Patel"/>
            <ActivityRow time="6d" actor="CVS" verb="filled" obj="Rosuvastatin · 30 day"/>
          </div>
        </div>

        {/* Stripe at bottom */}
        <div style={{ height: 4, background: 'var(--accent)' }}/>
      </aside>
    </div>
  );
};

// ── Sub-bits ────────────────────────────────────────────────────────────────
const KeyVal = ({ k, v, tip }) => {
  const inner = (
    <span style={{ fontSize: 13, fontWeight: 500 }}>{v}</span>
  );
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Mono size={9}>{k}</Mono>
      {tip ? <Tooltip {...tip}>{inner}</Tooltip> : inner}
    </div>
  );
};

const ActivityRow = ({ time, actor, verb, obj, accent }) => (
  <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
    <Mono size={9} style={{ width: 36, flexShrink: 0, paddingTop: 2 }}>{time.toUpperCase()}</Mono>
    <Dot size={5} color={accent ? 'var(--accent)' : 'rgba(255,255,255,0.35)'} style={{ marginTop: 6, flexShrink: 0 }}/>
    <div style={{ fontSize: 12, lineHeight: 1.4, flex: 1 }}>
      <span style={{ color: accent ? 'var(--accent)' : 'rgba(255,255,255,0.95)', fontWeight: 600 }}>{actor}</span>{' '}
      <span style={{ color: 'rgba(255,255,255,0.5)' }}>{verb}</span>{' '}
      <span style={{ color: '#fff' }}>{obj}</span>
    </div>
  </div>
);

// ── Shared inline styles ────────────────────────────────────────────────────
const cardBase = {
  background: '#0e0e0e',
  border: '1px solid rgba(255,255,255,0.07)',
  padding: 18,
};

const tableHead = {
  display: 'grid',
  gridTemplateColumns: '1.5fr 1fr 0.9fr 0.9fr 0.5fr',
  gap: 12, padding: '8px 0',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
  fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fontWeight: 700,
  letterSpacing: '0.12em', color: 'rgba(255,255,255,0.4)',
};

const tableRow = {
  display: 'grid',
  gridTemplateColumns: '1.5fr 1fr 0.9fr 0.9fr 0.5fr',
  gap: 12, padding: '10px 0',
  borderBottom: '1px solid rgba(255,255,255,0.04)',
  fontSize: 12.5, alignItems: 'center',
};

const medRow = {
  display: 'flex', gap: 10, alignItems: 'center',
  padding: '8px 10px', border: '1px solid rgba(255,255,255,0.06)',
};

const timelineRow = {
  display: 'flex', gap: 12, alignItems: 'center',
  padding: '10px 0', borderBottom: '1px solid rgba(255,255,255,0.05)',
};

const btnGhost = {
  background: 'transparent', color: 'rgba(255,255,255,0.7)',
  border: '1px solid #2a2a2a', padding: '7px 12px',
  fontFamily: 'Space Grotesk, sans-serif', fontWeight: 600, fontSize: 11.5,
  cursor: 'pointer',
};

const btnPrimary = {
  background: 'var(--accent)', color: '#fff',
  border: '1px solid var(--accent)', padding: '7px 14px',
  fontFamily: 'Space Grotesk, sans-serif', fontWeight: 600, fontSize: 11.5,
  cursor: 'pointer',
};

const kbd = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  fontFamily: 'JetBrains Mono', fontSize: 9, fontWeight: 700,
  padding: '2px 5px', border: '1px solid rgba(255,255,255,0.18)',
  color: 'rgba(255,255,255,0.7)', minWidth: 14,
};

const suggestionBtn = {
  display: 'block', width: '100%', textAlign: 'left',
  padding: '8px 12px', background: '#0a0a0a',
  border: '1px solid rgba(255,255,255,0.06)',
  color: 'rgba(255,255,255,0.85)',
  fontFamily: 'Space Grotesk, sans-serif', fontSize: 11.5, fontWeight: 500,
  cursor: 'pointer', lineHeight: 1.4,
};

window.RecordsOS = RecordsOS;
