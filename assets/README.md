# Handoff: 4UWHAt — Personalized Medical Records

## Overview

A patient-facing personal health record interface for **4UWHAt**, a black/white/hot-magenta brand inspired by stark technical/developer-tool aesthetics. The product gives a single patient a unified, queryable view of their entire medical record: vitals, labs, medications, conditions, imaging, visit timeline, care team, and an "Ask your record" AI command bar.

The package contains **two desktop directions** (1440×900) and **two mobile screens** (iPhone 390×812):

- **A · Records OS** — dense terminal-flavored dashboard. Three-column layout with nav rail, modular grid, and AI/activity rail.
- **B · Body Console** — cinematic single-focus layout with a refracting prism mark, a display-scale headline, and a "spectrum" of 6 body systems.
- **Mobile · Home** — greeting + signal-of-the-day, vitals, labs, meds, bottom tab bar.
- **Mobile · Lab Detail (LDL)** — display-scale value, range bar, 24-month trend chart, AI insight, related items.

## About the Design Files

The files in this bundle are **design references created in HTML/JSX**. They are prototypes showing intended look and behavior, not production code to copy directly. The task is to **recreate these designs in the target codebase's existing environment** (React, Vue, SwiftUI, native iOS/Android, etc.) using that codebase's established patterns, libraries, and component library. If no codebase environment exists yet, choose the most appropriate framework — React + TypeScript + Tailwind (or vanilla CSS modules) is a sensible default for a web app of this kind, and SwiftUI for native iOS.

The HTML uses inline-JSX (Babel standalone) for fast iteration; production should use compiled JSX/TSX with proper build tooling.

## Fidelity

**High-fidelity (hifi).** Pixel-precise mockups with final colors, typography, spacing, borders, and hover behaviors. Recreate pixel-perfectly using your codebase's existing libraries and patterns. Tokens, ranges, and copy should be reproduced exactly as documented below. Mock data is realistic and can stay as-is until real data is wired in.

## Files in this bundle

| Path | Contents |
|---|---|
| `Personalized Medical Records.html` | Entry HTML — pinned React/Babel script tags, mounts `<App/>`, defines tweak defaults, applies CSS-var theming. |
| `lib/colors_and_type.css` | Atomic CSS — design tokens (color, type, spacing, motion) and element defaults. **Import first on every surface.** |
| `lib/design-canvas.jsx` | Pan/zoom canvas wrapper — present-only chrome. Replace with your routing/layout in production. |
| `lib/ios-frame.jsx` | iPhone bezel — present-only chrome. Replace with native iOS or your mobile webview shell. |
| `lib/tweaks-panel.jsx` | Designer tweak panel — present-only. Skip in production. |
| `app/data.jsx` | Mock patient data: `PATIENT`, `VITALS`, `CONDITIONS`, `MEDS`, `LABS`, `TIMELINE`, `IMAGING`, `CARE_TEAM`, `SHARES`, `AI_SUGGESTIONS`. Use as a schema reference for typing real data. |
| `app/atoms.jsx` | Reusable primitives: `Sparkline`, `TrendChart`, `Eyebrow`, `Mono`, `Stripe`, `Dot`, `Tag`, `Avatar`, `RangeBar`, `Stat`, `PrismMark`, `Wordmark`, `Tooltip`. |
| `app/variant-records-os.jsx` | Desktop A. |
| `app/variant-body-console.jsx` | Desktop B. |
| `app/variant-mobile.jsx` | Mobile home + lab detail. |
| `assets/prism42-logo-white.png` | Brand mark (white outline + magenta stripes). |
| `assets/prism42-logo-pink.png` | Brand mark (all-magenta). |

> Note on assets: file names retain their original `prism42-` prefix because they came from the underlying design system. The product brand is now **4UWHAt**. Either rename the files or alias them on import — but the marks themselves are correct.

## Design Tokens

All tokens live in `lib/colors_and_type.css` as CSS custom properties under `:root`. Light theme overrides under `.theme-light`. **Default theme is dark.**

### Color

| Token | Hex | Use |
|---|---|---|
| `--p42-black` | `#000000` | Page background |
| `--p42-ink` | `#0a0a0a` | Slightly softer black for raised surfaces |
| `--p42-graphite` | `#141414` | Raised surface |
| `--p42-coal` | `#1f1f1f` | Card surface |
| `--p42-iron` | `#2a2a2a` | Hairline / divider |
| `--p42-steel` | `#3d3d3d` | Stronger border |
| `--p42-ash` | `#6e6e6e` | Subtle text |
| `--p42-fog` | `#a3a3a3` | Muted text |
| `--p42-bone` | `#d4d4d4` | — |
| `--p42-paper` | `#f4f4f4` | Card on light |
| `--p42-snow` | `#fafafa` | Raised on light |
| `--p42-white` | `#ffffff` | Primary text on dark |
| `--p42-pink` (= `--accent`) | `#ff0096` | The single brand accent. Hot magenta. |
| `--p42-pink-bright` | `#ff33ad` | Hover |
| `--p42-pink-dim` | `#cc0078` | Press |
| `--p42-pink-deep` | `#99005a` | On-light variant |
| `--p42-danger` | `#ff3355` | Errors only |
| `--p42-warn` | `#ffaa00` | "Out of range" labs (e.g. low Vitamin D) |
| `--p42-info` | `#66aaff` | "Watch" status (e.g. respiratory) |

Card hairline-on-card (used for inner dividers): `rgba(255,255,255,0.06)` to `rgba(255,255,255,0.08)`.

### Typography

- **Display + body**: Space Grotesk (Google Fonts, weights 400/500/600/700)
- **Mono**: JetBrains Mono (Google Fonts, weights 400/500/700) — used for labels, eyebrows, code, kbd, ranges, MRN, dates, ICD codes
- **Code (alt)**: Space Mono — loaded but not the default

Font sizes (`--fs-*`): 12 / 14 / 16 / 18 / 20 / 24 / 30 / 36 / 48 / 64 / 96 / 128 px.

Letter-spacing: `--tr-tight` -0.03em (display), `--tr-snug` -0.015em (h2/h3), `--tr-eyebrow` 0.16em (uppercase mono labels).

**Eyebrow style** (used everywhere as section labels): mono, 10–11px, weight 700, uppercase, letter-spacing 0.16em, color `var(--accent)`.

### Spacing

4-pt grid, full token list in CSS: 4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48 / 64 / 80 / 96 / 128.

### Borders & radii

- **Default border**: 1px `var(--border)` = `#2a2a2a` on dark
- **Inner card hairline**: 1px `rgba(255,255,255,0.06)` to `0.08`
- **Emphasis border**: 2px `var(--accent)` (focused inputs, AI bar, primary card edges)
- **Loud / brand stripe**: 3px–4px `var(--accent)`
- **Radii**: square (`--r-0`) or 2px max (`--r-1`, `--r-2`). The only rounded element is `--r-pill` (9999px) for status dots and inline tags.

### Shadows / glow

- `--shadow-1` minimal lift on dark
- `--shadow-2` modal/popover
- `--shadow-3` full overlay
- `--glow-pink` = `0 0 0 2px rgba(255,0,150,0.4), 0 0 32px rgba(255,0,150,0.4)` — used on the AI command bar and active state dots
- `--glow-soft` = `0 0 24px rgba(255,0,150,0.18)` — ambient glow on hero elements

### Motion

- `--ease-out` `cubic-bezier(0.2, 0.8, 0.2, 1)` — default
- `--ease-spring` `cubic-bezier(0.34, 1.4, 0.64, 1)` — only on small accent micro-interactions
- Durations: 120ms (fast) / 220ms (base) / 420ms (slow). Never longer than 500ms.

## Brand voice rules

- **Wordmark**: lowercase `4UWHAt` (per user direction). Lockup is wordmark + `PrismMark` SVG icon, 18×18, color = `--accent`.
- **Eyebrows**: UPPERCASE mono — only place we shout (`OVERVIEW`, `VITALS · LAST 12 READINGS`, `ASK YOUR RECORD`).
- **Numerals always** — `42 plugins`, `42ms`, `0`, never written-out numbers.
- **No emoji.** Use mono glyphs: `→ ↳ ✕ ▸ ● ◇ ↑ ↓`.
- **Em dashes** for asides. Arrows `→` (never `>>`) for CTAs.
- **Body copy is direct and slightly dry.** "The statin is working." — not "Your medication is helping you achieve your wellness goals."

## Screens / Views

### Screen 1 — Desktop A · Records OS (1440×900)

**Component**: `RecordsOS` in `app/variant-records-os.jsx`.

**Layout**: 3 columns: `220px | 1fr | 360px`, full height 900px, dark.

**Left rail (220px)**:
- Brand block (20px padding, hairline bottom): wordmark + `HEALTH / v4.2` mono caption
- Patient identity block: 32px magenta avatar with initials "MO" + name (13px semibold) + MRN (mono 9px)
- Nav list (12 items): row 36px tall, 12px h-pad, mono key letter (O T L M …) + label (13px) + count (mono 9px right). Active row: 2px left border `--accent`, background `rgba(255,0,150,0.08)`, fg `--accent`.
- Footer: pink dot + glow + `SYNCED · 2 MIN AGO`

**Top bar (56px tall, 28px h-pad, hairline bottom)**: eyebrow `OVERVIEW` / mono breadcrumb on left; `Export` / `Share` / `+ New entry` (primary magenta) buttons right.

**Main scroll area (24px / 28px padding)**:
1. **Hero row** — 1.2fr / 1fr grid, 16px gap.
   - **Patient card** (`#0e0e0e` bg, hairline border, 18px pad): eyebrow `PATIENT · SHE/HER`, 36px name, key-val pairs (AGE / DOB / BLOOD / HT / WT / PCP) in 18px gap row, magenta stripe, hairline divider, then 4 status tags.
   - **Signal card**: eyebrow `SIGNAL · TODAY`, 22px headline ("Your LDL dropped **50 points**…"), big magenta 48px number + sparkline of LDL trend, mono attribution.
2. **Vitals strip** — single card with 6 columns (HR, BP, SpO₂, HRV, Weight, Sleep). Each cell: mono label, 24px value + unit, 22px tall sparkline magenta, delta in magenta + range hint in dim mono right-aligned.
3. **Labs (1.4fr) / Meds + Conditions (1fr)** row, 16px gap.
   - **Labs**: 5-col grid: ANALYTE / VALUE / RANGE / TREND / FLAG. Header is mono 9px `rgba(255,255,255,0.4)`. Rows 12.5px, hairline bottom. Out-of-range value rendered in `#ffaa00`, with a `LOW` warning tag.
   - **Meds**: 4 rows, each `border 1px rgba(255,255,255,0.06)`, padding 8/10. Name 13px semibold, dose/freq mono. Adherence % + 36px progress bar magenta. Refills count in dim mono (warn if ≤1).
   - **Conditions**: 4-col grid (dot / name / icd / since). Active = magenta dot + glow + white text. Resolved = dim dot + dim text.
4. **Timeline (1.4fr) / Sharing (1fr)** row.
   - **Timeline**: 6 rows, each: mono date / 50px-wide colored tag / title 13px / clinician dim mono.
   - **Sharing**: 4 access cards, each padded 10/12 with `REVOKE` ghost button.

**Right rail (360px, hairline left)**:
- **AI command** (20/22 padding, hairline bottom): eyebrow + ⌘K kbd. Then a `2px solid var(--accent)` framed input with `--glow-pink`, prism mark icon + "ask anything_" with blinking caret. 4 suggestion buttons below, dark `#0a0a0a`, hairline border, hover → magenta.
- **Care team**: eyebrow + online count. 4 rows: avatar (28px, online indicator pink dot lower-right), name 12px / role mono, `MSG` ghost button.
- **Recent activity**: scrollable. Each row: mono time (36px wide) / dot / `<actor> <verb> <object>` body. "4UWHAt" rows are accent.
- **Bottom**: 4px-tall full-width magenta stripe (signature motif).

**Hover/interaction**:
- Card hover: border → `--accent` (no scale, no shadow change).
- Buttons: see Components below.
- All measurement values: dotted hover underline → `Tooltip` portal (see below).

### Screen 2 — Desktop B · Body Console (1440×900)

**Component**: `BodyConsole` in `app/variant-body-console.jsx`.

**Layout**: 2 columns, `1fr | 420px`, with absolute-positioned 56px top header.

**Header**: wordmark + 1px-vertical hairline + `MAYA OKAFOR · P42-0096-MAYA` mono, then 5 mono tab buttons (Signals / Timeline / Systems / Genome / Sharing) with 2px magenta bottom border on active, then live-status dot + `MO` avatar.

**Main (88/32/24/32 padding)**:
1. **Hero row** — 380px / 1fr grid, 32px gap, 360px tall.
   - **Left**: `PrismHero` SVG. White beam in left → triangular prism outline (110,60→110,300→270,180) → 6 colored refracted lines fanning out to right edge labeled CARDIO / METAB / RESP / IMMUNE / NUTRI / SLEEP. Mono `INPUT · UNIFIED` top-left and `OUTPUT · 6 SYSTEMS` top-right.
   - **Right**: eyebrow `YOUR BODY · APR 22, 2026`, 96px display "Mostly\nresolved." (resolved in `--accent`), 18px lead paragraph, hairline rule, 4-column hero stats: 42 DAYS streak / 58 BPM resting (accent) / 92 MG/DL LDL / 0 rescue inhaler. Each stat is hoverable.
2. **AI ribbon** — full-bleed, `#0a0a0a` bg, **3px left border in `--accent`**, hairline elsewhere. Prism mark + `ASK YOUR RECORD` mono + sample question with blinking magenta caret + 2 suggestion chips + ⌘K kbd.
3. **Spectrum** — 2-column grid of 6 `SpectrumRow` cards. Each card: index `0N/06` top-right, status dot+glow + system code (mono colored) + system name, 22px value, 12px detail line, **24-segment progress bar** at the bottom showing fill ratio (good 22/24, watch 16/24, low 11/24).

**Right rail (88/28/24/28 padding, `#040404` bg, hairline left)**:
- **Timeline**: vertical line at left=4px, 9px square dot per event (magenta + glow on visits, hairline ring otherwise). Each event: date mono / colored tag / title 12.5px / who·loc dim mono.
- **Collapsed modules**: MEDICATIONS, IMAGING, CARE TEAM. Each is `#0a0a0a` border-1px card with eyebrow + count, then content rows.

### Screen 3 — Mobile Home (iPhone 390×812)

**Component**: `MobileHome` in `app/variant-mobile.jsx`.

**Wraps `IOSDevice` dark**.

Top: wordmark + magenta `MO` avatar.

Scrollable body:
- Eyebrow `TODAY · APR 22`
- 32px headline "Hi Maya. **Your LDL is 92.**" (LDL value in `--accent`)
- 14px lead "50 points lower than 24 months ago. The statin is working."
- AI launcher button: full-width, 2px magenta border + `--glow-pink`, prism mark + `ASK YOUR RECORD` + arrow.
- Vitals 2×2 (HR / BP / HRV / Sleep) — each: mono label, 22px value (hoverable, tooltip), magenta sparkline, delta + abbreviated range row.
- Labs list (5 most recent): row of name (hoverable) + mono `RANGE x–y` + 50px sparkline + 13px mono value.
- Meds list (3): name + dose / freq + refill count.
- Footer: 48px magenta stripe + `END OF RECORD · v0.42` mono.

Bottom tab bar (56px, hairline top): 5 tabs HOME / LABS / MEDS / TIME / **ASK** (magenta prism mark, primary). Active tab: magenta + 2px top border.

### Screen 4 — Mobile Lab Detail (LDL) (iPhone 390×812)

**Component**: `MobileLabDetail`.

Top nav: `← Labs` magenta button + `APR 22, 2026` mono.

Body:
- Eyebrow `LDL CHOLESTEROL · NORMAL`
- **96px display value `92 mg/dL`** in `--accent`
- Mono caption `TARGET <100 · YOU'RE 8 BELOW`
- **`RangeBar`**: 350px wide, value=92, low=50, high=100 (target band), hardMin=40, hardMax=200. Tick labels under: 40 / 100 · TARGET / 200.
- 24-month trend card (`#0e0e0e` border): eyebrow `24 MONTH TREND` + `−50 PTS` mono. `TrendChart` 324×120 with magenta line, dots, gridlines, last point emphasized.
- AI insight card: 3px magenta left border, prism mark + `4UWHAt · INSIGHT` mono, 14px body.
- Related: 4 rows (Rx / Lab / Note / Plan) — each row has icon code, title, meta.

## Components

### `Tooltip` (atoms.jsx)

Hover-revealed reference info on any measurement.

- Wrapped element gets `border-bottom: 1px dotted rgba(255,255,255,0.18)` and `cursor: help`.
- Renders into a portal anchored above (or below) the element.
- 80ms hover delay.
- Surface: `#000` bg, 1px `--accent` border, top 2px magenta hairline (absolute), `--shadow-2` + 1px magenta outer ring, 10/12 padding, 180–260px wide.
- Content: optional `label` (mono eyebrow), `range` (white 12px value), `hint` (12.5px muted body), `source` (mono 9px footer separated by hairline).

Recreate with your tooltip primitive (e.g. Radix Tooltip), keeping the visual treatment exact.

### `Sparkline`, `TrendChart`, `RangeBar`

All custom SVG, parameterized. See `atoms.jsx`. The TrendChart's gridlines are at 0/25/50/75/100% with `rgba(255,255,255,0.06)`. Last data point dot is 3.5px filled magenta; earlier points are 2.25px hollow with magenta stroke.

### Buttons

| Variant | Style |
|---|---|
| **Primary** | `bg: var(--accent); color: #fff; border: 2px solid var(--accent); padding: 7–10px / 14–18px; font: Space Grotesk 600 11.5–14px`. Hover → bg `--accent-hover`. Press → bg `--accent-press` + `transform: translateY(1px)`. |
| **Outline** | `bg: transparent; color: #fff; border: 2px solid var(--p42-iron)`. Hover → border + fg → `--accent`. |
| **Ghost** | `bg: transparent; border: 1px solid var(--p42-iron); color: rgba(255,255,255,0.7); padding: 4–7px / 8–12px`. Hover → fg → `--accent`. |
| **Danger** | transparent, `--p42-danger` text + border. |

### Tag

Mono 9.5px / 700 / 0.12em / uppercase, padding 3/7, 2px radius, 1px border. Variants: default (`rgba(255,255,255,0.18)`), accent (magenta border + `rgba(255,0,150,0.08)` bg + magenta fg), color-prop (e.g. `#ffaa00`).

### Avatar

Square (no border-radius). Variants: outline (`rgba(255,255,255,0.2)` border, white fg), accent (magenta bg + black fg). Online indicator: 6×6 magenta dot + glow at lower-right with 1px black border.

### Card

Default: `bg: #0e0e0e`, 1px `rgba(255,255,255,0.07)` border, 18px padding. Hover: border → `--accent`, no other change.

### `kbd`

9px mono / 700, 2/5 padding, 1px `rgba(255,255,255,0.18)` border, fg `rgba(255,255,255,0.7)`, min-width 14px.

## Interactions & Behavior

- **Tooltip on every measurement** (vitals, hero stats, labs, meds, key-vals like Blood/Weight, spectrum values). 80ms delay in, no fade-out delay.
- **Card hover**: border color flip to `--accent`, 120ms.
- **Buttons**: see table.
- **AI bar caret**: `_` blinks via `@keyframes blink { 50% { opacity: 0; } }` on `.caret`, 1s step-end.
- **Tabs (Body Console)**: clicking sets `tab` state, restyles bottom border.
- **Nav (Records OS)**: clicking sets `activeModule` state. (In production: route per module.)
- **Mobile tab bar**: standard route per tab.
- **Lab detail back arrow**: pops navigation.
- **AI launcher** (mobile button / desktop ⌘K): opens AI conversation surface (not yet designed — flag with the team).

### Focus

`outline: 2px solid var(--accent); outline-offset: 2px` on `:focus-visible` (already in `colors_and_type.css`).

### Empty / loading / error states

Not designed in this round — flag with the team. Voice rules: keep dry. e.g. "Nothing here yet. That's not a bug." / "Couldn't reach the resolver. Check your token, then try again."

## State Management

For an MVP recreation:

- **Patient context** — single `Patient` object loaded by id from your records API. Type-shape mirrors `app/data.jsx`'s `PATIENT`.
- **Vitals** — keyed object with `value, unit, label, delta, range, hint, source, spark[]`. Refreshed from wearables/clinic ingestion pipelines.
- **Labs** — array; sorted by date desc. Each has `value, unit, range, flag ('normal'|'low'|'high'), trend[], dates[]`, `hint, source`.
- **Active module** (Records OS) and **active tab** (Body Console) — local UI state.
- **AI conversation** — separate store; the command bar dispatches a query and routes to the conversation view (not designed yet).
- **Tooltip open state** — local per-tooltip. If using Radix/Headless UI, lean on theirs.
- **Sharing/consent** — separate consent service; revoke action posts to that.

Data fetching: typical pattern is per-module react-query (or SWR / TanStack Query) hooks that depend on patient id. Compose on the dashboard.

## Mock data → real data

Treat `app/data.jsx` as a **schema reference**. The shapes are hand-tuned and worth preserving as TypeScript interfaces. Map your real EHR / FHIR resources onto them:

- `LABS[i]` ≈ FHIR `Observation` with category=laboratory
- `VITALS[k]` ≈ FHIR `Observation` with category=vital-signs
- `MEDS[i]` ≈ FHIR `MedicationStatement` / `MedicationRequest`
- `CONDITIONS[i]` ≈ FHIR `Condition` (icd → `code.coding`)
- `TIMELINE[i]` ≈ aggregate of `Encounter`, `Observation`, `Immunization`, `Communication`
- `IMAGING[i]` ≈ FHIR `ImagingStudy` / `DiagnosticReport`
- `CARE_TEAM[i]` ≈ FHIR `CareTeam.participant`
- `SHARES[i]` ≈ `Consent`

The `hint` and `source` fields are derived/computed UI strings — likely produced by your AI summarization layer or a curated content table. Do not store them on the raw FHIR record.

## Assets

| Asset | Origin | Use |
|---|---|---|
| `assets/prism42-logo-white.png` | 4UWHAt brand kit (originally named for the parent design system) | Wordmark mark on dark. |
| `assets/prism42-logo-pink.png` | 4UWHAt brand kit | All-magenta variant for photo overlays. |
| Icons | None imported. The HTML uses small custom SVGs (`PrismMark`) and Unicode glyphs (`→ ↳ ✕ ●`) only. The 4UWHAt design system's default icon recommendation is **Lucide** (2px stroke, square endcaps). |

If your codebase already has an icon library, use it — keep stroke 2px, square corners, currentColor. Replace the placeholder geometric outlines used in the mobile tab bar with real icons.

## Iconography rules

- Stroke: 2px, never less.
- Color: `currentColor`. Magenta for active state.
- Sizes: 16 (inline body), 20 (UI default), 24 (headers), 32+ (feature blocks).
- No drop-shadows. No rounded line caps unless part of a logo.
- Filled icons reserved for status (active dot, completed step).
- **No emoji.**

## Recreation checklist

- [ ] Vendor or load Space Grotesk + JetBrains Mono.
- [ ] Port tokens from `colors_and_type.css` into your token system (Tailwind config, design-token JSON, CSS vars, etc.).
- [ ] Build atoms first: `Sparkline`, `TrendChart`, `RangeBar`, `Tooltip`, `Tag`, `Eyebrow`/`Mono`, `Avatar`, `PrismMark`, `Wordmark`.
- [ ] Implement Records OS desktop view (Screen 1).
- [ ] Implement Body Console desktop view (Screen 2). Keep `PrismHero` SVG geometry exact.
- [ ] Implement mobile home (Screen 3) and lab detail (Screen 4).
- [ ] Wire tooltips on every measurement. Reuse the same primitive everywhere.
- [ ] Hook up routing for nav rail / tab bar / lab detail drilldown.
- [ ] Connect to real data sources (FHIR / your EHR layer).
- [ ] Define empty/loading/error states with the team — match the voice ("Nothing here yet. That's not a bug.").
- [ ] AI command bar: route to a real conversation surface (not yet designed).

## Caveats / open questions

1. **AI conversation surface not designed.** The command bar exists; the resulting chat/answer view is the next session.
2. **Empty / loading / error states not designed** — flagged above.
3. **Light theme tokens exist** but no light-mode mock was made. The system supports it (`.theme-light`).
4. **Density tweak** in the prototype is wired into the Tweaks panel but not yet pushed through the layouts. If your team wants a real "compact" mode, define the rules and rebuild from spacing tokens.
5. **Sharing / consent flows** are read-only here — the `REVOKE` button has no destination. Define consent revoke flow with privacy/legal.
6. **Genome and Wearables modules** are surfaced as nav items but have no detail screens yet.
7. **Logo files retain `prism42-` prefixes** — rename or alias when you import.

---

For questions: refer back to the HTML prototype. Open `Personalized Medical Records.html` in any modern browser — no build step needed.
