# Records OS — responsive layout fix

**Date:** 2026-05-06
**Branch:** `ui/records-os-responsive`
**Scope:** `web/app/records/RecordsOS.tsx` + `web/app/globals.css`

## Problem

User screenshot at half-screen viewport (~1100-1300px) showed:

- Patient name "Maya Okafor" wrapped letter-by-letter / syllable-by-syllable
  in the hero card, because the patient card was sharing center-column
  width with a 360px right rail, leaving ~280-390px for the name +
  6 KeyVal stats + Stripe.
- Signal-of-day prose ("Your LDL dropped 50 points in 24 months. The
  statin is working.") wrapped down a thin column at 22px font.
- 3-column layout (220px nav | 1fr main | 360px AI rail) didn't degrade
  at all between 1024-1440px — the inline grid template had no
  responsive fallback.
- Vitals strip had 6 fixed columns (`repeat(6, minmax(0, 1fr))`) which
  crushed each vital cell into ~80px at narrow widths.

## Fix — strategy

Inline `style={...}` props in the existing component took precedence
over any class-based CSS, so the responsive overrides live in
`globals.css` as `@media` blocks scoped to `[data-records-os-*]`
attribute selectors with `!important`. The TSX gets data-attributes
on the breakpoint-sensitive elements; no inline style is removed, so
the desktop layout at 1440px+ is byte-for-byte unchanged.

Additionally:

- `min-width: 0` added to flex children that previously could overflow
  (left rail aside, right rail aside, patient-name parent div, signal
  card, signal-figure parent).
- `text-wrap: balance` + `word-break: normal` + `overflow-wrap: normal`
  on the patient-name element so single-syllable wraps stop happening.
- `flex-wrap: wrap` added to the signal-of-day inner row (figure +
  sparkline) so the sparkline drops below the "92 mg/dL" label rather
  than crushing.
- `clamp()` on patient-name (20-36px), signal headline (15-22px),
  signal figure (28-48px) so type scales with viewport between
  breakpoints rather than hopping discretely.

## Breakpoints

| Range            | Layout                                                                |
| ---------------- | --------------------------------------------------------------------- |
| ≥ 1280px         | Full 3-col (left rail \| main \| right rail). Unchanged from main.    |
| 1024-1279px      | Right rail stacks below main (full-width row). Right resizer hidden. Vitals → 3 col. Hero cards stack. Labs+timeline rows stack. Patient-name + signal text scale via clamp(). |
| < 1024px         | Left rail also stacks above main. Both resizers hidden. Vitals → 2 col. Main padding shrinks 24/28 → 16/16. |

## Files changed

- `web/app/globals.css` — added 90 lines under existing overflow guard:
  two `@media` blocks (≤1279, ≤1023) with `!important` overrides on
  `data-records-os-*` selectors and `aria-label` selectors for the
  Resizer separators.
- `web/app/records/RecordsOS.tsx` — added data-attributes:
  `data-records-os-rail="left"|"right"`, `data-records-os-hero`,
  `data-records-os-vitals`, `data-records-os-row="labs-meds"|"timeline-shares"`,
  `data-records-os-topbar`, `data-records-os-main-content`,
  `data-records-os-patient-name`, `data-records-os-signal-headline`,
  `data-records-os-signal-figure`. Added `min-width: 0` to the patient
  card inner div, signal card, and signal figure div. Added
  `text-wrap: balance` + `word-break: normal` + `overflow-wrap: normal`
  to the patient-name span. Added `flex-wrap: wrap` to the signal-of-day
  figure+sparkline flex row.

## Why not Tailwind responsive variants

The component uses inline style props (CSS-in-JS) end-to-end — adopting
Tailwind for one element would create a styling-strategy seam that
violates the repo's existing styling vocabulary. Targeted media queries
in `globals.css` keep the seam at the layout boundary only and require
zero churn in the per-cell visual styling (which is the bulk of the
component). No new dependencies.

## Verification

- `npx tsc --noEmit`: clean.
- `npx next build`: clean (Turbopack, Next 16.2.4). All routes built.
- Desktop unchanged: media queries are `max-width:1279px` only.
  At ≥ 1280px every override is dormant and the inline styles win
  unmodified.

## Reproduction

1. Open `https://medomni.vercel.app/4UWHAt`.
2. Resize the window from 1440px wide to 1100px.
3. Before fix: patient name letters orphan onto separate lines, signal
   prose snakes down a 280px column, vitals cells crush.
4. After fix: at 1280px exactly, the right rail (Ask + Care Team +
   Activity) drops below main as a full-width row. Hero cards stack
   so the patient card has the full main width. Vitals reflow 6 → 3.
   Patient name scales smoothly via clamp() and stays on 1-2
   visually-balanced lines.
