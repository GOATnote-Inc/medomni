# Mobile-First Redesign — medomni Records OS

**Path:** `web/app/records/RecordsOS.tsx` (live at `https://medomni.vercel.app/4UWHAt`, reverse-proxied at `https://www.thegoatnote.com/4UWHAt`)
**Date:** 2026-05-06
**Status:** SPEC ONLY — no code changes in this PR. Implementation will land as ~6 follow-up PRs, one per pattern.
**Sibling reference:** `web/app/skills/page.tsx` (already mobile-friendly via single-stream + `auto-fit minmax(280px,1fr)`; can be used as a "we already shipped this pattern, here is the precedent" reference for reviewers).
**Predecessor PR:** [#101 ui(records-os): responsive layout — fix half-screen smushing](https://github.com/GOATnote-Inc/medomni/pull/101) merged 2026-05-06; ships responsive scaffolding at 1280/1024 breakpoints but underlying architecture is still 3-column-shrink, not mobile-first card-stack.

---

## 1. Executive summary

The Records OS today is a desktop-first three-column layout (220px left rail · 1fr main · 360px AI rail) with two `@media (max-width: …)` rescues that stack the rails below the main column at 1280px and 1024px. PR #101 removed the worst smushing artifacts (orphaned letters in the patient name, signal prose snaking down a 280px column, vitals cells crushed to 80px), but the page still presents desktop information density to a 390px iPhone viewport. The minimum useful width — i.e. the width below which content stops being legible without zoom — sits between 360-400px today, which is the width of every modern phone. Clinicians at the bedside cannot use the product on the device they actually hold.

The redesign reframes Records OS as a **single content stream by default** — one column of cards on phone, two-column on tablet, three-column on desktop — and layers responsiveness on top of the existing visual identity (Space Grotesk + JetBrains Mono + `#ff0096` magenta + sharp 0/2/4px radii) without changing the brand. We adopt ten patterns drawn from Apple HIG, Material 3, Netflix Hawkins, Stripe, Spotify, Apple Health, and the WCAG 2.2 + WAI carousel guidance, all cited inline. The benchmark is not "responsive enough"; it is the level shipped by Netflix, Stripe, and Apple Health, which all collapse complex desktop information architectures into thumb-reachable single streams without losing density on larger windows.

This SPEC is the deliverable for this PR. Implementation lands as a sequenced 6-PR series (§5) so each pattern is reviewable in isolation, can be reverted independently, and is small enough (~150 LOC) to be dispatched as an agent task. The first PR adds the mobile-first scaffolding (single-column at <768px, bottom tab bar) without touching desktop layout; the last PR does the polish pass (skeletons, `prefers-reduced-motion`, focus-visible). Every claim in §2 cites a primary source URL. Where a claim could not be verified in the cited spec, it is omitted.

---

## 2. Principles — the ten patterns

Each pattern lists: the rule, the source, why it applies to medomni.

### 2.1 Mobile-first, not mobile-rescued

Design the smallest viewport first; layer up. The Records OS today does the inverse — it designs at 1440px and adds `@media (max-width: …)` rescues. Material 3 and Apple Health both treat a phone column as the canonical layout and scale up by adding panes, not by shrinking columns.

- Material 3 Adaptive Design uses **window size classes — Compact (<600dp) / Medium (600-840dp) / Expanded (≥840dp)** — and the design starts at Compact, then progressively introduces a list-detail pane on Medium, then a navigation rail or persistent drawer on Expanded. ([m3.material.io/foundations/adaptive-design](https://m3.material.io/foundations/adaptive-design))
- Apple Health is the canonical EHR-shaped reference: the iPhone Summary tab is a single scrolling column; on iPad the same content surfaces inside a sidebar split-view, but the **summary cards themselves are unchanged**. ([Apple Support: Get started with Health on iPad](https://support.apple.com/guide/ipad/get-started-with-health-ipadf82bbc87/ipados))

**Why for medomni:** physicians at the bedside hold a phone. The whole product needs to work there first. Once the Compact layout is correct, the tablet and desktop layouts are *additions*, not *exceptions*.

### 2.2 Single content stream by default

On Compact, render every section in one column in editorial order (most-important first). No two-column tricks under 768px.

- Stripe shipped this pattern when it migrated the merchant Dashboard to mobile-responsive: complex tables, filter rails, and modals were reduced to a single vertical stream of cards per route, and after launch *"visits from mobile devices grew sharply, and experience quality improved substantially as measured by support volume, task success, and session length."* ([mattstromawn.com/projects/stripe-dashboard](https://mattstromawn.com/projects/stripe-dashboard/))
- Stripe's iPhone app product writeup cites the same principle: KPIs become full-width cards stacked vertically; charts get full width; everything else collapses. ([medium.com/swlh/exploring-the-product-design-of-the-stripe-dashboard-for-iphone](https://medium.com/swlh/exploring-the-product-design-of-the-stripe-dashboard-for-iphone-e54e14f3d87e))

**Why for medomni:** PR #101 preserves the 1.2fr/1fr two-column hero on phones via the data-attribute fallback. That is wrong for Compact — it produces a 180px-wide signal card that cannot show its value (`50 pts ↓`) at the 28-48px clamp size without the headline truncating. The fix is to stack hero + signal on Compact, full width each.

### 2.3 Auto-fit grids, not media-query columns

Card grids should use `repeat(auto-fit, minmax(<min>, 1fr))` so the column count is a function of the *card's* minimum width, not the *viewport's* width. This eliminates the 6→3→2 column staircase the Vitals strip uses today.

- Pattern is canonical CSS Grid; auto-fit collapses empty columns so the existing items grow to fill the row, while auto-fill keeps empty tracks. For dashboards, **auto-fit is correct** because we always have ≥1 item per section. ([CSS-Tricks: Auto-Sizing Columns in CSS Grid](https://css-tricks.com/auto-sizing-columns-css-grid-auto-fill-vs-auto-fit/), [MDN: Auto-placement in grid layout](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Grid_layout/Auto-placement))
- Card minimum widths should be chosen by content, not viewport. For Vitals, 280px is the smallest width where the value (`126 / 81`), unit, and 7-day sparkline read together; we use 280px as the floor. For Labs/Meds, the per-row content is taller and 320px is the right floor.

**Why for medomni:** the current `gridTemplateColumns: "repeat(6, minmax(0, 1fr))"` literal is what crushed Vitals to 80px-per-cell at 1100px. Replacing with `repeat(auto-fit, minmax(280px, 1fr))` makes the grid responsive without any media query and gives 1 column at iPhone SE (320px), 2 at iPhone 16 (390px), 3 at iPad portrait (768px), 4-5 on desktop.

### 2.4 Tables become cards on Compact

Five-column data tables (Labs, Conditions, Meds) cannot survive a 320px viewport without horizontal scroll, which is hostile. The pattern Stripe Dashboard, Linear, GitHub, and others use is: at <768px, each row becomes a stacked "summary card" — name on first line, value/status row, secondary metadata row, action chip — and the column headers disappear (each card carries its own labels).

- Stripe's responsive migration shipped this exact transformation. ([mattstromawn.com/projects/stripe-dashboard](https://mattstromawn.com/projects/stripe-dashboard/))
- Pattern is well-documented in the dashboard literature. ([artofstyleframe.com/blog/dashboard-design-patterns-web-apps](https://artofstyleframe.com/blog/dashboard-design-patterns-web-apps/))

**Why for medomni:** `tableHead` + `tableRow` in `RecordsOS.tsx:71-93` declare a 5-track grid `1.5fr 1fr 0.9fr 0.9fr 0.5fr`. At Compact this gives 5 tracks of ≤60px each, well below the 24×24 CSS-pixel WCAG 2.5.8 floor.

### 2.5 Sticky thumb input — bottom on mobile, top-right rail on desktop

The "Ask your record" command bar is the single most-used affordance. On mobile, it lives at `position: sticky; bottom: 0` with `env(safe-area-inset-bottom)` padding so it sits above the home indicator and rises with the iOS keyboard. On desktop, it stays in its current top-right-rail position. Same component, different anchor.

- iOS Safari: `position: fixed; bottom: 0` works because Safari auto-scrolls the visual viewport when an input is focused, pushing the fixed element above the keyboard. The bottom-anchored CTA pattern is canonical (Stripe Checkout, Apple Pay sheet, every messaging app). ([saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios](https://saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios), [bram.us — VirtualKeyboard API](https://www.bram.us/2021/09/13/prevent-items-from-being-hidden-underneath-the-virtual-keyboard-by-means-of-the-virtualkeyboard-api/))
- `env(safe-area-inset-bottom)` is a no-op on devices without a home indicator, so the same rule works on iPhone SE and iPhone 16 Pro Max.
- For Chrome Android and progressive enhancement, the **VirtualKeyboard API** with `keyboard-inset-height` env variable removes the visual viewport hack entirely. ([MDN: VirtualKeyboard API](https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard_API))

**Why for medomni:** the `AskYourRecord` component ships in the right rail at `>=1280px`. On Compact today it lives ~1200px down the page and is unreachable without scroll-to-top. Sticky-bottom placement matches every chat UI users already know.

### 2.6 Bottom tab bar on Compact, left rail on Expanded

The 13-item left rail nav (Overview / Timeline / Labs / Medications / Conditions / Vitals / Imaging / Wearables / Visit notes / Care team / Genome / Sharing / Receipts) is wrong for Compact in two ways: it doesn't fit (13 items × 32px = 416px tall, takes half a phone screen), and most items are secondary ("Receipts"). The Material 3 Compact pattern is: pick 3-5 primary destinations, surface them in a bottom navigation bar; everything else moves to a "More" sheet.

- Material 3 navigation guidance: *"a bottom navigation bar on small displays, a navigation rail on medium-size displays, or a persistent navigation drawer on large displays."* ([m3.material.io/components/navigation-bar/guidelines](https://m3.material.io/components/navigation-bar/guidelines))
- Apple Health uses exactly 3 tabs on iPhone (Summary, Sharing, Browse) and a sidebar on iPad. ([support.apple.com/en-us/104997](https://support.apple.com/en-us/104997))
- Spotify's tablet redesign uses a collapsible rail that *expands or collapses to a nav-rail* per orientation; same component, different mode. ([newsroom.spotify.com/2026-04-16/new-tablet-app-experience](https://newsroom.spotify.com/2026-04-16/new-tablet-app-experience/))

**Why for medomni:** medomni's 13 items are a textbook case for bottom nav. Pick the 4 primary tabs — **Overview · Labs · Meds · Ask** — push the other 9 into a "More" tray. Identical desktop rail stays.

### 2.7 Progressive disclosure

Show the top-N most-relevant items by default; "Show more" reveals the rest. On Compact, "RecentActivity" is currently rendered as 12 rows; that's 12 × ~56px = 672px of secondary content above the fold. Default to 5 (matching Apple Health's "Highlights") + lazy-load.

- Nielsen Norman Group, *Progressive Disclosure*: *"defer secondary options to a subsidiary screen, focusing users' attention on the primary options, which are the only ones shown by default."* ([nngroup.com/articles/progressive-disclosure](https://www.nngroup.com/articles/progressive-disclosure/))
- Apple Health Summary tab uses pinned highlights + "Show All" — same pattern. ([support.apple.com/en-us/104997](https://support.apple.com/en-us/104997))

**Why for medomni:** every section that today renders >5 items (Labs has 6, Medications has 5, RecentActivity has 12, Conditions has 4) gets a default cap of 5 and a "Show all (n)" link.

### 2.8 Fluid type with `clamp()`

Move all section-level type to `clamp(min, preferred-with-vw, max)` so a single rule covers 320-1920px. PR #101 already adopted this for the patient name (20-36px) and signal headline (15-22px); extend to all heading levels and the eyebrow tracking.

- The fluid-typography pattern has been canonical since 2020 and is documented at [Utopia.fyi](https://utopia.fyi/type/calculator/) and the OddBird "Reimagining Fluid Typography" piece ([oddbird.net/2025/02/12/fluid-type](https://www.oddbird.net/2025/02/12/fluid-type/)).
- WCAG accessibility note: *"if the maximum font size is less than or equal to 2.5 times the minimum font size, then the text will always pass WCAG SC 1.4.4 accessibility standards on all modern browsers."* ([clampgenerator.com/blog/best-font-size-clamp-generator](https://clampgenerator.com/blog/best-font-size-clamp-generator/))

**Why for medomni:** the existing `--fs-12 / 14 / 16 / 18 / 20 / 24 / 30 / 36 / 48 / 64` tokens are *step* values, not *fluid* values. We add a parallel `--fs-fluid-*` set keyed to viewport (see §4.1).

### 2.9 44pt touch targets, padding counts

Every interactive element must be at least 44×44 CSS points (Apple HIG) or 48×48dp (Material). WCAG 2.2 SC 2.5.8 (Level AA, 2023) sets a *floor* of 24×24 CSS pixels, but Apple HIG and Material both push the target higher and we adopt 44px as the medomni minimum.

- Apple HIG: *"Touch targets require minimum dimensions of 44×44 points as research demonstrates that smaller interactive elements result in 25% or higher tap error rates."* ([Apple HIG — Layout](https://developer.apple.com/design/human-interface-guidelines/layout))
- WCAG 2.2 SC 2.5.8: *"The target area for pointer inputs must be at least 24 by 24 CSS pixels"* — note that **padding counts toward target size**. ([w3.org/WAI/WCAG22/Understanding/target-size-minimum](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html))

**Why for medomni:** today's `btnGhost` (`padding: "7px 12px", fontSize: 11.5`) yields a ~30×26px hit target. Below the WCAG floor and well below Apple HIG. Audit pass: every clickable element gets `min-height: 44px` + `min-width: 44px` (or compensating spacing).

### 2.10 Skeleton screens, not spinners

While AskYourRecord's RAG retrieval (or any LanceDB / FHIR fetch) is in flight, show a card-shaped skeleton matching the eventual layout. Spinners feel slower than skeletons even when actual load times are identical.

- *"Research by Viget found that users consistently rate experiences with skeleton screens as 20% faster than identical wait times with traditional loading spinners."* ([clay.global/blog/skeleton-screen](https://clay.global/blog/skeleton-screen))
- NN/g: *Skeleton Screens 101*. ([nngroup.com/articles/skeleton-screens](https://www.nngroup.com/articles/skeleton-screens/))

**Why for medomni:** every Compact card that depends on a network fetch (PatientHero on patientId switch, AskYourRecord answer, future FHIR-fetch results from the Pattern B spike at p95=11ms) gets a skeleton variant.

---

## 3. Section-by-section redesign

The current section list, in editorial order on Compact:

1. **PatientHero** — name, age/sex/MRN, DOB, allergies, code-status pill, primary care, share button.
2. **ActiveSignal** — the "Your LDL dropped 50 points in 24 months. The statin is working." card.
3. **AskYourRecord** — command bar + answer panel (today: top of right rail).
4. **Vitals** — 6-cell strip (BP, HR, RR, SpO2, Temp, Wt).
5. **Labs** — 5-column table (Test, Value, Range, Date, Trend).
6. **Conditions** — 4 entries with status chip.
7. **Medications** — 5 rows with dose + adherence.
8. **Imaging** — `ImagingPanel` thumbnails.
9. **Wearables** — Apple Health-style ring + 7-day sparkline.
10. **CareTeam** — 5 avatars + names + roles.
11. **RecentActivity** — 12 timeline rows.

The breakpoint contract (final, supersedes PR #101's two-tier):

| Breakpoint  | Range          | Layout                                                      |
| ----------- | -------------- | ----------------------------------------------------------- |
| **Compact** | < 768px        | Single column. Bottom tab bar. Sticky-bottom Ask.           |
| **Medium**  | 768-1023px     | 2-column auto-fit grid (where the section can split). Left rail collapses to icon-only nav (Material 3 navigation rail). |
| **Expanded**| 1024-1279px    | 2-column main + AI rail stacks below (matches PR #101 1024-1279px tier).                          |
| **Large**   | ≥ 1280px       | 3-column (220px nav · 1fr main · 360px AI rail). **Byte-for-byte unchanged from PR #101's 1280px+ layout.** |

Compact and Medium are *new* tiers. Expanded and Large match PR #101.

### 3.1 PatientHero

| Viewport | Today                                                                | Target                                                              |
| -------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 320-767  | 1.2fr/1fr two-column inside a single hero card; data-attr stack at <1024 only | Single column. Avatar + name on top, vital-status row below, allergy chips on a third row, share button full-width below. Patient name at `clamp(22px, 5.5vw, 36px)`. |
| 768-1023 | (no path)                                                            | 2-up: name+demographics card | code-status+allergies card. Share button moves into card 1's header. |
| 1024-1279| Hero stacks above main; 1.2fr/1fr internal split kept                | Same as today (Expanded tier from PR #101). |
| ≥1280    | Full hero unchanged                                                  | Unchanged.                                                          |

Hero typography clamp ranges (already in `globals.css` from PR #101):
- Patient name: `clamp(20px, 4.5vw, 36px)` → tighten min to 22px on Compact since the name has its own card width budget.
- "Code Status: FULL CODE" pill stays mono 11px as-is — the pill itself ≥44px tall via padding.

### 3.2 AskYourRecord

This is the keystone change. Same `<AskYourRecord>` component, two different mounting positions controlled by a single `<media (min-width: 1024px)>` boundary.

| Viewport | Today                                                            | Target                                                              |
| -------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| 320-767  | Renders inside the right-rail card stack at the bottom of the page | `position: sticky; bottom: 0` with `padding-bottom: env(safe-area-inset-bottom)`. Collapsed by default to a single 56px-tall input + magenta send button; expands to a sheet on focus. The answer panel (when populated) renders as a card *above* the input, scroll-anchored so the input stays in view. |
| 768-1023 | (no path — Medium tier is new)                                   | Sticky-top inside main column at `top: 0`, full-width. Behaves like the desktop right-rail Ask but full width.            |
| ≥1024    | Right rail, scrolls with page                                    | **Unchanged** — keeps PR #101 layout.                              |

Implementation detail (for the future PR brief, not this SPEC's deliverable): the iOS keyboard handling is the gnarliest part. The Visual Viewport API + `env(keyboard-inset-height)` (Chromium/Edge with VirtualKeyboard API) covers most cases; iOS Safari auto-scrolls the visual viewport when an input is focused inside a `position: fixed; bottom: 0` element, so the same CSS works on iOS without JS. Cite [bram.us VirtualKeyboard API](https://www.bram.us/2021/09/13/prevent-items-from-being-hidden-underneath-the-virtual-keyboard-by-means-of-the-virtualkeyboard-api/) and [saricden iOS keyboard fix](https://saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios) in the PR description.

### 3.3 ActiveSignal

| Viewport | Today                                            | Target                                                  |
| -------- | ------------------------------------------------ | ------------------------------------------------------- |
| 320-767  | Compresses to a 180px-wide column inside hero    | Full-width card directly below PatientHero. Headline `clamp(15px, 4vw, 22px)`. The 50-point figure scales `clamp(28px, 9vw, 48px)`. Sparkline gets full card width. |
| 768-1023 | (no path)                                        | Same as Compact — full-width card. ActiveSignal is editorial, not a sidebar widget. |
| ≥1024    | Sits next to PatientHero in the 1.2fr/1fr split  | Unchanged.                                              |

### 3.4 Vitals

| Viewport | Today                                                                | Target                                                                |
| -------- | -------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 320-767  | `repeat(6, minmax(0, 1fr))` → 6 cells × ~50px each, illegible        | `grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px;` → renders as 1 column at iPhone SE (320), 1 column at iPhone 16 (390), 2 columns at iPhone 16 Pro Max (430). Each cell shows label + value + sparkline + trend. |
| 768-1023 | 6→3 col fallback from PR #101                                        | Same `auto-fit minmax(280px, 1fr)` → 2-3 cols depending on width.    |
| 1024-1279| 6→3 col                                                              | Same rule → 3-4 cols.                                                |
| ≥1280    | 6 cols                                                               | Same rule → 5-6 cols (depends on container width since rail).       |

The single `auto-fit minmax(280px, 1fr)` rule **replaces all four media-query tiers** and produces a smoother visual progression than the staircase.

### 3.5 Labs

This is the table-to-card transformation.

| Viewport | Today                                                                 | Target                                                                       |
| -------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 320-767  | 5-track grid at ~60px each track; sparkline crushed                   | Each lab is a card: line 1 `Test name | Value [unit] [▲/▼ delta]`, line 2 `Range … | Date`, line 3 sparkline (full width). Range ribbon (RangeBar) becomes a horizontal bar above sparkline. Tap → `<DetailDrawer>` (already exists). |
| 768-1023 | 5 tracks                                                              | Auto-fit grid `minmax(320px, 1fr)` → 2-up cards.                            |
| ≥1024    | 5 tracks                                                              | Keep table on Expanded+. The 5-column grid works ≥1024px because the column gets a real budget. |

Note: Labs is the only place in the redesign where the **layout primitive itself swaps** (table ↔ card) at the 1024px boundary. Conditions and Meds use the same swap. To keep a single component, render both DOMs and toggle visibility via CSS `display: none` at the breakpoint — same approach Stripe Dashboard uses. ([mattstromawn.com/projects/stripe-dashboard](https://mattstromawn.com/projects/stripe-dashboard/))

### 3.6 Conditions

Same table-to-card pattern as Labs. Default cap **5** with "Show all (n)" — today shows 4, so no cap is exercised yet, but the affordance ships now to avoid a regression when the patient context grows.

### 3.7 Medications

Same table-to-card pattern. Card front: drug name (display 16px) | dose (mono 12px) | adherence chip. Card back (drawer): refill date, prescriber, change history.

### 3.8 Imaging

`ImagingPanel` already renders thumbnails in a flex layout. Audit and convert to `auto-fit minmax(180px, 1fr)`.

### 3.9 Wearables

Wearables card in today's design has a ring chart (Apple Health analog) + 7-day sparkline + macros. The ring scales naturally; we constrain max-width 280px on Compact and let the macros (Steps / Sleep / HRV) wrap as a 3-up grid below.

Cite Apple Health iPad: *"the larger screen of the iPad lets you see far more data in a single page"* ([Apple Support: View your data in Health on iPad](https://support.apple.com/guide/ipad/view-your-health-data-ipadda7b012d/ipados)).

### 3.10 CareTeam

| Viewport | Today                                                  | Target                                                                                            |
| -------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| 320-767  | 5 avatars wrap, each ~80px wide, names truncate        | Horizontal scroll-snap rail. `scroll-snap-type: x mandatory; overflow-x: auto; flex-wrap: nowrap`. Each chip is a 200px-wide vCard (avatar + name + role + tap-to-message). Scroll markers below (`::scroll-marker` if available, else dots). |
| ≥768     | 5-up flex                                              | Vertical list inside CareTeam card (one row per member, full name+role). |

The horizontal scroll-snap rail is the **Netflix Hawkins card-rail pattern** ([netflixtechblog.com/hawkins-diving-into-the-reasoning-behind-our-design-system](https://netflixtechblog.com/hawkins-diving-into-the-reasoning-behind-our-design-system-964a7357547)) and the modern CSS scroll-snap implementation is documented at [Sara Soueidan: Are CSS Carousels accessible?](https://www.sarasoueidan.com/blog/css-carousels-accessibility/) and the [Chrome accessible-carousel guide](https://developer.chrome.com/blog/accessible-carousel). ARIA contract:

- Rail container: `role="region" aria-roledescription="carousel" aria-label="Care team"`.
- Each item: tab-stop with `aria-label="Dr. Lina Cho, primary care"`.
- `interactivity: inert` on off-screen items (per [MDN: CSS Carousels](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Overflow/Carousels)) to keep tab order sane on Compact.

### 3.11 RecentActivity

Default 5 entries + "Show all (n)" link → expands inline. After 20, paginate by date. Pattern is straight progressive disclosure ([NN/g: Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/)).

### 3.12 Left rail nav → bottom tab bar

| Viewport | Today                                                | Target                                                                |
| -------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| < 768    | Stacks above main (PR #101 < 1024 fallback)          | **Bottom tab bar.** 4 tabs: Overview · Labs · Meds · Ask. `position: fixed; bottom: 0; padding-bottom: env(safe-area-inset-bottom)`. 56px tall. Each tap = scroll to corresponding `id="section-…"` anchor. The remaining 9 nav items (Imaging / Wearables / Visit notes / Care team / Genome / Sharing / Receipts / Conditions / Vitals / Timeline) move into a "More" sheet — `<details>` opens a bottom sheet with the full nav. |
| 768-1023 | (PR #101 stacks above main)                          | Icon-only navigation rail at left, 64px wide. Labels appear on hover. (Material 3 medium pattern.) |
| ≥1024    | 220px left rail with all 13 items                    | Unchanged.                                                            |

Tab selection sync: scroll-driven IntersectionObserver flips `aria-current="page"` on the bottom tab whose `id` matches the most-visible section, mirroring the desktop rail's existing behavior. (Existing logic at `RecordsOS.tsx:1083-1091` already does the section→nav lookup.)

---

## 4. Token system spec

### 4.1 Type scale — fluid additions

Keep all existing `--fs-*` step tokens (already used inline). Add a parallel `--fs-fluid-*` set for **headings and section labels only**. Body copy stays at the step values (more predictable for clinicians who read records as text).

```
/* additive — does not replace existing --fs-* tokens */
--fs-fluid-eyebrow: clamp(10px, 0.65vw + 8px, 12px);
--fs-fluid-body:    clamp(14px, 0.4vw + 12.5px, 16px);
--fs-fluid-h3:      clamp(15px, 1.1vw + 12px, 22px);   /* section headings */
--fs-fluid-h2:      clamp(20px, 2.2vw + 14px, 30px);   /* card headlines */
--fs-fluid-h1:      clamp(22px, 3.5vw + 14px, 36px);   /* patient name */
--fs-fluid-figure:  clamp(28px, 5vw + 16px, 48px);     /* signal value */
```

Each clamp range satisfies max ≤ 2.5 × min (WCAG SC 1.4.4 reflow guarantee per [clampgenerator.com](https://clampgenerator.com/blog/best-font-size-clamp-generator/)). Verify by computation in the implementing PR.

### 4.2 Spacing — already correct

The existing `--sp-1 / 2 / 3 / 4 / 5 / 6 / 8 / 10 / 12 / 16 / 20 / 24 / 32` 4-pt scale ports cleanly. No changes.

### 4.3 Color — no changes

The existing palette (black/coal/iron/steel/ash/fog/bone/paper/snow/white + `--p42-pink #ff0096`) is the brand. Mobile redesign does not touch colors. Add only `--p42-skeleton: rgba(255,255,255,0.04)` and `--p42-skeleton-shine: rgba(255,255,255,0.08)` for the shimmer in §2.10.

### 4.4 Elevation tokens — already correct

`--shadow-1 / 2 / 3` and `--glow-pink / soft` cover everything needed for the skeleton + sticky-bottom drop shadow. Add `--elev-sticky: 0 -8px 24px rgba(0,0,0,0.6)` for the sticky-bottom AskYourRecord (shadow points up).

### 4.5 Container queries vs viewport breakpoints — decision

**Decision: viewport breakpoints for layout, container queries inside cards.**

Rationale: container queries (`@container (min-width: 320px)` + `cqi`/`cqb` units) are the right tool when a component lives in containers of varying width — e.g. a Vital cell rendered both at 280px (Compact) and 480px (Expanded with rail). But the **page-level** layout decisions (Compact vs Medium vs Expanded vs Large) are genuinely a function of viewport, not a function of any particular container, and viewport breakpoints are what PR #101 already uses. Mixing both buys nothing.

We adopt `@container` queries inside each card for *internal* layout (e.g. Vital cell switching from `column` to `row` when its container exceeds 380px) but keep `@media` for the four global tiers.

Source: [LogRocket: Container queries in 2026](https://blog.logrocket.com/container-queries-2026/) — *"Choose container breakpoints that reflect component needs (e.g., when labels wrap) rather than viewport sizes, while keeping some page-level media queries for global layout shifts."*

### 4.6 Two new design tokens

```
--bp-compact-max: 767px;
--bp-medium-max: 1023px;
--bp-expanded-max: 1279px;
/* > 1280 = Large, no token needed since it's the default */
```

Used only inside `globals.css` so that breakpoints have one source of truth.

---

## 5. PR sequence

Each PR is scoped, small (~150 LOC), independently revertible, and includes a one-paragraph "agent brief" so it can be dispatched as a task. Order is dependency-respecting.

### PR-1: tokens + breakpoint constants (no UI change)

**Title:** `tokens(records-os): add fluid type scale + breakpoint constants`
**Files:** `web/app/globals.css` only.
**Adds:** `--fs-fluid-*` tokens (§4.1), `--bp-*` constants (§4.6), `--p42-skeleton*` (§4.3), `--elev-sticky` (§4.4). No DOM changes anywhere.
**Verifies:** `npx tsc --noEmit` clean, `npx next build` clean, visual diff at 1440px = zero (only added unused vars).
**Agent brief:** *"Add the fluid-type-scale CSS custom properties + breakpoint constants listed in SPEC.md §4.1 and §4.6 to `web/app/globals.css`. Place them after the existing `--fs-*` block. No other files. No DOM changes. Run `npx tsc --noEmit` + `npx next build` + verify both pass before committing."*
**LOC est:** +20.

### PR-2: Compact bottom tab bar (mobile-only, dormant ≥768px)

**Title:** `ui(records-os): bottom tab bar on Compact (<768px)`
**Files:** `web/app/records/RecordsOS.tsx` (add component), `web/app/globals.css` (sticky-bottom + safe-area).
**Adds:** `<BottomTabBar>` component rendered always; CSS hides it ≥768px. Includes 4 tabs (Overview/Labs/Meds/Ask) + "More" sheet trigger. Wires existing `getElementById("section-…")` scroll-to-anchor logic.
**Risk:** none on desktop (display: none). On Compact, current left rail also stays (per PR #101) — both visible until PR-3 hides the left rail on Compact.
**Verifies:** at 1440px no visible change (DOM gain ~80 nodes hidden via CSS); at 390px the bar appears, all 4 tabs target ≥44×44 (§2.9).
**Agent brief:** *"Implement §3.12 BottomTabBar from SPEC.md: sticky bottom 56px, 4 tabs + More, render always, hide ≥768px via `globals.css`. Use existing scroll-to-anchor logic at RecordsOS.tsx:1083. Each tab target must be ≥44×44 CSS px including padding. Add `padding-bottom: env(safe-area-inset-bottom)`."*
**LOC est:** ~150.

### PR-3: Compact single-column hero + signal stack

**Title:** `ui(records-os): single-stream hero+signal on Compact`
**Files:** `RecordsOS.tsx` (add `data-records-os-hero-stack` attribute), `globals.css` (add Compact rule).
**Removes:** the 1.2fr/1fr `gridTemplateColumns` on PatientHero+ActiveSignal at <768px. Each becomes full-width.
**Also:** hides the left rail at <768px (now that PR-2 ships the bottom tab bar). Sets `--records-main-padding: 12px` at <768px to claw back horizontal space.
**Verifies:** screenshots at 320 / 390 / 430 / 768. Patient name reads on 1-2 lines without orphans. ActiveSignal headline + figure both render at clamp max.
**LOC est:** ~70.

### PR-4: auto-fit grids for Vitals + Imaging

**Title:** `ui(records-os): auto-fit grids replace fixed-column Vitals + Imaging`
**Files:** `RecordsOS.tsx` lines around 1352, ImagingPanel component.
**Replaces:** `repeat(6, minmax(0,1fr))` and equivalent in Imaging with `repeat(auto-fit, minmax(280px, 1fr))` (Vitals) and `minmax(180px, 1fr)` (Imaging). Removes the PR #101 `@media` overrides for these two sections.
**Verifies:** at every breakpoint 320/390/768/1024/1280/1440/1920, column count is `floor((available-width + gap) / (min + gap))` and columns scale evenly. No more crushing at 1100px.
**LOC est:** ~40.

### PR-5: table-to-card swap for Labs / Conditions / Meds

**Title:** `ui(records-os): card layout for Labs/Conditions/Meds on Compact`
**Files:** `RecordsOS.tsx` (Labs/Conditions/Meds sections). Renders both DOMs with `display: none` toggle at 1024px.
**Adds:** `<LabCard>`, `<ConditionCard>`, `<MedCard>` mobile DOMs alongside existing table DOMs. CSS shows table at ≥1024 and cards at <1024. Each card includes the 5 fields the table has.
**Risk:** doubles the rendered DOM size for these sections (+ ~30 nodes per row). Acceptable — these sections are virtualized at >50 rows in v2; we're well below.
**Verifies:** parity test — same data renders in table and card at the boundary; tap-target audit passes 44×44.
**LOC est:** ~180.

### PR-6: AskYourRecord sticky-bottom on Compact

**Title:** `ui(records-os): sticky-bottom AskYourRecord on Compact`
**Files:** `web/components/4uwhat/AskYourRecord.tsx`, `globals.css`.
**Adds:** new mount-mode prop `dock="bottom" | "rail"` (defaults to "rail" to keep desktop unchanged). At <768 the parent passes `dock="bottom"`. CSS uses `position: sticky; bottom: 0; padding-bottom: env(safe-area-inset-bottom); box-shadow: var(--elev-sticky)`.
**Behavior on iOS:** input focus auto-scrolls visual viewport; verified on iOS 17+ Safari per [saricden.com](https://saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios). Progressive enhancement on Chromium: if `navigator.virtualKeyboard` exists, call `navigator.virtualKeyboard.overlaysContent = true` and rely on `env(keyboard-inset-height)` ([MDN VirtualKeyboard API](https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard_API)).
**Test:** open `/4UWHAt` at 390×844 (iPhone 16 dimensions), focus the Ask input, confirm input rises with keyboard and remains tappable.
**LOC est:** ~120.

### PR-7: scroll-snap CareTeam rail + 5-item RecentActivity cap + skeletons

**Title:** `ui(records-os): scroll-snap CareTeam + progressive disclosure + skeletons`
**Files:** RecordsOS.tsx (CareTeam, RecentActivity), `globals.css` (skeleton shimmer keyframes).
**Adds:**
- CareTeam: at <768 swap to horizontal scroll-snap rail per §3.10 ARIA contract.
- RecentActivity: cap to 5; "Show all (12)" button reveals rest inline.
- Skeleton variants for PatientHero, AskYourRecord answer panel, Vitals — used while data is fetching (currently fixture data so skeletons mainly cover the AskYourRecord answer state).
**Verifies:** keyboard tab traverses CareTeam rail in DOM order; off-screen items have `interactivity: inert`; `prefers-reduced-motion: reduce` disables shimmer animation.
**LOC est:** ~140.

**Total LOC across 7 PRs: ~720**, all reviewable independently. Cumulative delta vs main can land over 5-7 working days at one PR/day.

---

## 6. Migration risks

### 6.1 Sticky bottom collides with mobile keyboard

**Risk:** `position: sticky; bottom: 0` on AskYourRecord could be hidden behind the iOS keyboard.
**Mitigation:** pattern in §2.5. iOS Safari's auto-scroll behavior covers most cases. For Chrome Android we opt-in to the VirtualKeyboard API. Tested on iOS 17 Safari + Chrome Android 124 in PR-6's test plan.
**Source:** [bram.us VirtualKeyboard API](https://www.bram.us/2021/09/13/prevent-items-from-being-hidden-underneath-the-virtual-keyboard-by-means-of-the-virtualkeyboard-api/), [saricden iOS keyboard fix](https://saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios)

### 6.2 Scroll anchoring after section reorder

**Risk:** when bottom-tab tap scrolls to `#section-labs`, the Labs section may not exist in the DOM yet (lazy-loaded?) or may be display: none under a different breakpoint.
**Mitigation:** all sections render eagerly today (no lazy state); the table-to-card swap at PR-5 keeps both DOMs at the same `id`. Test plan in PR-5 includes scroll-to-anchor at every breakpoint.

### 6.3 Focus management when nav becomes a sheet

**Risk:** "More" tray opens as a `<details>` sheet — focus must move into the sheet on open and return to the trigger on close. `<details>` element handles this natively; if we use a custom sheet for animation reasons, we use `inert` + a focus trap.
**Mitigation:** prefer native `<details><summary>` for the More tray. If animation requirements rule it out, reach for a vetted primitive (Radix Dialog with `modal=false`).
**Source:** [Apple HIG — Accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility) recommends VoiceOver focus return on dismissal.

### 6.4 NVDA / VoiceOver announcements for the carousel

**Risk:** scroll-snap CareTeam rail is invisible to screen readers without ARIA scaffolding.
**Mitigation:** §3.10 ARIA contract — `role="region" aria-roledescription="carousel"`, per-item labels, `interactivity: inert` on off-screen items. Test with VoiceOver on iOS and NVDA on Windows-Chrome.
**Source:** [Sara Soueidan: Are CSS Carousels accessible?](https://www.sarasoueidan.com/blog/css-carousels-accessibility/), [Chrome blog: Make accessible carousels](https://developer.chrome.com/blog/accessible-carousel)

### 6.5 Touch target audit may regress at smaller scales

**Risk:** `clamp()` on a button label can drop the rendered height below 44px at very small viewports if padding doesn't compensate.
**Mitigation:** every interactive element gets `min-height: 44px` explicitly, independent of font size. Audit in PR-2 and PR-6.
**Source:** [Apple HIG — Layout](https://developer.apple.com/design/human-interface-guidelines/layout), [WCAG 2.2 SC 2.5.8](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html)

### 6.6 PR #101's data-attribute fallback may collide with new CSS

**Risk:** PR #101 added `[data-records-os-grid]` and `[data-records-os-hero]` attributes with `!important` rules in `globals.css` lines ~205+. Mobile-first PR-3/PR-4 will need to either (a) remove those attributes or (b) layer underneath them.
**Mitigation:** PR-3 deletes the PR #101 fallback for the hero (replaced by mobile-first rule); PR-4 deletes the PR #101 vitals fallback (replaced by `auto-fit`). Each PR explicitly mentions which PR #101 rule it supersedes.

### 6.7 Existing inline-style architecture

**Risk:** RecordsOS.tsx is end-to-end inline `style={{...}}` (per PR #101 description: *"the component is inline-style CSS-in-JS end to end"*). Mobile-first patterns require rules that respond to breakpoints — not expressible in inline styles.
**Mitigation:** keep PR #101's strategy: layout-shape rules go in `globals.css` keyed to `data-records-os-*` attributes; per-cell visual styling stays inline. The styling-strategy seam lives at the layout boundary only.

---

## 7. Test plan

### 7.1 Viewport matrix

Run each PR's reviewer against this set:

| Width | Device                     | Layout tier   | Critical assertions                                             |
| ----- | -------------------------- | ------------- | --------------------------------------------------------------- |
| 320   | iPhone SE / smallest modern| Compact       | Single column. Hero name on 1-2 lines no orphans. Vitals 1 col. Labs cards 1-up. Bottom tab bar visible. AskYourRecord sticky-bottom. No horizontal scroll. |
| 390   | iPhone 16                  | Compact       | Same. Vitals 1 col. Labs cards 1-up. Hero hierarchy reads.      |
| 430   | iPhone 16 Pro Max          | Compact       | Vitals can render 1-2 cols.                                     |
| 768   | iPad portrait              | Medium        | 2-up auto-fit grids. Icon-only nav rail at left. AskYourRecord sticky-top.            |
| 1024  | iPad landscape             | Expanded      | Matches PR #101's `1024-1279` rules. Right rail stacks below.   |
| 1280  | small laptop               | Large         | Full 3-col. **Byte-for-byte unchanged** from main.              |
| 1440  | desktop                    | Large         | **Byte-for-byte unchanged** from main.                          |
| 1920  | large monitor              | Large         | Same as 1440 (the layout caps at 1440 visual width via existing max-width on the grid). |

### 7.2 Lighthouse mobile targets

Run Lighthouse in mobile emulation against each PR's preview deploy. Targets:

| Metric                          | Today (estimated) | After PR-7 target |
| ------------------------------- | ----------------- | ----------------- |
| Performance                     | ~70-80           | ≥ 90              |
| Accessibility                   | ~85-90 (target-size flags) | 100               |
| Best Practices                  | ≥ 90             | ≥ 95              |
| LCP (Largest Contentful Paint)  | unmeasured       | < 2.5s on 4G      |
| CLS (Cumulative Layout Shift)   | unmeasured       | < 0.1             |
| INP (Interaction to Next Paint) | unmeasured       | < 200ms           |

### 7.3 Screen reader smoke test

VoiceOver on iOS 17 Safari + NVDA on Windows-Chrome:

1. Bottom tab bar announces "Overview, tab 1 of 4, selected" / "Labs, tab 2 of 4".
2. CareTeam rail announces "Care team, carousel, region. Dr. Lina Cho, primary care, 1 of 5."
3. AskYourRecord input has accessible name "Ask your record" and is reachable via tab from the page top in <10 stops.
4. "Show all 12 activities" link announces correct count + state on toggle.

### 7.4 iOS Safari keyboard test

iPhone 16 simulator (or real device) at 390×844:

1. Open `/4UWHAt`, scroll to bottom of page.
2. Tap AskYourRecord input.
3. Keyboard rises; assert input visible above keyboard.
4. Type "what's the LDL trend"; assert no layout shift.
5. Dismiss keyboard; assert input returns to sticky-bottom position.

### 7.5 Regression-on-desktop test

Manual + Playwright (if available):

1. Visit `/4UWHAt` at 1440×900.
2. Diff screenshot against `main` HEAD before each PR merges.
3. Pixel diff threshold: zero. Any non-zero diff → blocker until investigated.

---

## 8. Cross-references

### 8.1 PR #101 (predecessor)

[PR #101 ui(records-os): responsive layout — fix half-screen smushing](https://github.com/GOATnote-Inc/medomni/pull/101) shipped two `@media (max-width: …)` rules and `clamp()` on hero typography. This SPEC subsumes that work — PR-3 supersedes the `<1024px` left-rail-stack rule, PR-4 supersedes the Vitals `6→3→2` staircase. PR-5 supersedes Labs+Conditions+Meds because that section's row grids were sized for ≥1024px and the half-screen rules from PR #101 do not target them. The 1280px+ layout in PR #101 is the **frozen reference** — every PR in this SPEC asserts byte-for-byte equality with PR #101's 1280px layout.

### 8.2 Skills registry as reference implementation

[`web/app/skills/page.tsx`](https://medomni.vercel.app/4UWHAt/skills) shipped 2026-05-05 (PR #98) and is the closest existing reference inside this repo for the patterns we want. It uses `auto-fit minmax(280px, 1fr)`, single-column on Compact, `clamp()` heading, and a fluid-type display. Reviewers of PR-3/PR-4/PR-5 should diff against the skills page to validate consistency. The skills page is also already cited as Cherny-cycle / trust-through-transparency surface and inherits the same `globals.css` tokens.

### 8.3 Brand identity contract

The redesign **does not change** Space Grotesk + JetBrains Mono + `#ff0096` magenta + black/coal/iron neutrals + sharp 0/2/4px radii. Everything in §3 layers responsiveness on top of the existing visual identity. Reviewers should reject any PR that introduces a new color, a new typeface, or rounds a corner the brand specifies as sharp.

---

## 9. Citation index

Primary sources cited in this SPEC, grouped by section:

**Apple Human Interface Guidelines + Apple Health**
- [HIG Layout](https://developer.apple.com/design/human-interface-guidelines/layout) — §2.9 (44pt touch targets), §6.5
- [HIG Accessibility](https://developer.apple.com/design/human-interface-guidelines/accessibility) — §6.3
- [Health on iPhone/iPad](https://support.apple.com/en-us/104997) — §2.1, §2.6
- [View health data on iPad](https://support.apple.com/guide/ipad/view-your-health-data-ipadda7b012d/ipados) — §3.9

**Material Design 3**
- [Adaptive design](https://m3.material.io/foundations/adaptive-design) — §2.1 (Compact/Medium/Expanded window size classes)
- [Navigation bar guidelines](https://m3.material.io/components/navigation-bar/guidelines) — §2.6, §3.12

**Netflix Hawkins**
- [Hawkins: Diving into the Reasoning Behind our Design System](https://netflixtechblog.com/hawkins-diving-into-the-reasoning-behind-our-design-system-964a7357547) — §3.10 (card-rail pattern, design-system-as-shared-vocabulary)

**Stripe**
- [Stripe Merchant Dashboard responsive migration (Matt Ström-Awn)](https://mattstromawn.com/projects/stripe-dashboard/) — §2.2, §2.4, §3.5
- [Exploring the Product Design of the Stripe Dashboard for iPhone](https://medium.com/swlh/exploring-the-product-design-of-the-stripe-dashboard-for-iphone-e54e14f3d87e) — §2.2

**Spotify**
- [Spotify tablet redesign 2026-04-16](https://newsroom.spotify.com/2026-04-16/new-tablet-app-experience/) — §2.6 (collapsible rail + adaptive orientation)

**WCAG / accessibility**
- [WCAG 2.2 SC 2.5.8 Target Size Minimum](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html) — §2.9, §6.5
- [Sara Soueidan: Are CSS Carousels accessible?](https://www.sarasoueidan.com/blog/css-carousels-accessibility/) — §3.10, §6.4
- [Chrome for Developers: Make accessible carousels](https://developer.chrome.com/blog/accessible-carousel) — §3.10, §6.4

**CSS — fluid + grid + container queries**
- [CSS-Tricks: Auto-fill vs auto-fit](https://css-tricks.com/auto-sizing-columns-css-grid-auto-fill-vs-auto-fit/) — §2.3
- [MDN: Auto-placement in grid layout](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Grid_layout/Auto-placement) — §2.3
- [Utopia.fyi fluid type calculator](https://utopia.fyi/type/calculator/) — §2.8
- [OddBird: Reimagining Fluid Typography](https://www.oddbird.net/2025/02/12/fluid-type/) — §2.8
- [ClampGenerator: best font-size clamp generator](https://clampgenerator.com/blog/best-font-size-clamp-generator/) — §2.8 (WCAG SC 1.4.4 floor)
- [LogRocket: Container queries in 2026](https://blog.logrocket.com/container-queries-2026/) — §4.5
- [MDN: VirtualKeyboard API](https://developer.mozilla.org/en-US/docs/Web/API/VirtualKeyboard_API) — §2.5, §3.2, §6.1
- [bram.us — VirtualKeyboard API](https://www.bram.us/2021/09/13/prevent-items-from-being-hidden-underneath-the-virtual-keyboard-by-means-of-the-virtualkeyboard-api/) — §2.5, §6.1
- [saricden — fixed elements respect virtual keyboard](https://saricden.com/how-to-make-fixed-elements-respect-the-virtual-keyboard-on-ios) — §2.5, §6.1
- [MDN: CSS Carousels Guide](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Overflow/Carousels) — §3.10

**Progressive disclosure + skeletons**
- [Nielsen Norman: Progressive Disclosure](https://www.nngroup.com/articles/progressive-disclosure/) — §2.7, §3.11
- [Nielsen Norman: Skeleton Screens 101](https://www.nngroup.com/articles/skeleton-screens/) — §2.10
- [Clay: Skeleton Screen UX](https://clay.global/blog/skeleton-screen) — §2.10 (Viget 20% perception study)

**Mobile EHR baseline (negative-space comparator)**
- [Mindbowser: Epic Canto features and EHR integration](https://www.mindbowser.com/understanding-epic-canto/) — §3 (baseline: existing mobile EHRs are device-fork, not responsive)
- [Mayo Clinic / PMC ambient-listening study (332 PCPs, Apr 2024 - Apr 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12657781/) — physician mobile EHR usage; 4.16 vs 5.11 min per note (-18.6%) with mobile + ambient.
- [Medical Economics: 51% of physicians use tablets to access EHRs](https://www.medicaleconomics.com/view/51-physicians-use-tablets-access-ehrs-survey-shows) — Compact/Medium tier audience size justification.

---

## 10. Out of scope (this SPEC)

Explicit non-goals:

- **No new typeface, no color additions, no radius softening.** The brand identity is fixed.
- **No new dependencies.** No Tailwind migration, no Radix, no react-aria, no shadcn. PR #101's reasoning still holds: *"adopting Tailwind for one fix would create a styling-strategy seam"* — and that judgment applies to every PR in this sequence.
- **No FHIR data wiring.** This SPEC redesigns the layout. The patient data still comes from `SAMPLE_*` fixtures + `usePatientId()` per `RecordsOS.tsx:8-13`. The Pattern B FHIR-fetch (separate spike at `findings/2026-05-04-pattern-b-spike/`) is orthogonal and ships on its own track.
- **No theming work.** The `theme-light` flag exists in `globals.css` but is not exercised in production; mobile-first redesign does not change that.
- **No new client-side state.** Bottom tab bar selection, More-sheet open state, RecentActivity expand state — all live in component-local `useState`. No global store, no URL param plumbing in this scope.
- **No PWA manifest changes.** Adding to-home-screen install prompt or offline-first behavior is a separate effort once mobile-first ships.

---

*End of SPEC. PRs PR-1 through PR-7 implement; this document is the contract they execute against.*
