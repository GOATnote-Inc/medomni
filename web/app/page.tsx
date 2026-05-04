// `/` is now the agent surface. The simple v0 ASK form lives at /classic.
//
// Why the swap (2026-05-03 PM): /agent shipped to 4 sovereign tools
// (guideline currency, calculators, PubMed, PrimeKG) with parallel
// dispatch and tool cards, and is the actual product. The on-ramp
// shouldn't be simpler than the demo. Visitors land directly on the
// real surface; deep-link history at /agent still resolves the same way
// because Next.js file-based routing means both URLs serve this page.
export { default } from "./agent/page";
