import type { NextConfig } from "next";

// Mounted at /4UWHAt because the canonical demo URL is
// https://www.thegoatnote.com/4UWHAt — v0-goat-note-landing-page-3c (which
// owns www.thegoatnote.com on Vercel) reverse-proxies /4UWHAt and
// /4UWHAt/* to medomni.vercel.app/4UWHAt and /4UWHAt/* respectively.
// basePath ensures internal links (`/api/agent`, `/_next/...`, asset
// paths, the agent route) all carry the prefix so the rewrite catches
// them and Next.js wires them up correctly under proxy.
//
// No `redirects()` block: Next.js redirect matching is case-insensitive,
// so a `/4uwhat → /4UWHAt` rule fires on the canonical /4UWHAt URL too
// and produces an infinite 307 self-loop. Lowercase typo recovery lives
// exclusively at the v0-goat-note-landing-page-3c edge layer (where the
// canonical user traffic enters); medomni.vercel.app/4uwhat returns 404
// as a result, which is acceptable — the medomni-direct URL is a
// fallback, not a primary surface.
const nextConfig: NextConfig = {
  reactCompiler: true,
  basePath: "/4UWHAt",
};

export default nextConfig;
