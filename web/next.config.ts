import type { NextConfig } from "next";

// Mounted at /4UWHAt because the canonical demo URL is
// https://www.thegoatnote.com/4UWHAt — v0-goat-note-landing-page-3c (which
// owns www.thegoatnote.com on Vercel) reverse-proxies /4UWHAt and
// /4UWHAt/* to medomni.vercel.app/4UWHAt and /4UWHAt/* respectively.
// basePath ensures internal links (`/api/agent`, `/_next/...`, asset
// paths, the agent route) all carry the prefix so the rewrite catches
// them and Next.js wires them up correctly under proxy.
//
// medomni.vercel.app/ (no path) returns 404 — that is acceptable for
// a basePath app. A previous `{source: "/", destination: "/4UWHAt"}`
// redirect caused a loop with the basePath rewrite (medomni.vercel.app/4UWHAt
// → 307 → /4UWHAt?cb=…). Removed.
const nextConfig: NextConfig = {
  reactCompiler: true,
  basePath: "/4UWHAt",
  async redirects() {
    return [
      // Lowercase typo recovery (also done at the v0 landing edge)
      { source: "/4uwhat", destination: "/4UWHAt", permanent: false, basePath: false },
      { source: "/4uwhat/:path*", destination: "/4UWHAt/:path*", permanent: false, basePath: false },
    ];
  },
};

export default nextConfig;
