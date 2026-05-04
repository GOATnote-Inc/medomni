import type { NextConfig } from "next";

// Mounted at /4UWHAt because the canonical demo URL is
// https://www.thegoatnote.com/4UWHAt — prism42-console (which owns
// www.thegoatnote.com on Vercel) reverse-proxies /4UWHAt and
// /4UWHAt/* to medomni.vercel.app/4UWHAt and /4UWHAt/* respectively.
// basePath ensures internal links (`/api/agent`, `/_next/...`, asset
// paths, the agent route) all carry the prefix so the rewrite catches
// them and Next.js wires them up correctly under proxy.
//
// The direct medomni.vercel.app URL still works: it just lives at
// medomni.vercel.app/4UWHAt now (root `/` redirects via the redirects
// block below so old bookmarks don't 404).
const nextConfig: NextConfig = {
  reactCompiler: true,
  basePath: "/4UWHAt",
  async redirects() {
    return [
      // Direct medomni.vercel.app/ users land at the basePath instead
      // of seeing a 404. Permanent so bookmarks update.
      { source: "/", destination: "/4UWHAt", permanent: true, basePath: false },
      // Lowercase typo recovery (also done at the prism42-console edge)
      { source: "/4uwhat", destination: "/4UWHAt", permanent: false, basePath: false },
      { source: "/4uwhat/:path*", destination: "/4UWHAt/:path*", permanent: false, basePath: false },
    ];
  },
};

export default nextConfig;
