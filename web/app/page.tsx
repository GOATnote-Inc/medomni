// /4UWHAt landing — Records OS dashboard.
//
// Replaces the prior agent-page re-export. The Records OS surface owns
// the header, three-column dashboard, and the embedded "Ask your record"
// command bar (which is the agent's `useChat` driver in a tighter
// dashboard frame). The bare full-page agent UI still ships at
// /4UWHAt/agent for direct-access fallback (PR #34/#35 contract:
// `web/app/agent/page.tsx` is unchanged here).
//
// SessionProvider wraps the tree so usePatientId/usePersona resolve from
// localStorage across navigations. DemoBanner stays per the public-demo
// PHI contract.

import { DemoBanner } from "@/components/DemoBanner";
import { SessionProvider } from "@/components/4uwhat/SessionProvider";
import { RecordsOS } from "./records/RecordsOS";

export default function Home() {
  return (
    <SessionProvider>
      <DemoBanner />
      <RecordsOS />
    </SessionProvider>
  );
}
