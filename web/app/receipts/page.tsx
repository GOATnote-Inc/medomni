"use client";

// /4UWHAt/receipts — per-turn audit trail surface.
//
// Renders every assistant turn the user has had with the medomni agent
// since they last cleared the receipts log. Data lives entirely client-
// side in localStorage (key `medomni:receipts:v1`); no server-side
// telemetry is added. The agent route at /api/agent is unchanged.
//
// Wrapped in SessionProvider + DemoBanner to match the rest of the
// 4UWHAt surface; the public-demo PHI contract still applies here.

import { useCallback, useEffect, useState, type CSSProperties } from "react";
import { DemoBanner } from "@/components/DemoBanner";
import { SessionProvider } from "@/components/4uwhat/SessionProvider";
import { ReceiptCard } from "@/components/4uwhat/ReceiptCard";
import { Eyebrow } from "@/components/4uwhat/Eyebrow";
import { Mono } from "@/components/4uwhat/Mono";
import { Wordmark } from "@/components/4uwhat/Wordmark";
import {
  clearReceipts,
  exportReceiptsAsMarkdown,
  loadReceipts,
  type Receipt,
} from "@/lib/4uwhat/receipts";
import { BASE_PATH } from "@/lib/basePath";

const pageShell: CSSProperties = {
  background: "#000",
  color: "#fff",
  minHeight: "calc(100vh - 40px)",
  fontFamily: "var(--font-display)",
  borderTop: "1px solid #1f1f1f",
};

const contentWrap: CSSProperties = {
  maxWidth: 960,
  margin: "0 auto",
  padding: "32px 28px 56px",
  display: "flex",
  flexDirection: "column",
  gap: 24,
};

const btnGhost: CSSProperties = {
  background: "transparent",
  color: "rgba(255,255,255,0.8)",
  border: "1px solid rgba(255,255,255,0.18)",
  padding: "7px 12px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 11.5,
  letterSpacing: "0.04em",
  cursor: "pointer",
};

const btnPrimary: CSSProperties = {
  background: "var(--accent)",
  color: "#fff",
  border: "1px solid var(--accent)",
  padding: "7px 14px",
  fontFamily: "var(--font-display)",
  fontWeight: 600,
  fontSize: 11.5,
  letterSpacing: "0.04em",
  cursor: "pointer",
};

function ReceiptsView() {
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  // Hydrate from localStorage on mount. We deliberately avoid reading
  // during render so SSR + first paint match.
  useEffect(() => {
    setReceipts(loadReceipts());
    setHydrated(true);
  }, []);

  // Auto-dismiss toast after 2.5s.
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2500);
    return () => clearTimeout(t);
  }, [toast]);

  const handleExport = useCallback(async () => {
    const md = exportReceiptsAsMarkdown(receipts);
    try {
      if (
        typeof navigator !== "undefined" &&
        navigator.clipboard?.writeText
      ) {
        await navigator.clipboard.writeText(md);
        setToast("Copied markdown to clipboard");
        return;
      }
      // Fallback: drop a textarea, select, execCommand("copy"). Works
      // on older browsers + private modes that gate clipboard API.
      if (typeof document !== "undefined") {
        const ta = document.createElement("textarea");
        ta.value = md;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
          setToast("Copied markdown to clipboard");
        } finally {
          document.body.removeChild(ta);
        }
      }
    } catch {
      setToast("Copy failed — try again");
    }
  }, [receipts]);

  const handleClear = useCallback(() => {
    if (typeof window === "undefined") return;
    const ok = window.confirm(
      `Clear all ${receipts.length} receipt${receipts.length === 1 ? "" : "s"}? This cannot be undone.`,
    );
    if (!ok) return;
    clearReceipts();
    setReceipts([]);
    setToast("Cleared");
  }, [receipts.length]);

  // Newest first.
  const ordered = [...receipts].sort((a, b) => b.timestamp - a.timestamp);

  return (
    <div style={pageShell}>
      {/* Top bar — wordmark + back link, matches RecordsOS top-bar density */}
      <div
        style={{
          height: 56,
          borderBottom: "1px solid #1f1f1f",
          padding: "0 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <Wordmark size={13} />
          <span style={{ color: "rgba(255,255,255,0.2)" }}>/</span>
          <Eyebrow>RECEIPTS</Eyebrow>
        </div>
        <a
          href={`${BASE_PATH}/`}
          style={{
            ...btnGhost,
            textDecoration: "none",
            display: "inline-flex",
            alignItems: "center",
          }}
        >
          ← Back to record
        </a>
      </div>

      <div style={contentWrap}>
        <header
          style={{ display: "flex", flexDirection: "column", gap: 10 }}
        >
          <h1
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 36,
              fontWeight: 700,
              letterSpacing: "-0.025em",
              lineHeight: 1.05,
              margin: 0,
            }}
          >
            Audit receipts
          </h1>
          <p
            style={{
              margin: 0,
              fontSize: 14,
              lineHeight: 1.55,
              color: "rgba(255,255,255,0.7)",
              maxWidth: 720,
            }}
          >
            Per-turn log of every interaction with the medomni clinical
            agent. Each receipt records the prompt, tools called, response,
            and verification metadata. Stored in your browser only — never
            sent to a third-party.
          </p>
        </header>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <Mono size={11} color="rgba(255,255,255,0.6)">
            {hydrated
              ? `${ordered.length} RECEIPT${ordered.length === 1 ? "" : "S"}`
              : "LOADING…"}
          </Mono>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {toast && (
              <Mono size={10} color="var(--accent)">
                {toast.toUpperCase()}
              </Mono>
            )}
            <button
              type="button"
              onClick={handleExport}
              disabled={!hydrated || ordered.length === 0}
              style={{
                ...btnPrimary,
                opacity: !hydrated || ordered.length === 0 ? 0.5 : 1,
                cursor:
                  !hydrated || ordered.length === 0 ? "not-allowed" : "pointer",
              }}
              aria-label="Copy all receipts as markdown to clipboard"
            >
              Export markdown
            </button>
            <button
              type="button"
              onClick={handleClear}
              disabled={!hydrated || ordered.length === 0}
              style={{
                ...btnGhost,
                opacity: !hydrated || ordered.length === 0 ? 0.5 : 1,
                cursor:
                  !hydrated || ordered.length === 0 ? "not-allowed" : "pointer",
              }}
              aria-label="Clear all receipts"
            >
              Clear all
            </button>
          </div>
        </div>

        {hydrated && ordered.length === 0 && (
          <div
            style={{
              border: "1px dashed rgba(255,0,150,0.25)",
              padding: "32px 24px",
              textAlign: "center",
              color: "rgba(255,255,255,0.65)",
              fontSize: 14,
              lineHeight: 1.6,
              background: "rgba(255,0,150,0.02)",
            }}
          >
            <div style={{ marginBottom: 8 }}>
              <Eyebrow>NO RECEIPTS YET</Eyebrow>
            </div>
            Interact with Maya Okafor&apos;s record to start logging. Every
            assistant turn from the &quot;Ask your record&quot; bar lands
            here automatically.
          </div>
        )}

        {hydrated && ordered.length > 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            {ordered.map((r) => (
              <ReceiptCard key={r.id} receipt={r} />
            ))}
          </div>
        )}

        <footer
          style={{
            marginTop: 16,
            paddingTop: 16,
            borderTop: "1px solid rgba(255,255,255,0.06)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            lineHeight: 1.55,
            color: "rgba(255,255,255,0.5)",
          }}
        >
          Receipts persist in localStorage; clear at any time via the
          button above. This is a public demo — never enter PHI.
        </footer>
      </div>
    </div>
  );
}

export default function ReceiptsPage() {
  return (
    <SessionProvider>
      <DemoBanner />
      <ReceiptsView />
    </SessionProvider>
  );
}
