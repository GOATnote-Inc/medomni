"use client";

import { useState } from "react";

const SAMPLE_PROMPT =
  "55 yo with 6mo of progressive exertional dyspnea, bilateral lower-extremity edema, JVP elevated. Workup so far: BNP 1850, EF 35% on echo. What's the next step?";

const DEMO_REPLY = `**Differential weighted toward HFrEF decompensation.** With NYHA II-III symptoms, EF 35%, and elevated BNP, the patient meets criteria for guideline-directed medical therapy (GDMT) initiation per AHA/ACC/HFSA 2022.

**Immediate priorities:**
1. **Volume status** — confirm euvolemia before adding negative-inotropic agents. Diuretic with loop agent if congested.
2. **GDMT four pillars** (initiate sequentially over 2-4 weeks if BP / renal allow):
   - ARNI (sacubitril/valsartan) preferred over ACEi
   - Beta-blocker (carvedilol, metoprolol succinate, or bisoprolol)
   - MRA (spironolactone or eplerenone)
   - SGLT2i (dapagliflozin or empagliflozin)
3. **Etiology workup** — ischemia evaluation if not done; consider cardiac MRI if non-ischemic suspected; iron studies (IV iron if deficient).

**Red flags to escalate:** rising creatinine >30%, K+ >5.5, symptomatic hypotension, persistent congestion despite oral diuretic.

**Citations:** AHA/ACC/HFSA 2022 HF Guideline · STRONG-HF 2022 (NEJM) · DAPA-HF 2019 · EMPEROR-Reduced 2020.

— *MedOmni Quick (sovereign tier, B300, NVFP4). For evaluation only — do not enter PHI.*`;

export function Composer() {
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamed, setStreamed] = useState("");
  const [done, setDone] = useState(false);

  async function send() {
    if (!prompt.trim() || streaming) return;
    setStreaming(true);
    setStreamed("");
    setDone(false);

    const text = DEMO_REPLY;
    const chunkSize = 4;
    for (let i = 0; i < text.length; i += chunkSize) {
      await new Promise((r) => setTimeout(r, 18));
      setStreamed(text.slice(0, i + chunkSize));
    }

    setStreaming(false);
    setDone(true);
  }

  function reset() {
    setPrompt("");
    setStreamed("");
    setDone(false);
  }

  return (
    <section className="flex flex-col gap-4">
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask a clinical question. e.g. 'Workup for new HFrEF with EF 35%?'"
          rows={4}
          maxLength={4000}
          className="w-full resize-none bg-transparent px-5 py-4 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none"
        />
        <div className="flex items-center justify-between border-t border-slate-100 px-3 py-2 bg-slate-50/60">
          <div className="flex gap-2">
            <button
              type="button"
              disabled
              title="Voice input ships in v0 day 4"
              aria-label="Voice (coming soon)"
              className="p-2 rounded-md text-slate-400 cursor-not-allowed"
            >
              {/* mic icon */}
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            </button>
            <button
              type="button"
              disabled
              title="Image upload ships in v0 day 3"
              aria-label="Upload (coming soon)"
              className="p-2 rounded-md text-slate-400 cursor-not-allowed"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
            </button>
            <button
              type="button"
              onClick={() => setPrompt(SAMPLE_PROMPT)}
              className="text-xs text-slate-500 hover:text-slate-800 px-2 py-1 rounded"
            >
              Try sample
            </button>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">{prompt.length}/4000</span>
            {done ? (
              <button
                type="button"
                onClick={reset}
                className="text-sm font-medium px-4 py-2 rounded-md bg-slate-200 text-slate-700 hover:bg-slate-300"
              >
                Ask again
              </button>
            ) : (
              <button
                type="button"
                onClick={send}
                disabled={!prompt.trim() || streaming}
                className="text-sm font-semibold px-4 py-2 rounded-md bg-[var(--color-accent)] text-white hover:bg-[var(--color-accent-hover)] disabled:bg-slate-300 disabled:cursor-not-allowed transition-colors"
              >
                {streaming ? "Streaming..." : "Ask MedOmni"}
              </button>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Quick stream panel */}
        <article className="rounded-xl border border-slate-200 bg-white p-5 min-h-[280px] flex flex-col">
          <header className="flex items-center justify-between mb-3 pb-3 border-b border-slate-100">
            <div>
              <h2 className="font-semibold text-slate-900 text-sm">MedOmni Quick</h2>
              <p className="text-xs text-slate-500">Sovereign · B300 · NVFP4</p>
            </div>
            <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
              live
            </span>
          </header>
          <div className="flex-1 text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
            {streamed || (
              <span className="text-slate-400 italic">
                Cited, sovereign answer streams here. No prompt or response is
                logged in demo mode.
              </span>
            )}
            {streaming && (
              <span className="inline-block w-1.5 h-4 bg-slate-700 ml-0.5 align-text-bottom animate-pulse" />
            )}
          </div>
        </article>

        {/* Peer stream panel — disabled */}
        <article className="rounded-xl border border-dashed border-slate-300 bg-slate-50/60 p-5 min-h-[280px] flex flex-col">
          <header className="flex items-center justify-between mb-3 pb-3 border-b border-slate-200">
            <div>
              <h2 className="font-semibold text-slate-500 text-sm">Frontier Peer</h2>
              <p className="text-xs text-slate-400">
                Anthropic Claude / OpenAI GPT · BAA-covered route
              </p>
            </div>
            <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded-full bg-slate-100 text-slate-500 border border-slate-200 font-medium">
              Pro · v1
            </span>
          </header>
          <div className="flex-1 flex flex-col items-center justify-center text-center gap-2 text-slate-500">
            <p className="text-sm font-medium">Pro tier — coming soon</p>
            <p className="text-xs leading-relaxed max-w-xs">
              Race-to-stream with a BAA-routed frontier model and reconciliation
              surface. Ships post-Nebius once Vertex Anthropic / OpenAI BAA
              umbrella is wired.
            </p>
          </div>
        </article>
      </div>
    </section>
  );
}
