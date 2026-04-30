"use client";

import { useState } from "react";

const SAMPLE_PROMPT =
  "lactate of 25 after housefire and altered mental status — what am I missing?";

const NOT_LIVE_NOTICE = `The model endpoint is being plugged in right now. This page does not fake an answer in the meantime — when Nemotron-3-Nano-Omni on the B300 starts answering here, it'll be real.

Refresh in a bit. Source: [github.com/GOATnote-Inc/medomni](https://github.com/GOATnote-Inc/medomni).`;

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

    const text = NOT_LIVE_NOTICE;
    const chunkSize = 4;
    for (let i = 0; i < text.length; i += chunkSize) {
      await new Promise((r) => setTimeout(r, 12));
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

      <article className="rounded-xl border border-slate-200 bg-white p-5 min-h-[300px] flex flex-col">
        <header className="flex items-center justify-between mb-3 pb-3 border-b border-slate-100">
          <div>
            <h2 className="font-semibold text-slate-900 text-sm">MedOmni</h2>
            <p className="text-xs text-slate-500">
              Nemotron-3-Nano-Omni · NVIDIA Blackwell B300 · open weights
            </p>
          </div>
          <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-medium">
            wiring up
          </span>
        </header>
        <div className="flex-1 text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
          {streamed || (
            <span className="text-slate-400 italic">
              Type a clinical question and press &quot;Ask MedOmni&quot;. The
              answer streams here.
            </span>
          )}
          {streaming && (
            <span className="inline-block w-1.5 h-4 bg-slate-700 ml-0.5 align-text-bottom animate-pulse" />
          )}
        </div>
      </article>
    </section>
  );
}
