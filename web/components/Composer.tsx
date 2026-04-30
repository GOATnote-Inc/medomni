"use client";

import { useState, useRef } from "react";

const SAMPLE_PROMPT =
  "lactate of 25 after housefire and altered mental status — what am I missing?";

export function Composer() {
  const [prompt, setPrompt] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [reasoning, setReasoning] = useState("");
  const [content, setContent] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    if (!prompt.trim() || streaming) return;
    setStreaming(true);
    setReasoning("");
    setContent("");
    setErrorMsg(null);
    setDone(false);

    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: prompt }],
        }),
        signal: ac.signal,
      });

      if (!res.ok) {
        let msg = `Error ${res.status}`;
        try {
          const j = await res.json();
          msg = j.error || msg;
        } catch {
          /* ignore */
        }
        setErrorMsg(msg);
        setStreaming(false);
        setDone(true);
        return;
      }

      if (!res.body) {
        setErrorMsg("Empty response from server");
        setStreaming(false);
        setDone(true);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done: rdone, value } = await reader.read();
        if (rdone) break;
        buffer += decoder.decode(value, { stream: true });

        // Parse SSE — split on double-newline event boundary
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const evt of events) {
          for (const line of evt.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") continue;
            try {
              const json = JSON.parse(data);
              const delta = json.choices?.[0]?.delta;
              if (!delta) continue;
              if (typeof delta.reasoning === "string" && delta.reasoning.length > 0) {
                setReasoning((prev) => prev + delta.reasoning);
              }
              if (typeof delta.content === "string" && delta.content.length > 0) {
                setContent((prev) => prev + delta.content);
              }
            } catch {
              /* skip un-parseable line */
            }
          }
        }
      }
    } catch (e) {
      const err = e as Error;
      if (err.name !== "AbortError") {
        setErrorMsg(err.message || "Connection failed");
      }
    } finally {
      setStreaming(false);
      setDone(true);
      abortRef.current = null;
    }
  }

  function reset() {
    abortRef.current?.abort();
    setPrompt("");
    setReasoning("");
    setContent("");
    setErrorMsg(null);
    setDone(false);
  }

  const empty = !reasoning && !content && !errorMsg;

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
              title="Voice input ships next"
              aria-label="Voice (coming soon)"
              className="p-2 rounded-md text-slate-400 cursor-not-allowed"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            </button>
            <button
              type="button"
              disabled
              title="Image upload ships next"
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
          <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
            live
          </span>
        </header>

        {empty && (
          <p className="flex-1 text-sm text-slate-400 italic">
            Type a clinical question and press &quot;Ask MedOmni&quot;. The
            model streams its reasoning, then its answer.
          </p>
        )}

        {errorMsg && (
          <div className="flex-1 text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-md p-3">
            <strong>Connection issue:</strong> {errorMsg}
          </div>
        )}

        {reasoning && (
          <details className="mb-3 text-xs text-slate-500 bg-slate-50 rounded-md p-3 border border-slate-200" open={streaming && !content}>
            <summary className="cursor-pointer font-medium text-slate-600">
              Reasoning trace {streaming && !content ? "(streaming)" : ""}
            </summary>
            <div className="mt-2 whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
              {reasoning}
              {streaming && !content && (
                <span className="inline-block w-1.5 h-3 bg-slate-500 ml-0.5 align-text-bottom animate-pulse" />
              )}
            </div>
          </details>
        )}

        {content && (
          <div className="flex-1 text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">
            {content}
            {streaming && (
              <span className="inline-block w-1.5 h-4 bg-slate-700 ml-0.5 align-text-bottom animate-pulse" />
            )}
          </div>
        )}
      </article>
    </section>
  );
}
