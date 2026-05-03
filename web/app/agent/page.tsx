"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { AudioRecorder } from "@/components/AudioRecorder";
import type { PubMedSearchResult } from "@/lib/tools/pubmed";
import type { PrimeKGSubgraphResult } from "@/lib/tools/primekg";

interface PubMedToolPart {
  type: "tool-pubmed_search";
  toolCallId: string;
  state: "input-streaming" | "input-available" | "output-available" | "output-error";
  input?: { query?: string; maxResults?: number };
  output?: PubMedSearchResult | { error: string };
}

interface PrimeKGToolPart {
  type: "tool-primekg_lookup";
  toolCallId: string;
  state: "input-streaming" | "input-available" | "output-available" | "output-error";
  input?: { query?: string; maxHops?: number; maxNodes?: number };
  output?: PrimeKGSubgraphResult | { error: string };
}

interface FilePart {
  type: "file";
  mediaType: string;
  url: string;
  filename?: string;
}

type Mode = "text" | "voice";

export default function AgentPage() {
  const [mode, setMode] = useState<Mode>("text");
  const [input, setInput] = useState("");
  const [pendingAudio, setPendingAudio] = useState<{ url: string; durationMs: number } | null>(null);
  const [audioError, setAudioError] = useState<string | null>(null);

  const { messages, sendMessage, status, stop } = useChat({
    transport: new DefaultChatTransport({ api: "/api/agent" }),
  });

  const busy = status === "submitted" || status === "streaming";
  const hasAudio = pendingAudio !== null;

  function handleAudio(url: string, durationMs: number) {
    setPendingAudio({ url, durationMs });
    setAudioError(null);
  }

  function clearAudio() {
    setPendingAudio(null);
  }

  async function submit() {
    if (busy) return;
    const trimmed = input.trim();
    if (mode === "voice" && pendingAudio) {
      const followup = trimmed ||
        "Transcribe the audio and answer the clinical question it contains. Cite guidelines and PMIDs where they meaningfully change the answer.";
      await sendMessage({
        text: followup,
        files: [
          {
            type: "file",
            mediaType: "audio/wav",
            url: pendingAudio.url,
            filename: `recording-${pendingAudio.durationMs}ms.wav`,
          } as FilePart,
        ],
      });
      setInput("");
      setPendingAudio(null);
      return;
    }
    if (mode === "text" && trimmed) {
      await sendMessage({ text: trimmed });
      setInput("");
    }
  }

  return (
    <main className="flex-1 w-full max-w-4xl mx-auto px-6 py-10 flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <p className="text-xs tracking-widest uppercase text-slate-500 font-medium">
            GOATnote · MedOmni · agent
          </p>
          <ModeToggle mode={mode} onChange={setMode} disabled={busy} />
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
          Agent loop with PubMed search
        </h1>
        <p className="max-w-2xl text-sm text-slate-600 leading-relaxed">
          Ask in text or voice. The model reasons, decides whether to consult
          PubMed, runs the search, then writes its answer with citations.
          Audio in, text out — single forward pass on Nemotron-3-Nano-Omni
          (no separate ASR model). Public demo, no PHI.
        </p>
      </header>

      <section className="flex flex-col gap-4">
        {messages.map((m) => (
          <div key={m.id} className="border border-slate-200 rounded-md p-4 bg-white">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
              {m.role === "user" ? "you" : "medomni"}
            </div>
            <div className="flex flex-col gap-3">
              {m.parts.map((part, i) => {
                const key = `${m.id}-${i}`;
                if (part.type === "text") {
                  return (
                    <div key={key} className="whitespace-pre-wrap text-slate-900 leading-relaxed">
                      {part.text}
                    </div>
                  );
                }
                if (part.type === "reasoning") {
                  // For voice messages the model is instructed to start with
                  // "Transcript: ..." — this lets the user verify the ASR
                  // before reading the answer. Auto-expanded while streaming.
                  const looksLikeTranscript = part.text.trimStart().toLowerCase().startsWith("transcript:");
                  return (
                    <details
                      key={key}
                      className="text-sm text-slate-500"
                      open={looksLikeTranscript || part.state === "streaming"}
                    >
                      <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400 hover:text-slate-600">
                        {looksLikeTranscript ? "transcript + reasoning" : "reasoning"}
                      </summary>
                      <div className="whitespace-pre-wrap mt-2 italic border-l-2 border-slate-200 pl-3">
                        {part.text}
                      </div>
                    </details>
                  );
                }
                if (part.type === "file" && (part as FilePart).mediaType?.startsWith("audio/")) {
                  return <AudioMessageChip key={key} url={(part as FilePart).url} />;
                }
                if (part.type === "tool-pubmed_search") {
                  return <PubMedToolCard key={key} part={part as PubMedToolPart} />;
                }
                if (part.type === "tool-primekg_lookup") {
                  return <PrimeKGToolCard key={key} part={part as PrimeKGToolPart} />;
                }
                return null;
              })}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-slate-500 italic">
            <span>working...</span>
            <button
              type="button"
              onClick={() => stop()}
              className="text-slate-400 hover:text-slate-600 underline"
            >
              stop
            </button>
          </div>
        )}
      </section>

      <form
        className="flex flex-col gap-2 sticky bottom-4"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        {audioError && (
          <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-md px-3 py-2">
            {audioError}
          </div>
        )}
        {pendingAudio && (
          <div className="flex items-center gap-3 px-3 py-2 rounded-md border border-slate-300 bg-slate-50 text-xs">
            <div className="flex items-center gap-2 text-slate-700">
              <MicGlyph />
              <span>Recording attached · {(pendingAudio.durationMs / 1000).toFixed(1)}s · 16 kHz mono PCM</span>
            </div>
            <button
              type="button"
              onClick={clearAudio}
              className="ml-auto text-slate-500 hover:text-slate-800"
            >
              clear
            </button>
          </div>
        )}
        <div className="flex gap-2 items-center">
          {mode === "voice" && (
            <AudioRecorder
              onAudio={handleAudio}
              onError={setAudioError}
              disabled={busy || hasAudio}
            />
          )}
          <input
            className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:border-slate-500"
            placeholder={
              mode === "voice"
                ? hasAudio
                  ? "Optional follow-up text + send..."
                  : "Tap the mic to record. Optional follow-up text below."
                : "Ask a clinical question..."
            }
            value={input}
            onChange={(e) => setInput(e.currentTarget.value)}
            disabled={busy}
          />
          <button
            type="submit"
            className="rounded-md bg-slate-900 text-white px-4 py-2 text-sm disabled:bg-slate-400"
            disabled={
              busy ||
              (mode === "text" && input.trim().length === 0) ||
              (mode === "voice" && !hasAudio && input.trim().length === 0)
            }
          >
            ask
          </button>
        </div>
      </form>
    </main>
  );
}

function ModeToggle({
  mode,
  onChange,
  disabled,
}: {
  mode: Mode;
  onChange: (m: Mode) => void;
  disabled: boolean;
}) {
  return (
    <div
      role="tablist"
      aria-label="input mode"
      className="inline-flex border border-slate-300 rounded-md text-xs overflow-hidden"
    >
      {(["text", "voice"] as const).map((m) => {
        const active = m === mode;
        return (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={active}
            disabled={disabled}
            onClick={() => onChange(m)}
            className={
              active
                ? "px-3 py-1.5 bg-slate-900 text-white font-medium"
                : "px-3 py-1.5 bg-white text-slate-600 hover:bg-slate-100 disabled:text-slate-400"
            }
          >
            {m}
          </button>
        );
      })}
    </div>
  );
}

function AudioMessageChip({ url }: { url: string }) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs">
      <MicGlyph />
      <span className="text-slate-700">voice input</span>
      <audio src={url} controls className="h-7" />
    </div>
  );
}

function MicGlyph() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-rose-600"
    >
      <rect x="9" y="2" width="6" height="13" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
    </svg>
  );
}

function PrimeKGToolCard({ part }: { part: PrimeKGToolPart }) {
  const query = part.input?.query;
  const output = part.output;

  return (
    <div className="border border-emerald-300 rounded-md bg-emerald-50/60 px-3 py-2 text-sm">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider">
        <span className="text-emerald-800 font-medium">tool · primekg_lookup</span>
        <span className="text-emerald-400">·</span>
        <span className="text-emerald-600">{part.state.replace("-", " ")}</span>
      </div>
      {query && (
        <div className="mt-1 text-slate-700">
          <span className="text-slate-400">entity:</span> {query}
          {part.input?.maxHops ? (
            <span className="text-slate-400"> · {part.input.maxHops}-hop</span>
          ) : null}
        </div>
      )}
      {output && "error" in output && (
        <div className="mt-2 text-rose-700 text-xs">error: {output.error}</div>
      )}
      {output && "n_nodes" in output && (
        <div className="mt-2 flex flex-col gap-2">
          <div className="text-xs text-slate-600">
            {output.seed_count} seed{output.seed_count === 1 ? "" : "s"}
            {output.seed_names.length > 0 && (
              <>
                {" "}
                ({output.seed_names.slice(0, 4).join(", ")}
                {output.seed_names.length > 4 ? ", …" : ""})
              </>
            )}
            {" · "}
            {output.n_nodes} node{output.n_nodes === 1 ? "" : "s"} ·{" "}
            {output.n_edges} edge{output.n_edges === 1 ? "" : "s"} ·{" "}
            {output.elapsed_ms} ms
          </div>
          {output.block && (
            <details className="text-xs">
              <summary className="cursor-pointer text-emerald-700 hover:text-emerald-900">
                subgraph block
              </summary>
              <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] leading-snug text-slate-700 bg-white/60 border border-emerald-200 rounded p-2 max-h-72 overflow-auto">
                {output.block}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function PubMedToolCard({ part }: { part: PubMedToolPart }) {
  const query = part.input?.query;
  const output = part.output;

  return (
    <div className="border border-slate-300 rounded-md bg-slate-50 px-3 py-2 text-sm">
      <div className="flex items-center gap-2 text-xs uppercase tracking-wider">
        <span className="text-slate-700 font-medium">tool · pubmed_search</span>
        <span className="text-slate-400">·</span>
        <span className="text-slate-500">{part.state.replace("-", " ")}</span>
      </div>
      {query && (
        <div className="mt-1 text-slate-600">
          <span className="text-slate-400">query:</span> {query}
        </div>
      )}
      {output && "error" in output && (
        <div className="mt-2 text-rose-700 text-xs">error: {output.error}</div>
      )}
      {output && "records" in output && (
        <div className="mt-2 flex flex-col gap-2">
          <div className="text-xs text-slate-500">
            {output.records.length} of {output.count} matches
          </div>
          {output.records.map((r) => (
            <div key={r.pmid} className="text-xs leading-relaxed">
              <a
                href={r.url}
                target="_blank"
                rel="noreferrer"
                className="text-slate-800 underline decoration-slate-400 underline-offset-2 hover:text-slate-900"
              >
                PMID {r.pmid}
              </a>{" "}
              · <span className="text-slate-700">{r.title}</span>
              <div className="text-slate-500">
                {r.authors}
                {r.authors && r.journal ? " · " : ""}
                {r.journal}
                {r.year ? ` (${r.year})` : ""}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
