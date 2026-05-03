"use client";

import { useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import type { PubMedSearchResult } from "@/lib/tools/pubmed";

interface PubMedToolPart {
  type: "tool-pubmed_search";
  toolCallId: string;
  state: "input-streaming" | "input-available" | "output-available" | "output-error";
  input?: { query?: string; maxResults?: number };
  output?: PubMedSearchResult | { error: string };
}

export default function AgentPage() {
  const [input, setInput] = useState("");
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: "/api/agent" }),
  });

  const busy = status === "submitted" || status === "streaming";

  return (
    <main className="flex-1 w-full max-w-4xl mx-auto px-6 py-10 flex flex-col gap-6">
      <header className="flex flex-col gap-2">
        <p className="text-xs tracking-widest uppercase text-slate-500 font-medium">
          GOATnote · MedOmni · agent (day 1)
        </p>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-slate-900">
          Agent loop with PubMed search
        </h1>
        <p className="max-w-2xl text-sm text-slate-600 leading-relaxed">
          The model reasons, decides whether to consult PubMed, runs the search,
          then writes its answer with citations. Reasoning, tool calls, and the
          final answer stream live below. No PHI; public demo.
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
                  return (
                    <details key={key} className="text-sm text-slate-500">
                      <summary className="cursor-pointer text-xs uppercase tracking-wider text-slate-400 hover:text-slate-600">
                        reasoning
                      </summary>
                      <div className="whitespace-pre-wrap mt-2 italic border-l-2 border-slate-200 pl-3">
                        {part.text}
                      </div>
                    </details>
                  );
                }
                if (part.type === "tool-pubmed_search") {
                  const tp = part as PubMedToolPart;
                  return <PubMedToolCard key={key} part={tp} />;
                }
                return null;
              })}
            </div>
          </div>
        ))}
        {busy && (
          <div className="text-xs text-slate-500 italic">working...</div>
        )}
      </section>

      <form
        className="flex gap-2 sticky bottom-4"
        onSubmit={(e) => {
          e.preventDefault();
          const trimmed = input.trim();
          if (!trimmed || busy) return;
          sendMessage({ text: trimmed });
          setInput("");
        }}
      >
        <input
          className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:border-slate-500"
          placeholder="Ask a clinical question..."
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          disabled={busy}
        />
        <button
          type="submit"
          className="rounded-md bg-slate-900 text-white px-4 py-2 text-sm disabled:bg-slate-400"
          disabled={busy || input.trim().length === 0}
        >
          ask
        </button>
      </form>
    </main>
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
