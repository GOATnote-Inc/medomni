"use client";

import { useRef, useState } from "react";
import { AudioRecorder } from "@/components/AudioRecorder";
import { ImageUpload, processImage } from "@/components/ImageUpload";

const SAMPLE_PROMPT =
  "lactate of 25 after housefire and altered mental status — what am I missing?";

type ContentBlock =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "audio_url"; audio_url: { url: string } };

export function Composer() {
  const [prompt, setPrompt] = useState("");
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [audioDataUrl, setAudioDataUrl] = useState<string | null>(null);
  const [audioDurationMs, setAudioDurationMs] = useState<number>(0);
  const [streaming, setStreaming] = useState(false);
  const [reasoning, setReasoning] = useState("");
  const [content, setContent] = useState("");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function send() {
    if ((!prompt.trim() && !imageDataUrl && !audioDataUrl) || streaming) return;
    setStreaming(true);
    setReasoning("");
    setContent("");
    setErrorMsg(null);
    setDone(false);

    const ac = new AbortController();
    abortRef.current = ac;

    // Build OpenAI-compat multimodal content
    let messageContent: string | ContentBlock[];
    if (imageDataUrl || audioDataUrl) {
      const blocks: ContentBlock[] = [];
      if (audioDataUrl) {
        blocks.push({ type: "audio_url", audio_url: { url: audioDataUrl } });
      }
      if (imageDataUrl) {
        blocks.push({ type: "image_url", image_url: { url: imageDataUrl } });
      }
      let textPart = prompt.trim();
      if (!textPart) {
        if (audioDataUrl) textPart = "Transcribe the audio and answer the clinical question it contains. Cite guidelines + year for any recommendations.";
        else textPart = "Describe and analyze this clinical image.";
      }
      blocks.push({ type: "text", text: textPart });
      messageContent = blocks;
    } else {
      messageContent = prompt;
    }

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: messageContent }],
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
    setImageDataUrl(null);
    setImageName(null);
    setAudioDataUrl(null);
    setAudioDurationMs(0);
    setReasoning("");
    setContent("");
    setErrorMsg(null);
    setDone(false);
  }

  function clearImage() {
    setImageDataUrl(null);
    setImageName(null);
  }

  function clearAudio() {
    setAudioDataUrl(null);
    setAudioDurationMs(0);
  }

  function handleAudio(dataUrl: string, durationMs: number) {
    setAudioDataUrl(dataUrl);
    setAudioDurationMs(durationMs);
    setErrorMsg(null);
  }

  function handleAudioError(msg: string) {
    setErrorMsg(msg);
  }

  function handleImage(dataUrl: string, name: string) {
    setImageDataUrl(dataUrl);
    setImageName(name);
    setErrorMsg(null);
  }

  function handleImageError(msg: string) {
    setErrorMsg(msg);
  }

  async function onDrop(e: React.DragEvent<HTMLElement>) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setErrorMsg(`Drop an image file. Got: ${file.type || "unknown"}`);
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setErrorMsg(`Image too large (${Math.round(file.size / 1024 / 1024)} MB; max 12 MB).`);
      return;
    }
    try {
      const dataUrl = await processImage(file);
      handleImage(dataUrl, file.name);
    } catch (err) {
      setErrorMsg(`Image processing failed: ${(err as Error).message}`);
    }
  }

  const empty = !reasoning && !content && !errorMsg;

  return (
    <section className="flex flex-col gap-4">
      <div
        className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden"
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
      >
        {imageDataUrl && (
          <div className="flex items-center gap-3 px-5 pt-4 pb-2 border-b border-slate-100 bg-slate-50/40">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageDataUrl}
              alt={imageName || "uploaded"}
              className="w-16 h-16 object-cover rounded border border-slate-200"
            />
            <div className="flex-1 min-w-0">
              <p className="text-xs text-slate-700 truncate font-medium">
                {imageName || "image"}
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                EXIF stripped · resized · sent as multimodal input
              </p>
            </div>
            <button
              type="button"
              onClick={clearImage}
              className="text-xs text-slate-500 hover:text-slate-800 px-2 py-1 rounded"
            >
              Remove
            </button>
          </div>
        )}

        {audioDataUrl && (
          <div className="flex items-center gap-3 px-5 pt-4 pb-2 border-b border-slate-100 bg-slate-50/40">
            <div className="w-16 h-16 rounded border border-slate-200 bg-rose-50 flex items-center justify-center text-rose-600">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs text-slate-700 truncate font-medium">
                Recording · {(audioDurationMs / 1000).toFixed(1)}s
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                16 kHz mono PCM WAV · sent as native Omni audio (no third-party STT)
              </p>
            </div>
            <button
              type="button"
              onClick={clearAudio}
              className="text-xs text-slate-500 hover:text-slate-800 px-2 py-1 rounded"
            >
              Remove
            </button>
          </div>
        )}

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder={
            audioDataUrl
              ? "Audio attached. Add a follow-up question or just press Ask MedOmni."
              : imageDataUrl
              ? "Add context or ask a question about the image..."
              : "Ask a clinical question. Or tap the mic to record. Or attach an image."
          }
          rows={4}
          maxLength={4000}
          className="w-full resize-none bg-transparent px-5 py-4 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none"
        />
        <div className="flex items-center justify-between border-t border-slate-100 px-3 py-2 bg-slate-50/60">
          <div className="flex gap-1 items-center">
            <AudioRecorder
              onAudio={handleAudio}
              onError={handleAudioError}
              disabled={streaming || !!audioDataUrl}
            />
            <ImageUpload
              onImage={handleImage}
              onError={handleImageError}
              disabled={streaming}
            />
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
                disabled={(!prompt.trim() && !imageDataUrl && !audioDataUrl) || streaming}
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
            Type, dictate, or attach an image. Reasoning + answer stream below.
          </p>
        )}

        {errorMsg && (
          <div className="flex-1 text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-md p-3">
            <strong>Issue:</strong> {errorMsg}
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
