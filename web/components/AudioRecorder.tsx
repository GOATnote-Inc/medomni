"use client";

// AudioRecorder.tsx — browser-side audio capture for native Nemotron-Omni.
// Path B (per MEDIARECORDER-AUDIO-SPEC.md, validated by REDTEAM-AUDIO.md):
//   getUserMedia → AudioContext + AudioWorklet → Float32 buffer
//   → OfflineAudioContext resample to 16 kHz
//   → 16-bit PCM WAV encode → base64 data URL
// No third-party STT. No webkitSpeechRecognition.
//
// Resilience patches from REDTEAM-AUDIO:
//  - silent GainNode(0) to kill iOS mic→speaker bleed
//  - RMS sanity check before send (silence rejected client-side)
//  - feature-gate AudioWorklet (graceful fallback message if undefined)
//  - track.onended + visibilitychange handlers (BT disconnect / background tab recovery)
//  - basePath-prefixed worklet path (next.config basePath: "/4UWHAt"
//    means /public assets must be requested at /4UWHAt/<file>)
//  - Blob + FileReader.readAsDataURL (no String.fromCharCode loop)
// Tap-to-toggle (NOT push-and-hold; that was the prior bug).

import { useEffect, useRef, useState } from "react";

const TARGET_SAMPLE_RATE = 16000; // Parakeet target
const MAX_DURATION_MS = 60000; // 60 sec → ~2.56 MB base64, under Vercel 4.5 MB cap
const MIN_DURATION_MS = 300; // reject sub-quarter-second blips
const MIN_RMS = 0.005; // reject ~silent recordings (mic muted, BT disconnect, etc.)

interface AudioRecorderProps {
  onAudio: (dataUrl: string, durationMs: number) => void;
  onError: (msg: string) => void;
  disabled?: boolean;
}

export function AudioRecorder({ onAudio, onError, disabled }: AudioRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [supported, setSupported] = useState<boolean | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const buffersRef = useRef<Float32Array[]>([]);
  const startedAtRef = useRef<number>(0);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputSampleRateRef = useRef<number>(48000);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const hasMedia = !!navigator.mediaDevices?.getUserMedia;
    const hasWorklet = typeof AudioWorkletNode !== "undefined";
    setSupported(hasMedia && hasWorklet);
  }, []);

  async function start() {
    if (recording || disabled) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // Recover from BT disconnect mid-record
      stream.getAudioTracks().forEach((t) => {
        t.onended = () => {
          if (recording) stop("Mic disconnected.");
        };
      });

      type WebkitWindow = Window & { webkitAudioContext?: typeof AudioContext };
      const w = window as unknown as WebkitWindow;
      const Ctor = window.AudioContext ?? w.webkitAudioContext;
      if (!Ctor) {
        throw new Error("AudioContext unsupported");
      }
      const ctx = new Ctor();
      ctxRef.current = ctx;
      inputSampleRateRef.current = ctx.sampleRate;

      // Public-asset paths must include the Next.js `basePath` prefix
      // (`/4UWHAt`, set in next.config.ts). Without it the URL resolves
      // to `https://www.thegoatnote.com/pcm-recorder-worklet.js`, which
      // falls outside the v0-goat-note-landing-page-3c project's
      // `/4UWHAt(/*) → medomni` rewrite and returns v0's static 404
      // page. AudioWorklet.addModule then fails with "Unable to load a
      // worklet's module." Verified 2026-05-04 against
      // www.thegoatnote.com/4UWHAt voice mode.
      await ctx.audioWorklet.addModule("/4UWHAt/pcm-recorder-worklet.js");

      const src = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, "pcm-recorder");
      // silent gain node to kill iOS mic→speaker bleed
      const silent = ctx.createGain();
      silent.gain.value = 0;
      buffersRef.current = [];
      node.port.onmessage = (e: MessageEvent<Float32Array>) => {
        buffersRef.current.push(e.data);
      };
      src.connect(node);
      node.connect(silent);
      silent.connect(ctx.destination);
      nodeRef.current = node;

      startedAtRef.current = Date.now();
      setElapsed(0);
      tickRef.current = setInterval(() => {
        const ms = Date.now() - startedAtRef.current;
        setElapsed(ms);
        if (ms >= MAX_DURATION_MS) stop();
      }, 200);

      setRecording(true);
    } catch (e) {
      const err = e as Error;
      onError(err.message || "Could not start recording");
      cleanup();
    }
  }

  async function stop(reason?: string) {
    if (!recording) return;
    setRecording(false);
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    const durationMs = Date.now() - startedAtRef.current;
    const buffers = buffersRef.current;
    const ctx = ctxRef.current;
    cleanup();

    if (reason) {
      onError(reason);
      return;
    }
    if (durationMs < MIN_DURATION_MS) {
      onError("Recording too short (release after at least 0.5s).");
      return;
    }

    const total = buffers.reduce((acc, b) => acc + b.length, 0);
    if (total === 0) {
      onError("No audio captured (mic may be muted or background tab).");
      return;
    }

    const merged = new Float32Array(total);
    let off = 0;
    for (const b of buffers) {
      merged.set(b, off);
      off += b.length;
    }

    let sumSq = 0;
    for (let i = 0; i < merged.length; i++) sumSq += merged[i] * merged[i];
    const rms = Math.sqrt(sumSq / merged.length);
    if (rms < MIN_RMS) {
      onError("Audio too quiet (RMS below threshold). Try again, closer to the mic.");
      return;
    }

    try {
      const inputRate = ctx?.sampleRate ?? inputSampleRateRef.current;
      const resampled = await resample(merged, inputRate, TARGET_SAMPLE_RATE);
      const wavBlob = encodeWav(resampled, TARGET_SAMPLE_RATE);
      const dataUrl = await blobToDataUrl(wavBlob);
      onAudio(dataUrl, durationMs);
    } catch (e) {
      onError(`Encoding failed: ${(e as Error).message}`);
    }
  }

  function cleanup() {
    nodeRef.current?.disconnect();
    nodeRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
  }

  useEffect(() => {
    function onVisibility() {
      if (document.hidden && recording) stop("Recording paused (tab backgrounded).");
    }
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [recording]);

  if (supported === false) {
    return (
      <button
        type="button"
        disabled
        title="Audio recording unavailable in this browser. Try Chrome or Safari."
        aria-label="Audio unsupported"
        className="p-2 rounded-md text-slate-300 cursor-not-allowed"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      </button>
    );
  }

  const sec = (elapsed / 1000).toFixed(1);

  return (
    <button
      type="button"
      onClick={() => (recording ? stop() : start())}
      disabled={disabled}
      title={recording ? "Tap to stop + send" : "Tap to record"}
      aria-label={recording ? "Stop recording" : "Start recording"}
      className={
        recording
          ? "px-3 py-2 rounded-md bg-rose-500 text-white text-xs font-medium animate-pulse flex items-center gap-1.5"
          : "p-2 rounded-md text-slate-500 hover:bg-slate-200 hover:text-slate-800 transition-colors"
      }
    >
      {recording ? (
        <>
          <span className="w-2 h-2 rounded-full bg-white" />
          <span>REC {sec}s</span>
        </>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      )}
    </button>
  );
}

async function resample(input: Float32Array, fromRate: number, toRate: number): Promise<Float32Array> {
  if (fromRate === toRate) return input;
  const length = Math.ceil((input.length * toRate) / fromRate);
  const ctx = new OfflineAudioContext(1, length, toRate);
  const buf = ctx.createBuffer(1, input.length, fromRate);
  buf.getChannelData(0).set(input);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  // 16-bit PCM, mono
  const numChannels = 1;
  const bytesPerSample = 2;
  const byteRate = sampleRate * numChannels * bytesPerSample;
  const blockAlign = numChannels * bytesPerSample;
  const dataLen = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buffer);

  function ws(off: number, s: string) {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  }
  ws(0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  ws(36, "data");
  view.setUint32(40, dataLen, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error || new Error("FileReader error"));
    reader.readAsDataURL(blob);
  });
}
