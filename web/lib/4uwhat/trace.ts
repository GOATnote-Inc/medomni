// 4UWHAt — inference-trace data model.
//
// Phase 1 of the flagship-demo plan ("make the stack visible"): the server
// agent loop measures every real stage it runs and streams a structured
// trace alongside the existing reasoning/text/tool UI-message stream. This
// file is the single source of truth for that trace's shape — the
// `/api/agent` route emits it, `TraceTimeline` renders it, and Phase 2
// extends it.
//
// Transport contract — the trace travels as AI-SDK custom data parts:
//   - one `data-stage` part PER stage, written with a stable `id` so a
//     stage that is later updated (e.g. an inference step whose duration
//     is only known once the step closes) reconciles in place rather than
//     duplicating. The id convention is `stage-${index}`.
//   - one terminal `data-turn-metrics` part with the turn summary.
// Both arrive in `message.parts` as `{ type: 'data-stage', data: TraceStage }`
// and `{ type: 'data-turn-metrics', data: TurnMetrics }` once useChat has
// reconciled the stream.
//
// Honesty rule (SPEC §6 Phase 1 / §8): every stage here is a stage that
// REALLY ran. No placeholder stages. Phase 2 adds 'guardrail' and
// 'retrieval' kinds when those stages actually execute — which is exactly
// why `TraceStageKind` is an open string union, not a closed enum.

/**
 * The kind of pipeline stage. Drives the kind-colored marker in the
 * timeline UI.
 *
 * EXTENSIBILITY CONTRACT (load-bearing): Phase 2 of the demo plan wires a
 * real NeMo Guardrails rail and a cuVS/nx-cugraph retrieval stage. When it
 * does, it adds `'guardrail'` and `'retrieval'` to this union and a color
 * for each in `TRACE_KIND_COLORS`. Keep this union open for that — do not
 * collapse it to the two Phase-1 members. Consumers that switch on `kind`
 * must keep a default branch so an unknown future kind degrades gracefully
 * instead of throwing.
 */
export type TraceStageKind =
  | "inference" // a model generation step (Nemotron-3-Nano-Omni on vLLM)
  | "tool" // a tool call (pubmed_search, clinical_calculate, MCP rule, ...)
  // -- Phase 2 will add: --
  | "guardrail" // a NeMo Guardrails input/output rail
  | "retrieval"; // a cuVS / nx-cugraph hybrid recall stage

/**
 * One measured stage of a single answer's pipeline.
 *
 * All times are milliseconds. `startedAtMs` is relative to the start of the
 * turn (the moment the `/api/agent` POST began its agent loop), NOT a wall
 * clock — this keeps the trace stable to render and free of absolute
 * timestamps that would vary per request.
 */
export interface TraceStage {
  /** Stable per-turn id. Convention: `stage-${index}` (0-based). */
  id: string;
  /**
   * Human-readable label, named honestly for what actually ran — e.g.
   * "Nemotron-3-Nano-Omni inference" or "tool: pubmed_search". Never a
   * generic or invented label.
   */
  label: string;
  /** Stage kind — drives the marker color. */
  kind: TraceStageKind;
  /** Offset from turn start, in ms, when this stage began. */
  startedAtMs: number;
  /** Measured wall time of the stage, in ms. */
  durationMs: number;
  /** Optional one-line detail (tool query, step index, token count, ...). */
  detail?: string;
}

/**
 * The whole-turn summary, emitted once when the turn completes.
 */
export interface TurnMetrics {
  /** Time-to-first-token: request start -> first streamed model token, ms. */
  ttftMs: number;
  /** Total turn wall time, ms. */
  totalMs: number;
  /** Output token count for the turn (assistant text + reasoning). */
  tokens: number;
  /** Sustained output throughput, tokens per second. */
  tokPerSec: number;
}

/**
 * The custom-data-part name -> payload-type map. Pass this to `useChat`
 * as its `UIMessage` data-type parameter so `message.parts` is typed:
 *
 *   useChat<UIMessage<unknown, TraceDataParts>>(...)
 *
 * and so the server's `writer.write` calls are checked against it.
 */
export interface TraceDataParts {
  /** A single stage. Streamed incrementally, one per stage, keyed by id. */
  stage: TraceStage;
  /** The terminal turn summary. Streamed once. */
  "turn-metrics": TurnMetrics;
}

/**
 * Kind -> accent color. The demo surface is dark; these are tuned to read
 * on `#0a0a0a`. `inference` uses the brand magenta (it is the headline
 * stage); tools and the Phase-2 kinds get distinct, legible hues.
 *
 * Keep one entry per `TraceStageKind` member. `traceKindColor()` falls back
 * gracefully for any kind not yet listed.
 */
export const TRACE_KIND_COLORS: Record<TraceStageKind, string> = {
  inference: "#ff0096", // brand magenta — the model generation step
  tool: "#22d3ee", // cyan — a tool call
  guardrail: "#a78bfa", // violet — Phase 2 NeMo Guardrails rail
  retrieval: "#fbbf24", // amber — Phase 2 cuVS / nx-cugraph retrieval
};

/** Neutral fallback for a kind not present in `TRACE_KIND_COLORS`. */
const TRACE_KIND_FALLBACK_COLOR = "rgba(255,255,255,0.55)";

/**
 * Resolve a stage kind to its marker color. Never throws — an unrecognized
 * kind (e.g. a Phase-3+ addition this build predates) yields a neutral
 * gray so the timeline still renders.
 */
export function traceKindColor(kind: string): string {
  return (
    (TRACE_KIND_COLORS as Record<string, string>)[kind] ??
    TRACE_KIND_FALLBACK_COLOR
  );
}

/**
 * Estimate a token count from a finished text blob.
 *
 * The vLLM streaming `delta` frames carry no usage object mid-stream, and
 * the OpenAI-compat `/v1/chat/completions` stream only emits `usage` if
 * `stream_options.include_usage` is set — which the agent route does not
 * send. Rather than add a request param (and risk perturbing the live
 * serving contract), Phase 1 estimates: ~4 characters per token is the
 * widely-used rule of thumb for English + clinical prose and is honest as
 * a labeled *estimate*. The trace UI labels it "est." so no fabricated
 * precision is implied. If Phase 2+ turns on `include_usage`, swap this
 * for the exact server count.
 *
 * Returns 0 for empty input so an answer with no streamed text (a pure
 * tool turn, a refusal) reports 0 rather than a misleading 1.
 */
const CHARS_PER_TOKEN = 4;
export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / CHARS_PER_TOKEN));
}

/**
 * Tokens-per-second, guarded against a divide-by-zero on a sub-millisecond
 * or unmeasured generation window. Rounded to one decimal for display.
 */
export function tokensPerSecond(tokens: number, durationMs: number): number {
  if (durationMs <= 0 || tokens <= 0) return 0;
  return Math.round((tokens / (durationMs / 1000)) * 10) / 10;
}
