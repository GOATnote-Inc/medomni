// Inline tests for the inference-trace data model (lib/4uwhat/trace.ts).
//
// No test runner is configured in web/package.json, so this file is a
// runnable assertion script intended for `tsx`. Exits non-zero on first
// failure. Same harness shape as SessionProvider.test.ts.
//
// Run:
//   cd web && npx tsx lib/4uwhat/trace.test.ts
//
// Covers:
//   1. estimateTokens — empty -> 0, ~4 chars/token, never below 1 for
//      non-empty input.
//   2. tokensPerSecond — basic rate, divide-by-zero guards, rounding.
//   3. traceKindColor — every declared kind resolves; an unknown future
//      kind degrades to a neutral fallback (the open-union contract).
//   4. TRACE_KIND_COLORS — has an entry for each TraceStageKind member,
//      including the Phase-2 kinds, so the union stays extensible.

import {
  TRACE_KIND_COLORS,
  estimateTokens,
  tokensPerSecond,
  traceKindColor,
  type TraceStage,
  type TraceStageKind,
  type TurnMetrics,
} from "./trace";

let passed = 0;
let failed = 0;
const failures: string[] = [];

async function test(name: string, fn: () => Promise<void> | void): Promise<void> {
  try {
    await fn();
    passed += 1;
    console.log(`  ok  ${name}`);
  } catch (e) {
    failed += 1;
    const msg = `${name}: ${(e as Error).message}`;
    failures.push(msg);
    console.error(`  FAIL ${msg}`);
  }
}

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

async function main(): Promise<void> {
  await test("estimateTokens: empty string is 0 tokens", () => {
    assert(estimateTokens("") === 0, `expected 0, got ${estimateTokens("")}`);
  });

  await test("estimateTokens: ~4 chars per token", () => {
    // 40 chars / 4 = 10
    const t = estimateTokens("a".repeat(40));
    assert(t === 10, `expected 10, got ${t}`);
  });

  await test("estimateTokens: non-empty input never reports 0", () => {
    // A 1-char string rounds to 0 by the /4 rule, but the floor is 1 so a
    // turn with output never misleadingly reads "0 tokens".
    assert(estimateTokens("a") === 1, `expected 1, got ${estimateTokens("a")}`);
    assert(estimateTokens("ab") === 1, `expected 1, got ${estimateTokens("ab")}`);
  });

  await test("tokensPerSecond: basic rate", () => {
    // 100 tokens in 2000 ms = 50 tok/s
    const r = tokensPerSecond(100, 2000);
    assert(r === 50, `expected 50, got ${r}`);
  });

  await test("tokensPerSecond: rounds to one decimal", () => {
    // 10 tokens in 3000 ms = 3.333... -> 3.3
    const r = tokensPerSecond(10, 3000);
    assert(r === 3.3, `expected 3.3, got ${r}`);
  });

  await test("tokensPerSecond: zero/negative duration -> 0 (no divide-by-zero)", () => {
    assert(tokensPerSecond(100, 0) === 0, "0 duration must yield 0");
    assert(tokensPerSecond(100, -5) === 0, "negative duration must yield 0");
  });

  await test("tokensPerSecond: zero tokens -> 0", () => {
    assert(tokensPerSecond(0, 1000) === 0, "0 tokens must yield 0");
  });

  await test("traceKindColor: every declared kind resolves to a color", () => {
    const kinds: TraceStageKind[] = [
      "inference",
      "tool",
      "guardrail",
      "retrieval",
    ];
    for (const k of kinds) {
      const c = traceKindColor(k);
      assert(
        typeof c === "string" && c.length > 0,
        `kind ${k} produced no color`,
      );
    }
  });

  await test("traceKindColor: inference is the brand magenta", () => {
    assert(
      traceKindColor("inference") === "#ff0096",
      `inference color drift: ${traceKindColor("inference")}`,
    );
  });

  await test("traceKindColor: unknown future kind degrades to a fallback", () => {
    // The open-union contract: a Phase-3+ kind this build predates must
    // NOT throw — it must yield a neutral color so the timeline renders.
    const c = traceKindColor("some-future-kind-2027");
    assert(
      typeof c === "string" && c.length > 0,
      "unknown kind must yield a non-empty fallback color",
    );
    // And it must NOT collide with a real kind's color (it's a fallback).
    assert(
      c !== traceKindColor("inference"),
      "fallback must be distinct from the inference color",
    );
  });

  await test("TRACE_KIND_COLORS: has an entry for each kind incl. Phase-2", () => {
    // The Phase-2 extensibility contract: 'guardrail' and 'retrieval' must
    // already be color-mapped so Phase 2 only needs to emit them.
    for (const k of ["inference", "tool", "guardrail", "retrieval"]) {
      assert(
        k in TRACE_KIND_COLORS,
        `TRACE_KIND_COLORS missing kind: ${k}`,
      );
    }
  });

  await test("TraceStage / TurnMetrics shapes compile with expected fields", () => {
    // Compile-time contract pinned at runtime: constructing the shapes the
    // /api/agent route emits. If a field is renamed/dropped this fails to
    // compile, which is the point.
    const stage: TraceStage = {
      id: "stage-0",
      label: "Nemotron-3-Nano-Omni inference",
      kind: "inference",
      startedAtMs: 0,
      durationMs: 1234,
      detail: "step 1 · vLLM · final answer",
    };
    const metrics: TurnMetrics = {
      ttftMs: 320,
      totalMs: 4500,
      tokens: 512,
      tokPerSec: 47.3,
    };
    assert(stage.id === "stage-0", "stage id field");
    assert(stage.kind === "inference", "stage kind field");
    assert(metrics.ttftMs === 320, "metrics ttftMs field");
    assert(metrics.tokPerSec === 47.3, "metrics tokPerSec field");
    // detail is optional — a stage with no detail is valid.
    const bare: TraceStage = {
      id: "stage-1",
      label: "tool: pubmed_search",
      kind: "tool",
      startedAtMs: 100,
      durationMs: 50,
    };
    assert(bare.detail === undefined, "detail must be optional");
  });

  console.log("");
  console.log(`trace tests: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
}

void main();
