// Inline render tests for TraceTimeline + SystemPanel — the Phase-1
// "make the stack visible" UI components.
//
// No test runner is configured in web/package.json, so this file is a
// runnable assertion script intended for `tsx`. Exits non-zero on first
// failure. Same harness shape as SessionProvider.test.ts.
//
// Run:
//   cd web && npx tsx components/4uwhat/TraceTimeline.test.ts
//
// Both components are presentational enough to render headlessly with
// react-dom/server's renderToStaticMarkup (react-dom is already a
// dependency). renderToStaticMarkup does NOT run effects — so SystemPanel
// renders its static-facts shell (the pre-poll state), which is exactly
// the state we want to assert is honest before any /api/telemetry response.
//
// Covers:
//   1. TraceTimeline self-hides when there are no stages and no metrics.
//   2. TraceTimeline renders each stage's label + measured duration.
//   3. TraceTimeline renders the turn summary (TTFT, tok/s, total).
//   4. TraceTimeline labels the token count as an estimate (honesty).
//   5. SystemPanel renders the model, serving stack, and sovereignty line.
//   6. SystemPanel's pre-poll metrics state reads "unavailable", not a
//      fabricated number.

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { TraceTimeline } from "./TraceTimeline";
import { SystemPanel } from "./SystemPanel";
import type { TraceStage, TurnMetrics } from "@/lib/4uwhat/trace";

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

// Fixtures mirroring what the /api/agent route emits.
const STAGES: TraceStage[] = [
  {
    id: "stage-0",
    label: "Nemotron-3-Nano-Omni inference",
    kind: "inference",
    startedAtMs: 0,
    durationMs: 1820,
    detail: "step 1 · vLLM · requested 1 tool call",
  },
  {
    id: "stage-1",
    label: "tool: pubmed_search",
    kind: "tool",
    startedAtMs: 1820,
    durationMs: 640,
    detail: '"sepsis lactate clearance"',
  },
  {
    id: "stage-2",
    label: "Nemotron-3-Nano-Omni inference",
    kind: "inference",
    startedAtMs: 2460,
    durationMs: 2110,
    detail: "step 2 · vLLM · final answer",
  },
];

const METRICS: TurnMetrics = {
  ttftMs: 410,
  totalMs: 4570,
  tokens: 512,
  tokPerSec: 47.3,
};

async function main(): Promise<void> {
  await test("TraceTimeline: renders nothing with no stages and no metrics", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: [], streaming: false }),
    );
    assert(html === "", `expected empty render, got: ${html.slice(0, 80)}`);
  });

  await test("TraceTimeline: renders each stage label", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: STAGES, metrics: METRICS }),
    );
    assert(
      html.includes("Nemotron-3-Nano-Omni inference"),
      "inference stage label missing",
    );
    assert(html.includes("tool: pubmed_search"), "tool stage label missing");
  });

  await test("TraceTimeline: renders measured per-stage durations", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: STAGES, metrics: METRICS }),
    );
    // 1820 ms -> "1.8 s", 640 ms -> "640 ms", 2110 ms -> "2.1 s".
    assert(html.includes("1.8 s"), "1820ms stage duration not rendered as 1.8 s");
    assert(html.includes("640 ms"), "640ms stage duration not rendered");
    assert(html.includes("2.1 s"), "2110ms stage duration not rendered as 2.1 s");
  });

  await test("TraceTimeline: renders the turn summary (TTFT, tok/s, total)", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: STAGES, metrics: METRICS }),
    );
    assert(html.includes("TTFT"), "TTFT label missing from summary");
    assert(html.includes("410 ms"), "TTFT value (410 ms) missing");
    assert(html.includes("47.3"), "tok/s value (47.3) missing");
    assert(html.includes("4.6 s"), "total time (4570ms -> 4.6 s) missing");
  });

  await test("TraceTimeline: labels token count as an estimate (honesty)", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: STAGES, metrics: METRICS }),
    );
    // The trace must not imply exact token precision the vLLM stream
    // does not provide. The footnote says "estimated".
    assert(html.includes("estimated"), "token count must be labeled estimated");
    assert(html.includes("512"), "estimated token count value missing");
  });

  await test("TraceTimeline: shows a stage count for stage-only (still streaming)", () => {
    // Mid-stream: stages arrived, no turn-metrics yet.
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, {
        stages: [STAGES[0]],
        streaming: true,
      }),
    );
    assert(html.includes("1 STAGE"), "singular stage count missing");
    assert(html.includes("measuring"), "streaming state should show 'measuring'");
  });

  await test("TraceTimeline: stage count pluralizes", () => {
    const html = renderToStaticMarkup(
      createElement(TraceTimeline, { stages: STAGES, metrics: METRICS }),
    );
    assert(html.includes("3 STAGES"), "plural stage count missing");
  });

  await test("SystemPanel: renders model and serving stack", () => {
    const html = renderToStaticMarkup(createElement(SystemPanel, {}));
    assert(
      html.includes("Nemotron-3-Nano-Omni"),
      "model name missing from SystemPanel",
    );
    assert(html.includes("vLLM"), "serving stack missing from SystemPanel");
  });

  await test("SystemPanel: renders the sovereignty line", () => {
    const html = renderToStaticMarkup(createElement(SystemPanel, {}));
    assert(
      html.includes("Sovereign") && html.includes("no third-party AI APIs"),
      "sovereignty line missing or incomplete",
    );
  });

  await test("SystemPanel: pre-poll metrics state is honest (unavailable, no fake numbers)", () => {
    // renderToStaticMarkup runs no effects, so this is the state before
    // the first /api/telemetry response. It must say metrics are
    // unavailable rather than show a fabricated value.
    const html = renderToStaticMarkup(createElement(SystemPanel, {}));
    assert(
      html.includes("Metrics unavailable"),
      "pre-poll SystemPanel must show 'Metrics unavailable'",
    );
    // The "still serving" reassurance must be present so an unavailable
    // telemetry read does not read as a model outage.
    assert(
      html.includes("still") && html.includes("serving"),
      "unavailable state must clarify the model is still serving",
    );
  });

  console.log("");
  console.log(`TraceTimeline/SystemPanel tests: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
}

void main();
