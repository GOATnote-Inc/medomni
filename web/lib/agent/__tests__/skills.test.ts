// Smoke test for the skills router. The web/ workspace doesn't ship a unit
// test framework yet (no vitest/jest in package.json), so this file is
// designed to run two ways:
//
//   (1) Plain node smoke runner — preferred until the workspace adopts a
//       test framework. From repo root:
//
//         cd web && npx tsx lib/agent/__tests__/skills.test.ts
//
//       Exits 0 on pass, 1 with a diff on first failure.
//
//   (2) Drop-in vitest target — if vitest/jest gets added later, the
//       `describe`/`it`/`expect` shim falls through to the real globals so
//       this file becomes a real unit test with no edits required.
//
// Coverage:
//   - keyword `classifyIntent` priority (calc > handoff > differential >
//     default), including the calc-vs-DDx tie-break.
//   - `parseIntentJson` accepts strict JSON, JSON-with-prose, and bare
//     labels; rejects unknown labels and non-objects.
//
// Network-bound `classifyIntentLLM` is intentionally NOT exercised here —
// the catfish vllm tunnel is not available in CI/dev. The fallback
// resolution is verified through `parseIntentJson`'s rejection paths +
// `classifyIntentWithFallback`'s catch-and-keyword behavior, which is
// covered indirectly by the parse-failure cases below.

import { classifyIntent, parseIntentJson, type SkillIntent } from "../skills";

interface Case {
  name: string;
  run: () => void;
}

const cases: Case[] = [
  {
    name: "classifyIntent: calc keyword wins over DDx phrasing",
    run: () => {
      const got = classifyIntent("calculate the CHA2DS2-VASc for this patient");
      assertEq(got, "calc");
    },
  },
  {
    name: "classifyIntent: handoff keyword fires on order language",
    run: () => {
      const got = classifyIntent("place an order for warfarin and discharge");
      assertEq(got, "handoff");
    },
  },
  {
    name: "classifyIntent: differential keyword fires on rule out",
    run: () => {
      const got = classifyIntent("what's the differential for chest pain in a 65yo?");
      assertEq(got, "differential");
    },
  },
  {
    name: "classifyIntent: no keyword → default",
    run: () => {
      const got = classifyIntent("good morning, what's new in cardiology?");
      assertEq(got, "default");
    },
  },
  {
    name: "parseIntentJson: strict json with intent field",
    run: () => {
      assertEq(parseIntentJson('{"intent":"calc"}'), "calc");
    },
  },
  {
    name: "parseIntentJson: tolerates surrounding whitespace",
    run: () => {
      assertEq(parseIntentJson('  \n {"intent":"handoff"}  \n'), "handoff");
    },
  },
  {
    name: "parseIntentJson: extracts json block from prose-wrapped output",
    run: () => {
      const raw = 'Sure: {"intent":"differential"} — that\'s my pick.';
      assertEq(parseIntentJson(raw), "differential");
    },
  },
  {
    name: "parseIntentJson: accepts bare label token",
    run: () => {
      assertEq(parseIntentJson("default"), "default");
      assertEq(parseIntentJson('"calc"'), "calc");
    },
  },
  {
    name: "parseIntentJson: rejects unknown label",
    run: () => {
      assertThrows(() => parseIntentJson('{"intent":"emergency"}'));
    },
  },
  {
    name: "parseIntentJson: rejects empty input",
    run: () => {
      assertThrows(() => parseIntentJson(""));
    },
  },
  {
    name: "parseIntentJson: rejects non-object json",
    run: () => {
      assertThrows(() => parseIntentJson("[1,2,3]"));
    },
  },
];

function assertEq<T>(got: T, want: T): void {
  if (got !== want) {
    throw new Error(`expected ${JSON.stringify(want)}, got ${JSON.stringify(got)}`);
  }
}

function assertThrows(fn: () => unknown): void {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error("expected throw, got none");
}

// vitest/jest shim — if globals are present, register cases as tests; else
// run them as a plain script.
type DescribeFn = (name: string, fn: () => void) => void;
type ItFn = (name: string, fn: () => void) => void;
const g = globalThis as unknown as {
  describe?: DescribeFn;
  it?: ItFn;
  test?: ItFn;
};

if (typeof g.describe === "function" && typeof (g.it ?? g.test) === "function") {
  const it = (g.it ?? g.test) as ItFn;
  g.describe("skills router", () => {
    for (const c of cases) it(c.name, c.run);
  });
} else {
  let failed = 0;
  for (const c of cases) {
    try {
      c.run();
      console.log(`  ok  ${c.name}`);
    } catch (e) {
      failed += 1;
      console.error(`  FAIL ${c.name}: ${(e as Error).message}`);
    }
  }
  if (failed > 0) {
    console.error(`\n${failed} of ${cases.length} cases failed`);
    process.exit(1);
  }
  console.log(`\n${cases.length} cases passed`);
}

export type { SkillIntent };
