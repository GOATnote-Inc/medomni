// Inline tests for guideline-currency.ts. No test runner is configured
// in web/package.json, so this file is a runnable assertion script
// intended for `tsx`. Exits non-zero on first failure.
//
// Run:
//   cd web && npx tsx lib/tools/guideline-currency.test.ts
//
// Covers (per A2 spec):
//   1. In-registry hit — "h pylori treatment" → relevant match, notInRegistry=false
//   2. Out-of-registry — "asthma controller therapy" → either real hit OR
//      notInRegistry=true with registryTopics, NEVER unrelated nearest neighbors
//      (the demo bug returned PPSV23/H. pylori/BPH for this query)
//   3. Completely off-topic — "quantum cryptography" → notInRegistry=true, no matches
//   4. Empty query → notInRegistry=true, no matches
//   5. Generic medical word "therapy" alone must not pull entries

import { guidelineCurrencyCheck } from "./guideline-currency";

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

async function runAll(): Promise<void> {
  await test("in-registry: h pylori treatment hits H. pylori entry", async () => {
    const out = await guidelineCurrencyCheck({ query: "h pylori treatment" });
    assert(out.notInRegistry === false, `expected notInRegistry=false, got ${out.notInRegistry}`);
    assert(out.match_count >= 1, `expected >=1 match, got ${out.match_count}`);
    const top = out.matches[0];
    assert(
      /pylori/i.test(top.concept) || /pylori/i.test(top.stale_default),
      `top match should be H. pylori, got: ${top.concept}`,
    );
  });

  await test("out-of-registry: asthma controller therapy", async () => {
    // Either it lands a relevant asthma hit (registry expanded) or
    // it returns notInRegistry=true. NEVER unrelated nearest neighbors
    // (PPSV23 / BPH / H. pylori) — that was the demo bug.
    const out = await guidelineCurrencyCheck({ query: "asthma controller therapy" });
    if (out.notInRegistry) {
      assert(out.match_count === 0, "notInRegistry=true must have 0 matches");
      assert(
        Array.isArray(out.registryTopics) && out.registryTopics.length > 0,
        "registryTopics must be present and non-empty when notInRegistry=true",
      );
    } else {
      // If we did match, it must actually be the asthma entry, not BPH/PPSV23/H. pylori.
      for (const m of out.matches) {
        const blob = `${m.concept} ${m.stale_default}`.toLowerCase();
        assert(
          /asthma|controller|gina|ics|formoterol|saba/.test(blob),
          `unrelated match returned: id=${m.id} concept="${m.concept}" — this is the demo bug`,
        );
      }
    }
  });

  await test("completely off-topic: quantum cryptography → notInRegistry", async () => {
    const out = await guidelineCurrencyCheck({ query: "quantum cryptography" });
    assert(out.notInRegistry === true, `expected notInRegistry=true, got ${out.notInRegistry}`);
    assert(out.match_count === 0, `expected 0 matches, got ${out.match_count}`);
    assert(out.matches.length === 0, "matches array must be empty");
    assert(
      Array.isArray(out.registryTopics) && out.registryTopics.length > 0,
      "registryTopics must surface for out-of-registry queries",
    );
  });

  await test("empty query → notInRegistry, no matches", async () => {
    const out = await guidelineCurrencyCheck({ query: "" });
    assert(out.notInRegistry === true, "empty query must signal notInRegistry");
    assert(out.match_count === 0, "empty query must return zero matches");
  });

  await test("single generic word 'therapy' must not pull unrelated entries", async () => {
    // 'therapy' is a stopword in tokenize? No — stopwords list does not
    // include "therapy". But the threshold rule (1-token query needs
    // ratio===1 against the entry's tokens) plus whole-word match means
    // it should match many entries — but only those where 'therapy' is
    // a literal token. We don't claim notInRegistry here; we just claim
    // every returned match must legitimately contain 'therapy'.
    const out = await guidelineCurrencyCheck({ query: "therapy" });
    for (const m of out.matches) {
      const blob = `${m.concept} ${m.stale_default}`.toLowerCase();
      assert(/\btherapy\b/.test(blob), `match ${m.id} doesn't actually contain 'therapy'`);
    }
  });

  await test("statin primary prevention hits the new entry", async () => {
    const out = await guidelineCurrencyCheck({ query: "statin primary prevention ASCVD" });
    assert(out.notInRegistry === false, "statin query should hit the new GCR-045 entry");
    const ids = out.matches.map((m) => m.id);
    assert(
      ids.some((id) => id.includes("statin") || id.includes("GCR-045")),
      `expected statin entry in matches, got: ${ids.join(", ")}`,
    );
  });

  await test("hypertension BP target hits the new entry", async () => {
    const out = await guidelineCurrencyCheck({ query: "hypertension BP target adults" });
    assert(out.notInRegistry === false, "BP-target query should hit GCR-047");
    const ids = out.matches.map((m) => m.id);
    assert(
      ids.some((id) => id.includes("htn") || id.includes("GCR-047")),
      `expected hypertension entry, got: ${ids.join(", ")}`,
    );
  });

  await test("vitamin D supplementation hits the new entry", async () => {
    const out = await guidelineCurrencyCheck({ query: "vitamin D supplementation healthy adults" });
    assert(out.notInRegistry === false, "vitamin D query should hit GCR-048");
    const ids = out.matches.map((m) => m.id);
    assert(
      ids.some((id) => id.includes("vitamin") || id.includes("GCR-048")),
      `expected vitamin D entry, got: ${ids.join(", ")}`,
    );
  });

  await test("registry size matches expected expansion", async () => {
    const out = await guidelineCurrencyCheck({ query: "anything" });
    assert(out.registry_size >= 48, `registry_size should be >=48 after A2 expansion, got ${out.registry_size}`);
  });
}

void (async () => {
  await runAll();
  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) {
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
})();
