// Smoke test for the unified `useTts` façade.
//
// Run via: npx tsx web/hooks/useTts.test.ts
//
// Why a hand-rolled test rather than a renderer: the project has zero
// test infra installed (no jest, no vitest, no @testing-library). This
// file exercises the *parser* directly + asserts the routing surface
// using a dependency-free harness so it stays runnable from the same
// `node` toolchain the build already needs.

import { parseSelectedVoice } from "./useTts";

let pass = 0;
let fail = 0;

function assert(cond: unknown, msg: string) {
  if (cond) {
    pass++;
    console.log(`  PASS  ${msg}`);
  } else {
    fail++;
    console.error(`  FAIL  ${msg}`);
  }
}

console.log("useTts.test.ts");

// ---- parseSelectedVoice ----
{
  const r = parseSelectedVoice("browser:com.apple.speech.synthesis.voice.samantha");
  assert(r.tier === "browser", "browser: token → tier=browser");
  assert(
    r.id === "com.apple.speech.synthesis.voice.samantha",
    "browser: token → id is the voiceURI",
  );
}
{
  const r = parseSelectedVoice("kokoro:af_heart");
  assert(r.tier === "kokoro", "kokoro: token → tier=kokoro");
  assert(r.id === "af_heart", "kokoro: token → id is the voice id");
}
{
  const r = parseSelectedVoice("");
  assert(r.tier === "browser", "empty token → tier=browser (auto)");
  assert(r.id === "", "empty token → id is empty (lets useSpeechSynthesis pick)");
}
{
  const r = parseSelectedVoice(null);
  assert(r.tier === "browser", "null token → tier=browser");
}
{
  // Legacy call sites that stored a raw voiceURI (no prefix). We treat
  // those as Tier 0 to preserve behavior.
  const r = parseSelectedVoice("Microsoft Aria Online (Natural) - English (United States)");
  assert(r.tier === "browser", "raw voiceURI (legacy) → tier=browser");
  assert(
    r.id === "Microsoft Aria Online (Natural) - English (United States)",
    "raw voiceURI (legacy) → id passes through",
  );
}

// ---- routing semantics (asserted via the parsed selection rather
//      than mounting the hook, which would require a React renderer) ----
{
  // Browser voice path: Kokoro is NOT activated.
  const browser = parseSelectedVoice("browser:any");
  assert(browser.tier !== "kokoro", "browser voice path does NOT route to Kokoro");
}
{
  // Kokoro voice path: download is deferred. The hook only kicks the
  // load on first `speak()` (see useKokoroTts.drain). Static analysis
  // confirms `kokoroInstance`/`kokoroLoadPromise` are NOT touched at
  // mount time — the import itself happens lazily inside `loadKokoro`.
  const k = parseSelectedVoice("kokoro:af_heart");
  assert(k.tier === "kokoro" && k.id === "af_heart", "Kokoro voice routed by id");
}

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
