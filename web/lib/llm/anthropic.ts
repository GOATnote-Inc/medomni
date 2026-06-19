// Anthropic client + provider selector for the MedOmni inference path.
//
// 2026-06: MedOmni's inference was migrated off the self-hosted Nemotron-Omni
// vLLM container (Brev pod `exact-kind-orca`, ~$100/day) to the Claude API.
// The migration is gated behind MEDOMNI_LLM_PROVIDER so the old vLLM path stays
// callable for instant rollback while the GPU is kept warm. See CLAUDE.md §0/§2
// for the sanctioned-cloud-key override of the original sovereign design.
//
// The API key is read by the SDK from the environment automatically (the
// canonical env-var name); we never reference the literal here, both to keep
// the key out of source and because the repo's pre-commit blocklist forbids it
// outside .env.example.

import Anthropic from "@anthropic-ai/sdk";

// Primary clinical model. Opus 4.8 is the accuracy winner for emergency-med
// reasoning in our own HealthCraft eval (Pass@1 23.7% vs GPT-5.5 13.7%).
export const OPUS_MODEL_ID = "claude-opus-4-8";

// Latency-critical intent classifier (?profile=v_final). Haiku, not Opus —
// this is a 1-of-5 router behind a 2s budget that already degrades to a
// keyword heuristic on timeout.
export const CLASSIFIER_MODEL_ID = "claude-haiku-4-5";

export type LlmProvider = "anthropic" | "vllm";

/**
 * Active inference provider. Defaults to `anthropic` (the post-migration
 * default). Set MEDOMNI_LLM_PROVIDER=vllm to fall back to the self-hosted
 * Nemotron endpoint (instant rollback while orca is warm).
 */
export function llmProvider(): LlmProvider {
  return process.env.MEDOMNI_LLM_PROVIDER === "vllm" ? "vllm" : "anthropic";
}

let _client: Anthropic | null = null;

/** Lazily-constructed singleton Anthropic client (resolves the key from env). */
export function getAnthropic(): Anthropic {
  if (!_client) _client = new Anthropic();
  return _client;
}
