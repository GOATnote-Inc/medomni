import { DemoBanner } from "@/components/DemoBanner";
import { Composer } from "@/components/Composer";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <>
      <DemoBanner />
      <main className="flex-1 w-full max-w-6xl mx-auto px-6 py-10 flex flex-col gap-8">
        <header className="flex flex-col gap-2">
          <p className="text-xs tracking-widest uppercase text-slate-500 font-medium">
            GOATnote · MedOmni
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-900">
            Sovereign medical reasoning, cited and reproducible.
          </h1>
          <p className="max-w-2xl text-base text-slate-600 leading-relaxed">
            Nemotron-3-Nano-Omni on NVIDIA Blackwell B300, with a 30-fixture
            held-out clinical benchmark, a 9-layer reproducibility manifest, and
            cross-family judge ensemble. No cloud LLM keys in the data path.
            <span className="block mt-2 text-sm text-slate-500">
              N=30 held-out mean 0.378 · manifest sha256{" "}
              <code className="font-mono text-xs">f9372e0cc948</code> · open
              source at{" "}
              <a
                href="https://github.com/GOATnote-Inc/medomni"
                className="underline decoration-slate-400 underline-offset-2 hover:text-slate-700"
                target="_blank"
                rel="noreferrer"
              >
                GOATnote-Inc/medomni
              </a>
            </span>
          </p>
        </header>

        <Composer />

        <section className="text-xs text-slate-500 leading-relaxed border-t border-slate-200 pt-6">
          <p>
            <strong className="text-slate-700">For clinicians and researchers:</strong>{" "}
            this surface is for evaluation. Answers stream from a sovereign,
            on-premise NVIDIA stack with no third-party LLM calls. Demo mode
            does not log or retain prompts. Frontier-augmented Pro tier
            (Anthropic / OpenAI via BAA-covered route) ships post-Nebius.
          </p>
        </section>
      </main>
      <Footer />
    </>
  );
}
