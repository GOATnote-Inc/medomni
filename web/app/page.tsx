import { DemoBanner } from "@/components/DemoBanner";
import { Composer } from "@/components/Composer";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <>
      <DemoBanner />
      <main className="flex-1 w-full max-w-4xl mx-auto px-6 py-10 flex flex-col gap-8">
        <header className="flex flex-col gap-3">
          <p className="text-xs tracking-widest uppercase text-slate-500 font-medium">
            GOATnote · MedOmni
          </p>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-slate-900">
            Free, open medical reasoning. Built on NVIDIA Nemotron.
          </h1>
          <p className="max-w-2xl text-base text-slate-600 leading-relaxed">
            Ask a clinical question below. Multimodal — text, image, voice.
            Runs on NVIDIA Blackwell B300; no signup, no fees, no API keys.
            Open-weight stack on the same lineage that NVIDIA backs for
            <a
              href="https://huggingface.co/google/medgemma-4b-it"
              className="underline decoration-slate-400 underline-offset-2 hover:text-slate-700"
              target="_blank"
              rel="noreferrer"
            >{" "}MedGemma{" "}</a>
            and friends.
          </p>
          <p className="max-w-2xl text-sm text-slate-500 leading-relaxed">
            Source: <a
              href="https://github.com/GOATnote-Inc/medomni"
              className="underline decoration-slate-400 underline-offset-2 hover:text-slate-700"
              target="_blank"
              rel="noreferrer"
            >GOATnote-Inc/medomni</a>{" "}
            · Apache 2.0 · model card and methodology in repo.
          </p>
        </header>

        <Composer />

        <section className="text-xs text-slate-500 leading-relaxed border-t border-slate-200 pt-6 space-y-2">
          <p>
            <strong className="text-slate-700">Pricing:</strong> free for
            clinicians and researchers. The compute is sponsored. No PHI
            routing — public demo only.
          </p>
          <p>
            <strong className="text-slate-700">What this is:</strong> Nemotron-3-Nano-Omni
            served on a Blackwell B300, exposed through a thin BFF. Open-weight
            model, open-source app code, manifest-locked answers. The aim is a
            MedGemma-equivalent reference deployment on the NVIDIA stack —
            something any institution can replicate behind their own firewall.
          </p>
        </section>
      </main>
      <Footer />
    </>
  );
}
