export function Footer() {
  return (
    <footer className="w-full border-t border-slate-200 bg-white">
      <div className="max-w-6xl mx-auto px-6 py-6 flex flex-col sm:flex-row gap-4 sm:items-center sm:justify-between text-xs text-slate-500">
        <div>
          © {new Date().getFullYear()} GOATnote, Inc. ·{" "}
          <a
            href="https://github.com/GOATnote-Inc/medomni/blob/main/LICENSE"
            className="underline decoration-slate-300 underline-offset-2 hover:text-slate-700"
            target="_blank"
            rel="noreferrer"
          >
            Apache 2.0
          </a>
          {" · "}
          <a
            href="https://thegoatnote.com"
            className="underline decoration-slate-300 underline-offset-2 hover:text-slate-700"
          >
            thegoatnote.com
          </a>
        </div>
        <div className="flex gap-4">
          <a
            href="https://github.com/GOATnote-Inc/medomni"
            className="hover:text-slate-700"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
          <a
            href="https://github.com/GOATnote-Inc/medomni/blob/main/findings/research/2026-04-29-medomni-v1-northstar/SPEC.md"
            className="hover:text-slate-700"
            target="_blank"
            rel="noreferrer"
          >
            Architecture
          </a>
          <a
            href="https://github.com/GOATnote-Inc/medomni/tree/main/results"
            className="hover:text-slate-700"
            target="_blank"
            rel="noreferrer"
          >
            Manifests
          </a>
        </div>
      </div>
    </footer>
  );
}
