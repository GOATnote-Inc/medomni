export function DemoBanner() {
  return (
    <div
      role="alert"
      aria-live="polite"
      className="w-full bg-amber-50 border-b border-amber-200 text-amber-900 text-sm py-2 px-4 text-center font-medium"
    >
      <strong className="font-semibold">DEMO</strong>
      <span className="mx-2 text-amber-400">·</span>
      For evaluation only. <span className="font-semibold">Do not enter PHI</span>{" "}
      or personally identifiable patient information.
    </div>
  );
}
