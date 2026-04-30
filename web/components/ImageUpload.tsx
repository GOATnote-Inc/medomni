"use client";

import { useCallback, useRef, useState } from "react";

// Client-side image picker with EXIF strip + 2048px long-edge clamp.
// Re-encodes via canvas → JPEG q=0.92, which guarantees no GPS / camera
// metadata survives. Output is a data: URL that the parent passes as
// content[].image_url.url to the OpenAI-compat multimodal API.
//
// Supports drag-drop on a parent container via the exposed onDrop handler.

const MAX_BYTES = 12 * 1024 * 1024; // 12 MB raw
const LONG_EDGE_MAX = 2048;
const ACCEPT = "image/jpeg,image/png,image/webp";

export interface ImageUploadHandle {
  open: () => void;
  handleFiles: (files: FileList | File[]) => void;
}

interface ImageUploadProps {
  onImage: (dataUrl: string, fileName: string) => void;
  onError: (msg: string) => void;
  disabled?: boolean;
}

export async function processImage(file: File): Promise<string> {
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const i = new Image();
    i.onload = () => {
      URL.revokeObjectURL(url);
      resolve(i);
    };
    i.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("decode failed"));
    };
    i.src = url;
  });

  const ratio = Math.min(1, LONG_EDGE_MAX / Math.max(img.width, img.height));
  const w = Math.round(img.width * ratio);
  const h = Math.round(img.height * ratio);

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("canvas ctx unavailable");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  return canvas.toDataURL("image/jpeg", 0.92);
}

export function ImageUpload({ onImage, onError, disabled }: ImageUploadProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState(false);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const file = Array.from(files)[0];
      if (!file) return;
      if (file.size > MAX_BYTES) {
        onError(`Image too large (${Math.round(file.size / 1024 / 1024)} MB; max 12 MB).`);
        return;
      }
      if (!ACCEPT.split(",").includes(file.type)) {
        onError(`Unsupported type: ${file.type}. Use JPEG, PNG, or WebP.`);
        return;
      }
      setBusy(true);
      try {
        const dataUrl = await processImage(file);
        onImage(dataUrl, file.name);
      } catch (e) {
        onError(`Image processing failed: ${(e as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [onImage, onError],
  );

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          if (e.target.files) handleFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        disabled={disabled || busy}
        onClick={() => inputRef.current?.click()}
        title="Attach an image (EXIF stripped client-side)"
        aria-label="Attach image"
        className={
          busy
            ? "p-2 rounded-md text-slate-400 cursor-wait"
            : "p-2 rounded-md text-slate-500 hover:bg-slate-200 hover:text-slate-800 transition-colors"
        }
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
      </button>
    </>
  );
}

// Tiny helper for parent to wire drag-drop on an outer container
export function useDropImage(handler: (file: File) => void) {
  return useCallback(
    (e: React.DragEvent<HTMLElement>) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handler(f);
    },
    [handler],
  );
}
