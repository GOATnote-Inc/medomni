"""Pod-side generators for the four PREREG benchmarks.

These functions assume a local OpenAI-compatible vllm endpoint serving the
model under test (e.g. base V0 or V0+V2.5 LoRA). They emit per-item JSONL
records with full provenance: the prompt sha, decode params hash, output
text, and the timestamp/seed for reproducibility.

NO openai SDK import. NO cloud LLM keys. httpx-only against 127.0.0.1.

Datasets are loaded from local cache when available; the caller passes
explicit paths so we do NOT depend on `datasets.load_dataset` network.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

# Decode params per PREREG eval_protocol.
DECODE_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
    "chat_template_kwargs": {"enable_thinking": False},
}


def decode_params_hash(params: dict | None = None) -> str:
    p = params if params is not None else DECODE_PARAMS
    s = json.dumps(p, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass
class GenRecord:
    item_id: str
    benchmark: str
    arm: str  # "v0" | "v25"
    seed: int
    trial: int
    prompt: str
    prompt_sha256: str
    response: str
    decode_params_sha256: str
    duration_ms: int
    expected_answer: str | None = None  # for MCQ scoring; None for open-ended
    rubric: list[dict] | None = None  # for HealthBench-style rubric grading

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "arm": self.arm,
            "seed": self.seed,
            "trial": self.trial,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "response": self.response,
            "decode_params_sha256": self.decode_params_sha256,
            "duration_ms": self.duration_ms,
            "expected_answer": self.expected_answer,
            "rubric": self.rubric,
        }


def _require_local(url: str, label: str) -> None:
    if not url.startswith(("http://127.0.0.1", "http://localhost")):
        raise ValueError(
            f"{label} must be 127.0.0.1/localhost; got {url!r}. "
            "Generators are pod-side; cloud serving violates CLAUDE.md §2."
        )


def _generate(
    client: httpx.Client,
    *,
    model: str,
    prompt: str,
    timeout_s: float,
    decode_params: dict,
    system_prompt: str | None = None,
    seed: int | None = None,
) -> tuple[str, int]:
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": msgs,
        **decode_params,
    }
    if seed is not None:
        body["seed"] = seed  # vllm honors deterministic seed.
    t0 = time.time()
    resp = client.post("chat/completions", json=body, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    msg = payload["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning") or ""
    return text, int((time.time() - t0) * 1000)


# ---------------------------------------------------------------------------
# Dataset loaders (cache-first; no datasets.load_dataset network call)
# ---------------------------------------------------------------------------


def load_medqa_usmle(cache_dir: Path, split: str = "test") -> list[dict]:
    """Load MedQA-USMLE 4-options test set from local HF cache.

    Looks for `datasets--GBaker--MedQA-USMLE-4-options/**/<split>-*.parquet`.
    Falls back to load_dataset only if local cache absent (and we then
    persist a JSONL alongside the cache for next-run determinism).
    """
    items: list[dict] = []
    parquets = list(cache_dir.rglob("*MedQA-USMLE-4-options*/**/" + f"{split}*.parquet"))
    if not parquets:
        # Last-resort online fetch — caller can disable by erroring on empty.
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
            for i, row in enumerate(ds):
                items.append(
                    {
                        "item_id": f"medqa-{split}-{i:05d}",
                        "question": row["question"],
                        "options": row["options"],
                        "answer_idx": row.get("answer_idx") or row.get("answer"),
                    }
                )
            return items
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"MedQA cache empty under {cache_dir} and online fallback failed: {e}"
            ) from e
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise RuntimeError("pyarrow needed to read MedQA parquet from cache") from e
    for pqf in parquets:
        tbl = pq.read_table(pqf)
        rows = tbl.to_pylist()
        for i, row in enumerate(rows):
            items.append(
                {
                    "item_id": f"medqa-{split}-{pqf.stem}-{i:05d}",
                    "question": row["question"],
                    "options": row["options"],
                    "answer_idx": row.get("answer_idx") or row.get("answer"),
                }
            )
    return items


def load_pubmedqa_l(cache_dir: Path) -> list[dict]:
    """Load PubMedQA-L (labeled) test set from local cache.

    Restricts to `pqa_labeled` split parquets only — the unlabeled and
    artificial splits are >250k rows of unlabeled / synthetic data and
    must NOT enter the eval set.
    """
    items: list[dict] = []
    all_parquets = list(cache_dir.rglob("*PubMedQA*/**/*.parquet"))
    parquets = [p for p in all_parquets if "pqa_labeled" in str(p)]
    if not parquets:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
            for i, row in enumerate(ds):
                items.append(
                    {
                        "item_id": f"pubmedqa-{i:05d}",
                        "question": row["question"],
                        "context": " ".join(row["context"]["contexts"])
                        if isinstance(row.get("context"), dict)
                        else (row.get("context") or ""),
                        "answer_yn": row["final_decision"],
                    }
                )
            return items
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"PubMedQA cache empty + online failed: {e}") from e
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise RuntimeError("pyarrow needed to read PubMedQA parquet") from e
    for pqf in parquets:
        tbl = pq.read_table(pqf)
        for i, row in enumerate(tbl.to_pylist()):
            ctx = row.get("context")
            ctx_text = ""
            if isinstance(ctx, dict):
                ctx_text = " ".join(ctx.get("contexts") or [])
            elif isinstance(ctx, str):
                ctx_text = ctx
            items.append(
                {
                    "item_id": f"pubmedqa-{pqf.stem}-{i:05d}",
                    "question": row["question"],
                    "context": ctx_text,
                    "answer_yn": row.get("final_decision"),
                }
            )
    return items


def load_medxpertqa_text(repo_dir: Path) -> list[dict]:
    """Load MedXpertQA-Text from a clone of TsinghuaC3I/MedXpertQA.

    Repo layout: data/Text/test.jsonl (verified per repo README at SHA pinned
    by the caller). Each row has `id`, `question`, `options` (dict A-J),
    `label`, `medical_task`, `body_system`, `question_type`.
    """
    target = repo_dir / "data" / "Text" / "test.jsonl"
    if not target.exists():
        # Try alternate layouts seen across versions of the repo.
        candidates = list(repo_dir.rglob("test*.jsonl"))
        if not candidates:
            raise RuntimeError(
                f"MedXpertQA-Text test split not found under {repo_dir}; "
                "clone https://github.com/TsinghuaC3I/MedXpertQA to that path."
            )
        target = candidates[0]
    items: list[dict] = []
    with target.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append(
                {
                    "item_id": str(row.get("id", f"medxpertqa-text-{i:05d}")),
                    "question": row["question"],
                    "options": row.get("options", {}),
                    "answer_letter": row.get("label"),
                    "medical_task": row.get("medical_task"),
                    "body_system": row.get("body_system"),
                    "question_type": row.get("question_type"),
                }
            )
    return items


def load_healthbench_hard(pin_path: Path) -> list[dict]:
    """Load the medomni HealthBench-Hard items.

    Two acceptable shapes:
    1. medomni pin YAML (shape: examples: [...])
    2. Tonic/Health-Bench-Eval-OSS-2025-07 JSONL split (shape per row:
       {prompt: [messages], rubrics: [...], prompt_id: str})

    For (2) the loader treats `pin_path` as either:
       - a path to the JSONL file directly, or
       - the medomni pin YAML; if `examples` empty, the loader looks at the
         pin's `upstream.cache_path` field and resolves a JSONL there.
    """
    import yaml  # type: ignore

    # Direct JSONL path (suffix check + content sniff for HF blob symlinks)
    if pin_path.suffix == ".jsonl":
        return _load_healthbench_jsonl(pin_path)
    head = pin_path.read_text(errors="ignore")[:64].lstrip()
    if head.startswith("{"):
        return _load_healthbench_jsonl(pin_path)

    data = yaml.safe_load(pin_path.read_text()) or {}
    examples = data.get("examples", []) or []
    if examples:
        out: list[dict] = []
        for i, ex in enumerate(examples):
            out.append(
                {
                    "item_id": str(ex.get("id", f"hb-{i:05d}")),
                    "messages": ex.get("messages")
                    or [{"role": "user", "content": ex.get("prompt", "")}],
                    "rubrics": ex.get("rubrics") or [],
                }
            )
        return out

    # Fall back to upstream JSONL pointed to by the pin metadata.
    upstream = data.get("upstream") or {}
    cache = upstream.get("cache_path") or upstream.get("local_jsonl")
    if cache:
        return _load_healthbench_jsonl(Path(cache))
    raise RuntimeError(
        f"HealthBench pin {pin_path} has no `examples:` and no `upstream.cache_path:`. "
        "Pass the Tonic/Health-Bench-Eval-OSS hard*.jsonl path directly via "
        "--healthbench-pin <path>.jsonl"
    )


def _load_healthbench_jsonl(path: Path) -> list[dict]:
    """Tonic/Health-Bench-Eval-OSS-2025-07 hard*.jsonl reader."""
    out: list[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                messages = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in prompt
                ]
            elif isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = []
            out.append(
                {
                    "item_id": str(row.get("prompt_id", f"hb-{i:05d}")),
                    "messages": messages,
                    "rubrics": row.get("rubrics") or [],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Per-benchmark prompt builders + generators
# ---------------------------------------------------------------------------


_MCQ_SYSTEM = (
    "You are a medical question-answering assistant. Read the question and "
    "candidate answers, reason carefully, and respond with the single best "
    "letter (e.g. 'A'). Output ONLY the letter on the final line."
)

_PUBMEDQA_SYSTEM = (
    "You are a biomedical question-answering assistant. Use only the supplied "
    "context. Respond with one of: yes, no, maybe — exactly that token on the "
    "final line."
)


def _build_mcq_prompt_medqa(item: dict) -> str:
    options = item["options"]
    if isinstance(options, dict):
        opt_lines = [f"{k}. {v}" for k, v in sorted(options.items())]
    else:
        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        opt_lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
    return (
        f"Question: {item['question']}\n\n"
        + "\n".join(opt_lines)
        + "\n\nAnswer with a single letter."
    )


def _build_mcq_prompt_medxpertqa(item: dict) -> str:
    options = item["options"]
    opt_lines = [f"{k}. {v}" for k, v in sorted(options.items())]
    return (
        f"Question: {item['question']}\n\n"
        + "\n".join(opt_lines)
        + "\n\nAnswer with a single letter."
    )


def _build_pubmedqa_prompt(item: dict) -> str:
    return (
        f"Context: {item['context']}\n\n"
        f"Question: {item['question']}\n\n"
        "Respond with yes, no, or maybe."
    )


def gen_for_items(
    *,
    items: list[dict],
    benchmark: str,
    arm: str,
    serve_url: str,
    serve_model: str,
    seed: int,
    trial: int,
    timeout_s: float = 180.0,
    out_jsonl: Path | None = None,
    decode_params: dict | None = None,
) -> Iterator[GenRecord]:
    """Yield (and optionally persist) GenRecords for a benchmark + arm + seed.

    Streams to JSONL as it goes — interrupt-safe.
    """
    _require_local(serve_url, "--serve-url")
    params = decode_params or DECODE_PARAMS
    params_sha = decode_params_hash(params)
    client = httpx.Client(base_url=serve_url, timeout=timeout_s)

    fh = None
    if out_jsonl is not None:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        fh = out_jsonl.open("a")

    try:
        for item in items:
            if benchmark == "medqa":
                prompt = _build_mcq_prompt_medqa(item)
                expected = item.get("answer_idx") or item.get("answer_letter")
                rubric = None
                sys_p = _MCQ_SYSTEM
            elif benchmark == "medxpertqa-text":
                prompt = _build_mcq_prompt_medxpertqa(item)
                expected = item.get("answer_letter")
                rubric = None
                sys_p = _MCQ_SYSTEM
            elif benchmark == "pubmedqa":
                prompt = _build_pubmedqa_prompt(item)
                expected = item.get("answer_yn")
                rubric = None
                sys_p = _PUBMEDQA_SYSTEM
            elif benchmark == "healthbench-hard":
                # Use the multi-turn `messages` list verbatim; treat last user
                # message as the prompt for sha purposes.
                msgs = item.get("messages") or []
                last_user = next(
                    (m["content"] for m in reversed(msgs) if m.get("role") == "user"),
                    "",
                )
                prompt = last_user
                expected = None
                rubric = item.get("rubrics") or []
                sys_p = None
            else:
                raise ValueError(f"unknown benchmark: {benchmark}")

            try:
                if benchmark == "healthbench-hard":
                    body = {
                        "model": serve_model,
                        "messages": msgs,
                        **params,
                    }
                    if seed is not None:
                        body["seed"] = seed
                    t0 = time.time()
                    resp = client.post("chat/completions", json=body, timeout=timeout_s)
                    resp.raise_for_status()
                    msg = resp.json()["choices"][0]["message"]
                    response = msg.get("content") or msg.get("reasoning") or ""
                    duration_ms = int((time.time() - t0) * 1000)
                else:
                    response, duration_ms = _generate(
                        client,
                        model=serve_model,
                        prompt=prompt,
                        timeout_s=timeout_s,
                        decode_params=params,
                        system_prompt=sys_p,
                        seed=seed,
                    )
            except httpx.HTTPError as e:
                response = f"[GEN_FAIL: {type(e).__name__}: {e}]"
                duration_ms = -1

            rec = GenRecord(
                item_id=str(item["item_id"]),
                benchmark=benchmark,
                arm=arm,
                seed=seed,
                trial=trial,
                prompt=prompt,
                prompt_sha256=prompt_hash(prompt),
                response=response,
                decode_params_sha256=params_sha,
                duration_ms=duration_ms,
                expected_answer=expected,
                rubric=rubric,
            )
            if fh is not None:
                fh.write(json.dumps(rec.to_dict()) + "\n")
                fh.flush()
            yield rec
    finally:
        if fh is not None:
            fh.close()
        client.close()
