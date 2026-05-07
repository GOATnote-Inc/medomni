"""MANIFEST.sha256 builder + verifier.

The manifest is a content-addressable record of every artifact that defines
a ship-rule eval run: the adapter, base-model snapshot dir, eval scripts,
corpus pin files, decode params, and per-benchmark output JSONs. It enables
bit-exact reproduction (modulo nondeterministic kernels) and bit-exact diff
of two runs.

Format: one line per file, "<sha256>  <repo-relative-path>".
Lines starting with "#" are comments (single-line metadata), e.g. the
git rev-parse of HEAD at run time.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a single file. Stable across Python versions."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def git_rev_parse(repo: Path, ref: str = "HEAD") -> str:
    """Return short git SHA at ref. Empty string if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", ref],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def gather_files(roots: Iterable[Path], exts: tuple[str, ...] | None = None) -> list[Path]:
    """Recursively collect files under each root. If exts given, filter by suffix.

    Skips dot-dirs, __pycache__, .venv, node_modules.
    """
    skip_dirs = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".ruff_cache"}
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            out.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                p = Path(dirpath) / fn
                if exts and p.suffix not in exts:
                    continue
                out.append(p)
    return sorted(out)


def write_manifest(
    *,
    out_path: Path,
    repo_root: Path,
    files: Iterable[Path],
    metadata: dict | None = None,
) -> dict[str, str]:
    """Write a sha256 manifest. Returns {rel_path: sha256} for in-process use."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_path: dict[str, str] = {}
    lines: list[str] = []
    if metadata:
        for k in sorted(metadata):
            v = metadata[k]
            lines.append(f"# {k}={v}")
    for p in sorted(set(files)):
        if not p.is_file():
            continue
        digest = sha256_file(p)
        try:
            rel = p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            # outside repo (e.g. /workspace/v2.5-prod/adapter on pod) — store absolute.
            rel = p.as_posix()
        by_path[rel] = digest
        lines.append(f"{digest}  {rel}")
    out_path.write_text("\n".join(lines) + "\n")
    return by_path


def write_manifest_jsonl(out_path: Path, entries: list[dict]) -> None:
    """Companion JSONL for richer per-file metadata (size, role, sha)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e, sort_keys=True) + "\n")


def verify_manifest(manifest_path: Path, repo_root: Path) -> tuple[bool, list[str]]:
    """Re-hash every entry; return (all_ok, list_of_failures)."""
    failures: list[str] = []
    text = manifest_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            failures.append(f"malformed line: {line!r}")
            continue
        expected_sha, rel = parts
        candidate = Path(rel)
        if not candidate.is_absolute():
            candidate = repo_root / rel
        if not candidate.is_file():
            failures.append(f"missing: {rel}")
            continue
        got = sha256_file(candidate)
        if got != expected_sha:
            failures.append(f"sha mismatch on {rel}: got {got}, want {expected_sha}")
    return (len(failures) == 0, failures)
