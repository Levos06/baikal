#!/usr/bin/env python3
"""
Concatenate repository source-like files into one UTF-8 text bundle.
Skips VCS, caches, common data/binary paths, and files larger than --max-bytes.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path


# Directory names (any path component) to skip entirely
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".ipynb_checkpoints",
        "node_modules",
        "venv",
        ".venv",
        ".env",
        ".cursor",
        ".gemini",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)

# Path prefixes (relative posix) to skip
SKIP_PREFIXES = (
    "data_processed/",
    "data_processed_100k/",
    "data_processed_50k/",
    "data_processed_50k_v2/",
    "logs/",
)

# Filename suffixes to skip (lowercase)
SKIP_FILE_SUFFIXES = frozenset(
    {
        ".h5",
        ".pyc",
        ".pyo",
        ".so",
        ".dll",
        ".dylib",
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
        ".o",
        ".a",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
    }
)

# Extensions treated as "source" (lowercase, leading dot)
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyw",
        ".pyi",
        ".sh",
        ".bash",
        ".zsh",
        ".cmake",
        ".make",
        ".sql",
        ".rs",
        ".go",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".hh",
        ".java",
        ".kt",
        ".scala",
        ".rb",
        ".php",
        ".cs",
        ".fs",
        ".swift",
        ".m",
        ".mm",
        ".pl",
        ".lua",
        ".r",
        ".jl",
        ".cu",
        ".cuh",
        ".md",
        ".rst",
        ".tex",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".dockerignore",
        ".graphql",
        ".proto",
        ".ipynb",
        ".tsx",
        ".ts",
        ".jsx",
        ".js",
        ".mjs",
        ".cjs",
        ".css",
        ".scss",
        ".html",
        ".vue",
        ".svelte",
    }
)

# Basenames (exact match) always included if under root
EXTRA_NAMES = frozenset({"Dockerfile", "Makefile", "GNUmakefile", "Justfile"})


def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.endswith(".egg-info")


def should_skip_path(rel_posix: str) -> bool:
    for p in SKIP_PREFIXES:
        if rel_posix == p.rstrip("/") or rel_posix.startswith(p):
            return True
    parts = rel_posix.split("/")
    for part in parts:
        if part in ("checkpoints", "adj_matrices"):
            return True
    return False


def is_candidate_file(path: Path, rel: Path) -> bool:
    name = path.name
    if name in EXTRA_NAMES:
        return True
    lower = name.lower()
    for suf in SKIP_FILE_SUFFIXES:
        if lower.endswith(suf):
            return False
    ext = path.suffix.lower()
    return ext in CODE_EXTENSIONS


def collect(root: Path, max_bytes: int, out: Path) -> tuple[int, int, int]:
    root = root.resolve()
    written_files = 0
    skipped_large = 0
    skipped_read = 0

    lines: list[str] = []
    header_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"# Source bundle generated at {header_ts} UTC")
    lines.append(f"# Root: {root}")
    lines.append(f"# Max bytes per file: {max_bytes}")
    lines.append("")

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # prune dirs in-place
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        base = Path(dirpath)

        for fname in sorted(filenames):
            fpath = base / fname
            try:
                rel = fpath.resolve().relative_to(root)
            except ValueError:
                continue
            rel_posix = rel.as_posix()
            if should_skip_path(rel_posix):
                continue
            if not is_candidate_file(fpath, rel):
                continue
            try:
                size = fpath.stat().st_size
            except OSError:
                skipped_read += 1
                continue
            if size > max_bytes:
                skipped_large += 1
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped_read += 1
                continue

            lines.append(f"{'=' * 72}")
            lines.append(f"FILE: {rel_posix}")
            lines.append(f"BYTES: {size}")
            lines.append(f"{'=' * 72}")
            lines.append(text.rstrip("\n"))
            lines.append("")
            written_files += 1

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written_files, skipped_large, skipped_read


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of this script)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .txt path (default: <root>/all_repo_source.txt)",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=2_000_000,
        help="Skip files larger than this (default: 2_000_000)",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out or (root / "all_repo_source.txt")).resolve()

    n_ok, n_large, n_err = collect(root, args.max_bytes, out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"Files included: {n_ok}, skipped (too large): {n_large}, skipped (read/stat error): {n_err}")


if __name__ == "__main__":
    main()
