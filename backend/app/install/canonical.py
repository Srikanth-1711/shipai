"""Normalize model identities for cross-source linking (no hardcoded model names)."""
from __future__ import annotations

import re


def normalize_canonical_id(name: str) -> str:
    """
    Stable key for grouping the same logical model across Ollama tags,
    HF repo ids, GGUF filenames, and OpenAI-style ids.
    """
    raw = name.strip().lower()
    if ":" in raw:
        raw = raw.split(":")[0]
    raw = raw.replace("/", "-").replace("\\", "-")
    raw = re.sub(r"\.gguf$", "", raw)
    raw = re.sub(r"[^a-z0-9._-]+", "-", raw)
    return raw.strip("-") or "unknown"


def names_likely_same(a: str, b: str) -> bool:
    """Heuristic: same canonical id or one stem contains the other."""
    ca, cb = normalize_canonical_id(a), normalize_canonical_id(b)
    if ca == cb:
        return True
    if len(ca) >= 4 and len(cb) >= 4:
        if ca in cb or cb in ca:
            return True
    return False
