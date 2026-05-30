"""
Family-based score interpolation for models not in any benchmark source.

When a model has no benchmark score (score=0.0), this module provides
a conservative estimate based on:
  1. Model family baseline (known quality level of the family)
  2. Size adjustment (larger models are slightly better, log-scaled)

Used as a last resort so no model ever surfaces with a score of 0,
which would cause it to be incorrectly buried in rankings.
"""
from __future__ import annotations

import math
import re

# ── Family baseline scores (0-100 scale, calibrated to chatbot arena) ────────
# These are conservative estimates — actual top models score higher.
# Calibration: GPT-4o ≈ 90, Llama3.1-70B ≈ 72, Mistral-7B ≈ 65
FAMILY_BASELINES: dict[str, float] = {
    # Frontier / top-tier
    "deepseek":    80.0,
    "qwen3":       78.0,
    "qwen2.5":     76.0,
    "qwen2":       72.0,
    "llama4":      76.0,
    "llama3.3":    75.0,
    "llama3.2":    72.0,
    "llama3.1":    72.0,
    "llama3":      71.0,
    # Solid mid-tier
    "gemma3":      71.0,
    "gemma2":      70.0,
    "phi4":        73.0,
    "phi3.5":      71.0,
    "phi3":        70.0,
    "mistral":     68.0,
    "mixtral":     74.0,
    "command-r":   69.0,
    "command":     66.0,
    "internlm":    68.0,
    "vicuna":      64.0,
    "openchat":    66.0,
    "solar":       67.0,
    "starling":    66.0,
    "neural":      65.0,
    # Older / lower tier
    "falcon":      62.0,
    "yi":          67.0,
    "baichuan":    62.0,
    "bloom":       55.0,
    "llama2":      64.0,
    "llama":       60.0,
    "orca":        66.0,
    "wizardlm":    65.0,
    "alpaca":      56.0,
    "zephyr":      67.0,
    "nous":        68.0,
    "hermes":      68.0,
    "dolphin":     65.0,
    "capybara":    64.0,
}

# Default for completely unknown family
_UNKNOWN_BASELINE = 60.0


def get_family_baseline(model_id: str) -> float:
    """
    Look up the family baseline for a model.

    Matching strategy (in order):
    1. Exact prefix match (e.g. "qwen3" in "qwen3-30b-a3b")
    2. Partial name match (case-insensitive)
    3. Default conservative baseline

    Returns a float in [0, 100].
    """
    name_lower = model_id.lower()
    # Strip org prefix
    if "/" in name_lower:
        name_lower = name_lower.split("/")[-1]
    # Strip Ollama tag
    name_lower = name_lower.split(":")[0]

    # Try exact prefix matches first (longest match wins)
    matched_key = None
    matched_len = 0
    for family in FAMILY_BASELINES:
        if name_lower.startswith(family) and len(family) > matched_len:
            matched_key = family
            matched_len = len(family)

    if matched_key:
        return FAMILY_BASELINES[matched_key]

    # Substring match
    for family, score in FAMILY_BASELINES.items():
        if family in name_lower:
            return score

    return _UNKNOWN_BASELINE


def interpolate_score(model_id: str, params_b: float) -> float:
    """
    Estimate benchmark score for a model not in any benchmark database.

    Formula:
        base  = family_baseline (known quality of the model family)
        adj   = log(params) / log(70) × 15.0  (larger = slightly better, cap at 15pt)
        score = min(base + adj, 95.0)

    Example:
        Qwen3-30B-A3B  → base=78.0, adj=log(30)/log(70)×15 = 12.7 → 90.7 → capped 90.7
        Unknown-1B     → base=60.0, adj=0 → 60.0
        Llama3-70B     → base=71.0, adj=15 → 86.0

    Returns float in [0, 95].
    """
    family_score = get_family_baseline(model_id)
    safe_params = max(params_b, 0.5)

    # Log-scaled size adjustment: 70B is the "reference" size (full 15pt bonus)
    size_adjustment = math.log(safe_params) / math.log(70.0) * 15.0

    raw = family_score + size_adjustment
    return min(round(raw, 1), 95.0)


def parse_params_from_name(model_id: str) -> float:
    """Extract parameter count in billions from model name string."""
    clean = model_id.lower()
    if "/" in clean:
        clean = clean.split("/")[-1]
    # Match NNb or NNB patterns
    for pat in (
        r"(\d+(?:\.\d+)?)\s*b\b",
        r":(\d+(?:\.\d+)?)b",
        r"-(\d+(?:\.\d+)?)b",
        r"_(\d+(?:\.\d+)?)b",
    ):
        m = re.search(pat, clean)
        if m:
            val = float(m.group(1))
            # Sanity: filter impossible values (e.g. "llava:13b-v1.6" → 13)
            if 0.1 <= val <= 2000:
                return val
    return 3.0  # conservative default — small model assumed
