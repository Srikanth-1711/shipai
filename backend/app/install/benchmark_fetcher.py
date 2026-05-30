"""
Live benchmark fetching from 6 sources with tiered scoring.

Source priority:
  current tier  → current (live vote / live eval) — highest trust
  frozen tier   → scored on static benchmark dataset — decays with time
  self_reported → ShipAI bundled matrix — always fallback

Static JSON is always the last fallback — never breaks offline.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger("shipai.benchmarks")

_CACHE_DIR = Path.home() / ".shipai" / "cache" / "benchmarks"

# ── Benchmark source registry (6 live sources, tiered by trust) ─────────────
# current tier: live human votes or live evaluation — no recency decay
# frozen tier:  static benchmark dataset runs — decayed over time
BENCHMARK_SOURCES: list[dict[str, Any]] = [
    # ── current-tier (live votes, live evals — no decay) ──────────────────
    {
        "name": "chatbot_arena",
        "tier": "current",
        "url": "https://huggingface.co/api/spaces/lmsys/chatbot-arena-leaderboard",
        "ttl_hours": 12,
        "parser": "arena",
    },
    {
        "name": "lmsys_vision_arena",
        "tier": "current",
        "url": "https://huggingface.co/api/spaces/lmsys/vision-arena-leaderboard",
        "ttl_hours": 12,
        "parser": "arena",
    },
    # ── frozen-tier (static benchmark, recency-decayed) ───────────────────
    {
        "name": "open_llm_leaderboard",
        "tier": "frozen",
        "url": "https://huggingface.co/api/spaces/open-llm-leaderboard/open_llm_leaderboard",
        "ttl_hours": 24,
        "parser": "hf_leaderboard",
    },
    {
        "name": "aider_coding",
        "tier": "frozen",
        "url": "https://aider.chat/assets/leaderboard.json",
        "ttl_hours": 48,
        "parser": "aider",
    },
    {
        "name": "livebench",
        "tier": "frozen",
        "url": "https://livebench.ai/livebench.json",
        "ttl_hours": 48,
        "parser": "livebench",
    },
    {
        "name": "artificial_analysis",
        "tier": "current",
        "url": "https://artificialanalysis.ai/api/models",
        "ttl_hours": 24,
        "parser": "artificial_analysis",
    },
]


# ── Evidence grading ─────────────────────────────────────────────────────────
EVIDENCE_MULTIPLIERS: dict[str, float] = {
    "direct":        1.00,
    "variant":       0.95,
    "base_model":    0.85,
    "line_interp":   0.70,
    "self_reported": 0.50,
    "interpolated":  0.65,   # family interpolation — between self_reported and direct
    "none":          0.00,
}


@dataclass
class BenchmarkScore:
    """A benchmark score for a specific model from a specific source."""
    score: float
    source: str
    tier: str = "frozen"
    evidence_grade: str = "direct"
    decay_applied: float = 0.0
    date: Optional[datetime] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark data for all models."""
    scores: dict[str, BenchmarkScore] = field(default_factory=dict)
    sources_fetched: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    used_fallback: bool = False


# ── Recency decay for FROZEN tier ────────────────────────────────────────────

def recency_decay(benchmark_date: datetime | None) -> float:
    """
    Apply recency decay to frozen-tier scores.

    Score multiplier:
      < 180 days:  1.00 (no decay)
      < 365 days:  0.90
      < 730 days:  0.75
      < 1095 days: 0.60
      ≥ 1095 days: 0.50 (floor)
    """
    if benchmark_date is None:
        return 0.60
    days_old = (datetime.now() - benchmark_date).days
    if days_old < 0:
        return 1.00
    if days_old < 180:
        return 1.00
    if days_old < 365:
        return 0.90
    if days_old < 730:
        return 0.75
    if days_old < 1095:
        return 0.60
    return 0.50


def apply_decay(score: BenchmarkScore) -> BenchmarkScore:
    """Apply recency decay to a frozen-tier score in-place."""
    if score.tier != "frozen":
        return score
    decay = recency_decay(score.date)
    score.score = score.score * decay
    score.decay_applied = round(1.0 - decay, 3)
    return score


# ── Cache layer ──────────────────────────────────────────────────────────────

def _cache_path(source_name: str) -> Path:
    return _CACHE_DIR / f"{source_name}.json"


def _is_cache_fresh(source_name: str, ttl_hours: int) -> bool:
    path = _cache_path(source_name)
    if not path.is_file():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < ttl_hours * 3600


def _load_cache(source_name: str) -> dict[str, Any] | None:
    path = _cache_path(source_name)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(source_name: str, data: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(source_name)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except OSError:
        logger.debug("Failed to write benchmark cache: %s", path)


# ── Model key normalization ───────────────────────────────────────────────────

def _normalize_model_key(model_id: str) -> str:
    """Normalize model ID for cross-source matching."""
    key = model_id.lower().strip()
    if "/" in key:
        key = key.split("/")[-1]
    # Strip Ollama version tag (e.g. llama3.1:8b → llama3.1)
    if ":" in key:
        key = key.split(":")[0]
    for suffix in ("-instruct", "-chat", "-hf", "-gguf", "-fp16", "-bf16", "-preview"):
        key = key.replace(suffix, "")
    return key.strip("-")


# ── Source-specific parsers ───────────────────────────────────────────────────

def _parse_hf_leaderboard_response(
    data: Any, source_name: str = "open_llm_leaderboard"
) -> dict[str, BenchmarkScore]:
    """Parse HuggingFace Open LLM Leaderboard API response."""
    scores: dict[str, BenchmarkScore] = {}
    if not isinstance(data, dict):
        return scores
    models_data = data.get("data", data.get("models", []))
    if not isinstance(models_data, list):
        return scores
    for item in models_data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model", item.get("Model", ""))
        avg_score = item.get("average", item.get("Average", 0))
        if not model_id or not avg_score:
            continue
        key = _normalize_model_key(str(model_id))
        try:
            score_val = float(avg_score)
        except (ValueError, TypeError):
            continue
        scores[key] = BenchmarkScore(
            score=score_val,
            source=source_name,
            tier="frozen",
            evidence_grade="direct",
        )
    return scores


def _parse_arena_response(data: Any, source_name: str = "chatbot_arena") -> dict[str, BenchmarkScore]:
    """Parse Chatbot Arena ELO scores — normalize ELO to 0-100."""
    scores: dict[str, BenchmarkScore] = {}
    if not isinstance(data, dict):
        return scores
    models_data = data.get("data", data.get("models", []))
    if not isinstance(models_data, list):
        return scores
    for item in models_data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model", item.get("Model", ""))
        elo = item.get("elo", item.get("rating", item.get("Arena Elo", 0)))
        if not model_id or not elo:
            continue
        key = _normalize_model_key(str(model_id))
        try:
            elo_val = float(elo)
        except (ValueError, TypeError):
            continue
        # ELO range 800-1400 → normalize to 0-100
        normalized = max(0.0, min(100.0, (elo_val - 800) / 6.0))
        scores[key] = BenchmarkScore(
            score=round(normalized, 1),
            source=source_name,
            tier="current",
            evidence_grade="direct",
        )
    return scores


def _parse_aider_response(data: Any) -> dict[str, BenchmarkScore]:
    """
    Parse aider.chat coding leaderboard JSON.
    Expected shape: [{model: str, percent_correct: float, ...}]
    """
    scores: dict[str, BenchmarkScore] = {}
    items = data if isinstance(data, list) else data.get("models", [])
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model", item.get("name", ""))
        pct = item.get("percent_correct", item.get("score", item.get("pass_rate_2", 0)))
        if not model_id or pct is None:
            continue
        key = _normalize_model_key(str(model_id))
        try:
            score_val = float(pct)
        except (ValueError, TypeError):
            continue
        scores[key] = BenchmarkScore(
            score=round(score_val, 1),
            source="aider_coding",
            tier="frozen",
            evidence_grade="direct",
        )
    return scores


def _parse_livebench_response(data: Any) -> dict[str, BenchmarkScore]:
    """
    Parse livebench.ai JSON.
    Expected shape: {models: [{model: str, overall_average: float}]}
    """
    scores: dict[str, BenchmarkScore] = {}
    if not isinstance(data, dict):
        return scores
    models_data = data.get("models", data.get("results", []))
    for item in (models_data if isinstance(models_data, list) else []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("model", item.get("name", ""))
        avg = item.get("overall_average", item.get("average", item.get("score", 0)))
        if not model_id or avg is None:
            continue
        key = _normalize_model_key(str(model_id))
        try:
            score_val = float(avg)
        except (ValueError, TypeError):
            continue
        scores[key] = BenchmarkScore(
            score=round(score_val, 1),
            source="livebench",
            tier="frozen",
            evidence_grade="direct",
        )
    return scores


def _parse_artificial_analysis_response(data: Any) -> dict[str, BenchmarkScore]:
    """
    Parse artificialanalysis.ai API.
    Expected shape: [{model_name: str, quality_index: float, ...}]
    """
    scores: dict[str, BenchmarkScore] = {}
    items = data if isinstance(data, list) else data.get("models", [])
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_name", item.get("name", item.get("model", "")))
        quality = item.get("quality_index", item.get("quality", item.get("score", 0)))
        if not model_id or quality is None:
            continue
        key = _normalize_model_key(str(model_id))
        try:
            score_val = float(quality)
            if not (score_val == score_val) or score_val != score_val:  # NaN check
                continue
            import math as _math
            if not _math.isfinite(score_val):
                continue
        except (ValueError, TypeError):
            continue
        scores[key] = BenchmarkScore(
            score=round(score_val, 1),
            source="artificial_analysis",
            tier="current",
            evidence_grade="direct",
        )
    return scores


# Dispatcher — source name → parser function
_PARSERS = {
    "hf_leaderboard":     _parse_hf_leaderboard_response,
    "arena":              _parse_arena_response,
    "aider":              _parse_aider_response,
    "livebench":          _parse_livebench_response,
    "artificial_analysis": _parse_artificial_analysis_response,
}


# ── Score merging ─────────────────────────────────────────────────────────────

def merge_scores(
    existing: dict[str, BenchmarkScore],
    incoming: dict[str, BenchmarkScore],
) -> dict[str, BenchmarkScore]:
    """
    Merge benchmark scores. Current tier always overrides frozen tier.
    For same tier, higher score wins.
    """
    result = dict(existing)
    for model_id, new_score in incoming.items():
        if model_id not in result:
            result[model_id] = new_score
            continue
        old = result[model_id]
        if new_score.tier == "current" and old.tier == "frozen":
            result[model_id] = new_score
        elif new_score.tier == old.tier and new_score.score > old.score:
            result[model_id] = new_score
    return result


# ── Bundled fallback ──────────────────────────────────────────────────────────

def load_bundled_benchmark_fallback() -> dict[str, BenchmarkScore]:
    """Load benchmark scores from bundled model_capabilities.json. Always works offline."""
    from app.install.capability_fetcher import load_bundled_matrix

    scores: dict[str, BenchmarkScore] = {}
    try:
        matrix = load_bundled_matrix()
        for key, entry in matrix.get("models", {}).items():
            caps = entry.get("capabilities", {})
            if caps.get("embeddings"):
                continue
            reasoning = float(caps.get("reasoning_score", 0) or 0)
            json_rel = float(caps.get("json_reliable", 0) or 0)
            code_gen = float(caps.get("code_generation", 0) or 0)
            composite = (reasoning + json_rel + code_gen) / 3.0 * 100
            oid = entry.get("ollama_id", key)
            norm_key = _normalize_model_key(oid)
            scores[norm_key] = BenchmarkScore(
                score=round(composite, 1),
                source="bundled_matrix",
                tier="frozen",
                evidence_grade="self_reported",
            )
    except Exception:
        logger.debug("Failed to load bundled benchmark fallback")
    return scores


# ── Main async fetcher ────────────────────────────────────────────────────────

async def fetch_all_benchmarks(
    client: httpx.AsyncClient | None = None,
    offline: bool = False,
) -> BenchmarkResult:
    """
    Fetch benchmark scores from all 6 configured sources.

    Priority: current tier > frozen tier. Frozen scores decay with time.
    Cache on disk per source. Falls back to bundled matrix if all fail.
    """
    result = BenchmarkResult()

    if offline:
        result.scores = load_bundled_benchmark_fallback()
        result.used_fallback = True
        return result

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    assert client is not None

    try:
        for source in BENCHMARK_SOURCES:
            source_name = source["name"]
            tier = source["tier"]
            ttl = source.get("ttl_hours", 24)
            parser_key = source.get("parser", "hf_leaderboard")

            try:
                # ── Try disk cache first ──────────────────────────────────
                if _is_cache_fresh(source_name, ttl):
                    cached = _load_cache(source_name)
                    if cached and isinstance(cached.get("scores"), dict):
                        parsed: dict[str, BenchmarkScore] = {}
                        for mid, sc_data in cached["scores"].items():
                            if isinstance(sc_data, dict):
                                parsed[mid] = BenchmarkScore(
                                    score=float(sc_data.get("score", 0)),
                                    source=source_name,
                                    tier=tier,
                                    evidence_grade=sc_data.get("evidence_grade", "direct"),
                                )
                        if tier == "frozen":
                            for sc in parsed.values():
                                apply_decay(sc)
                        result.scores = merge_scores(result.scores, parsed)
                        result.sources_fetched.append(f"{source_name}(cache)")
                        continue

                # ── Live fetch ────────────────────────────────────────────
                r = await client.get(source["url"], timeout=8.0)
                if r.status_code != 200:
                    result.sources_failed.append(source_name)
                    continue

                raw_data = r.json()

                # Parse with correct parser
                parser_fn = _PARSERS.get(parser_key, _parse_hf_leaderboard_response)
                if parser_key == "arena":
                    parsed = _parse_arena_response(raw_data, source_name)
                elif parser_key == "hf_leaderboard":
                    parsed = _parse_hf_leaderboard_response(raw_data, source_name)
                else:
                    parsed = parser_fn(raw_data)  # type: ignore[arg-type]

                if tier == "frozen":
                    for sc in parsed.values():
                        apply_decay(sc)

                # Write to disk cache
                _save_cache(source_name, {
                    "fetched_at": time.time(),
                    "source": source_name,
                    "scores": {
                        mid: {"score": sc.score, "evidence_grade": sc.evidence_grade}
                        for mid, sc in parsed.items()
                    },
                })

                result.scores = merge_scores(result.scores, parsed)
                result.sources_fetched.append(source_name)

            except Exception as exc:
                logger.debug("Benchmark source %s failed: %s", source_name, exc)
                result.sources_failed.append(source_name)
                continue

    finally:
        if owns_client:
            await client.aclose()

    # ── Fallback to bundled if nothing fetched ────────────────────────────
    if not result.scores:
        result.scores = load_bundled_benchmark_fallback()
        result.used_fallback = True

    return result


def get_benchmark_for_model(
    scores: dict[str, BenchmarkScore],
    model_id: str,
) -> BenchmarkScore | None:
    """
    Look up benchmark score for a model with 3-level fuzzy matching:
      1. Exact key match
      2. Normalized key match
      3. Partial substring match → marked as 'variant' evidence
    """
    if model_id in scores:
        return scores[model_id]

    norm = _normalize_model_key(model_id)
    if norm in scores:
        return scores[norm]

    for scored_key, score in scores.items():
        if norm in scored_key or scored_key in norm:
            return BenchmarkScore(
                score=score.score * EVIDENCE_MULTIPLIERS["variant"],
                source=score.source,
                tier=score.tier,
                evidence_grade="variant",
                decay_applied=score.decay_applied,
            )

    return None
