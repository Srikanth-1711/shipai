"""
Fetch live model lists from HTTP APIs — no LLM required.

Flow (target):
  startup → fetch_all_benchmarks() runs async
          → scores cached to disk (TTL per source)
          → fetch_all_intelligence() runs:
              1. HF queries (6 types) → 500+ unique models
              2. Benchmark scores merged in (live cache FIRST)
              3. Unknown models → family interpolation
              4. Bundled matrix as final fallback
          → RichModelInfo.benchmark_score is always populated
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

logger = logging.getLogger("shipai.web_intelligence")

OLLAMA_SEARCH = "https://ollama.com/api/search"
HF_MODELS = "https://huggingface.co/api/models"

# Models in frontier_models.json released within 14 days → always included
# (they won't be in benchmark sources yet)
_FRONTIER_WINDOW_DAYS = 14


@dataclass
class RichModelInfo:
    ollama_id: str
    pull_count: int = 0
    size_gb: Optional[float] = None
    benchmark_score: float = 0.0
    benchmark_source: str = ""          # NEW: tracks where the score came from
    hf_id: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    params_b: Optional[float] = None
    released_days_ago: Optional[int] = None   # NEW: age tracking for frontier window

    @property
    def name(self) -> str:
        return self.ollama_id


def _parse_params(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", name.lower())
    return float(m.group(1)) if m else 3.0


# ── Ollama search ─────────────────────────────────────────────────────────────

async def fetch_ollama_search(
    client: httpx.AsyncClient,
    limit: int = 100,
) -> List[RichModelInfo]:
    models: List[RichModelInfo] = []
    try:
        r = await client.get(OLLAMA_SEARCH, params={"q": "", "limit": limit}, timeout=12.0)
        if r.status_code != 200:
            return models
        data = r.json()
        items = data if isinstance(data, list) else data.get("models", data.get("results", []))
        for item in items:
            name = item.get("name") or item.get("model", "")
            if not name:
                continue
            pulls = int(item.get("pullCount") or item.get("pulls") or 0)
            models.append(RichModelInfo(
                ollama_id=name,
                pull_count=pulls,
                sources=["ollama_web"],
                params_b=_parse_params(name),
            ))
    except Exception as e:
        logger.debug("Ollama search failed: %s", e)
    return models


# ── HuggingFace queries (6 types = 300+ models before dedup) ─────────────────

async def fetch_huggingface_trending(
    client: httpx.AsyncClient,
    limit: int = 100,
) -> List[RichModelInfo]:
    """Trending text-generation models on HuggingFace."""
    models: List[RichModelInfo] = []
    try:
        r = await client.get(
            HF_MODELS,
            params={"sort": "trending", "limit": limit, "filter": "text-generation"},
            timeout=12.0,
        )
        if r.status_code != 200:
            return models
        for item in r.json():
            mid = item.get("modelId") or item.get("id", "")
            if not mid:
                continue
            downloads = int(item.get("downloads") or 0)
            short = mid.split("/")[-1] if "/" in mid else mid
            models.append(RichModelInfo(
                ollama_id=short, hf_id=mid, pull_count=downloads,
                sources=["hf_trending"], params_b=_parse_params(short),
            ))
    except Exception as e:
        logger.debug("HF trending failed: %s", e)
    return models


async def fetch_huggingface_gguf_by_downloads(
    client: httpx.AsyncClient,
    limit: int = 100,
) -> List[RichModelInfo]:
    """GGUF models sorted by downloads — core of local model discovery."""
    models: List[RichModelInfo] = []
    try:
        r = await client.get(
            HF_MODELS,
            params={"search": "GGUF", "sort": "downloads", "limit": limit,
                    "filter": "text-generation"},
            timeout=12.0,
        )
        if r.status_code != 200:
            return models
        for item in r.json():
            mid = item.get("modelId") or item.get("id", "")
            if not mid:
                continue
            downloads = int(item.get("downloads") or 0)
            short = mid.split("/")[-1] if "/" in mid else mid
            models.append(RichModelInfo(
                ollama_id=short, hf_id=mid, pull_count=downloads,
                sources=["hf_gguf_downloads"], params_b=_parse_params(short),
            ))
    except Exception as e:
        logger.debug("HF GGUF downloads search failed: %s", e)
    return models


async def fetch_huggingface_gguf_recent(
    client: httpx.AsyncClient,
    limit: int = 100,
) -> List[RichModelInfo]:
    """Recently modified GGUF repos — catches newly quantized models."""
    models: List[RichModelInfo] = []
    try:
        r = await client.get(
            HF_MODELS,
            params={"search": "GGUF", "sort": "lastModified", "limit": limit,
                    "filter": "text-generation"},
            timeout=12.0,
        )
        if r.status_code != 200:
            return models
        for item in r.json():
            mid = item.get("modelId") or item.get("id", "")
            if not mid:
                continue
            downloads = int(item.get("downloads") or 0)
            short = mid.split("/")[-1] if "/" in mid else mid
            # Parse age for frontier window
            last_mod = item.get("lastModified", "")
            days_ago = _days_since(last_mod)
            models.append(RichModelInfo(
                ollama_id=short, hf_id=mid, pull_count=downloads,
                sources=["hf_gguf_recent"], params_b=_parse_params(short),
                released_days_ago=days_ago,
            ))
    except Exception as e:
        logger.debug("HF GGUF recent failed: %s", e)
    return models


async def fetch_huggingface_top_liked(
    client: httpx.AsyncClient,
    limit: int = 100,
) -> List[RichModelInfo]:
    """Top-liked text-generation models — quality signal independent of downloads."""
    models: List[RichModelInfo] = []
    try:
        r = await client.get(
            HF_MODELS,
            params={"sort": "likes", "limit": limit, "filter": "text-generation"},
            timeout=12.0,
        )
        if r.status_code != 200:
            return models
        for item in r.json():
            mid = item.get("modelId") or item.get("id", "")
            if not mid:
                continue
            likes = int(item.get("likes") or 0)
            downloads = int(item.get("downloads") or 0)
            short = mid.split("/")[-1] if "/" in mid else mid
            models.append(RichModelInfo(
                ollama_id=short, hf_id=mid,
                pull_count=max(downloads, likes),   # use max as popularity signal
                sources=["hf_top_liked"], params_b=_parse_params(short),
            ))
    except Exception as e:
        logger.debug("HF top liked failed: %s", e)
    return models


async def fetch_huggingface_trending_7d(
    client: httpx.AsyncClient,
    limit: int = 50,
) -> List[RichModelInfo]:
    """
    Text-generation models sorted by downloads in last 7 days.
    Catches viral models that aren't yet in benchmark databases.
    """
    models: List[RichModelInfo] = []
    try:
        r = await client.get(
            HF_MODELS,
            params={
                "sort": "downloads7Days", "limit": limit,
                "filter": "text-generation",
            },
            timeout=12.0,
        )
        if r.status_code != 200:
            return models
        for item in r.json():
            mid = item.get("modelId") or item.get("id", "")
            if not mid:
                continue
            downloads = int(item.get("downloads") or 0)
            short = mid.split("/")[-1] if "/" in mid else mid
            models.append(RichModelInfo(
                ollama_id=short, hf_id=mid, pull_count=downloads,
                sources=["hf_trending_7d"], params_b=_parse_params(short),
            ))
    except Exception as e:
        logger.debug("HF 7d trending failed: %s", e)
    return models


def _days_since(iso_date_str: str) -> Optional[int]:
    """Parse ISO 8601 date and return days since today. None on parse failure."""
    if not iso_date_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
        now = datetime.now(tz=timezone.utc)
        return (now - dt).days
    except (ValueError, TypeError):
        return None


# ── Capability matrix as seed ─────────────────────────────────────────────────

def models_from_capability_matrix(
    matrix: dict[str, Any],
    include_old: bool = True,
) -> List[RichModelInfo]:
    """
    Load models from bundled capability matrix.

    If include_old=False: only include entries with released_date within 14 days.
    This implements the "14-day frontier window" — curated list only for new models;
    auto-discovery takes over after 14 days.
    """
    out: List[RichModelInfo] = []
    for key, entry in matrix.get("models", {}).items():
        oid = entry.get("ollama_id", key)
        caps = entry.get("capabilities", {})
        bench = float((caps.get("reasoning_score") or caps.get("json_reliable") or 0)) * 100

        # 14-day frontier window check
        released_str = entry.get("released_date", "")
        days_ago = _days_since(released_str)

        if not include_old and days_ago is not None and days_ago > _FRONTIER_WINDOW_DAYS:
            # Old enough that auto-discovery will pick it up — skip direct inclusion
            # but still use its score data for benchmark filling
            out.append(RichModelInfo(
                ollama_id=oid,
                size_gb=entry.get("size_gb"),
                benchmark_score=bench,
                benchmark_source="capability_matrix",
                sources=["capability_matrix_score_only"],
                params_b=_parse_params(oid),
                released_days_ago=days_ago,
            ))
            continue

        out.append(RichModelInfo(
            ollama_id=oid,
            size_gb=entry.get("size_gb"),
            benchmark_score=bench,
            benchmark_source="capability_matrix",
            sources=["capability_matrix"],
            params_b=_parse_params(oid),
            released_days_ago=days_ago,
        ))
    return out


# ── Merge helpers ─────────────────────────────────────────────────────────────

def merge_model_sources(*groups: List[RichModelInfo]) -> List[RichModelInfo]:
    """Merge model lists from multiple sources. Best score + all sources kept."""
    merged: dict[str, RichModelInfo] = {}
    for group in groups:
        for m in group:
            key = m.ollama_id.lower().split(":")[0]
            if key not in merged:
                merged[key] = m
                continue
            ex = merged[key]
            ex.pull_count = max(ex.pull_count, m.pull_count)
            if m.benchmark_score > ex.benchmark_score:
                ex.benchmark_score = m.benchmark_score
                ex.benchmark_source = m.benchmark_source
            if m.size_gb:
                ex.size_gb = m.size_gb
            if m.params_b and m.params_b > 0:
                ex.params_b = m.params_b
            for s in m.sources:
                if s not in ex.sources:
                    ex.sources.append(s)
    return list(merged.values())


def merge_benchmark_scores_into_models(
    benchmark_result,  # BenchmarkResult from benchmark_fetcher
    models: List[RichModelInfo],
) -> List[RichModelInfo]:
    """
    PRIMARY benchmark wiring.

    For each model:
      1. Look up score in live benchmark cache (highest trust)
      2. If found: overwrite model.benchmark_score (live > static matrix)
      3. If not found + score == 0: apply family interpolation
      4. Always set benchmark_source for traceability

    This is the function that makes benchmark_fetcher.py the PRIMARY source.
    """
    from app.install.benchmark_fetcher import get_benchmark_for_model
    from app.install.model_family_scorer import interpolate_score

    for model in models:
        params = model.params_b or _parse_params(model.ollama_id)

        # Step 1: Look up in live benchmark cache
        live_score = get_benchmark_for_model(benchmark_result.scores, model.ollama_id)

        if live_score is not None and live_score.score > 0:
            # Live benchmark source wins — overwrite static matrix score
            model.benchmark_score = round(live_score.score, 1)
            model.benchmark_source = f"{live_score.source}({live_score.evidence_grade})"
        elif model.benchmark_score == 0.0:
            # Step 3: No live data AND no static score → interpolate from family
            interpolated = interpolate_score(model.ollama_id, params)
            model.benchmark_score = interpolated
            model.benchmark_source = "family_interpolation"

    return models


# ── Main entry point ──────────────────────────────────────────────────────────

async def fetch_all_intelligence(
    matrix: dict[str, Any],
    client: httpx.AsyncClient | None = None,
    offline: bool = False,
) -> List[RichModelInfo]:
    """
    Full model discovery pipeline.

    Target flow:
      1. fetch_all_benchmarks() → live scores (6 sources, cached)
      2. 6 HF query types + Ollama search → 500+ unique model IDs
      3. merge_benchmark_scores_into_models() → live score fills in
      4. Unknown models → family interpolation (never score=0)
      5. Bundled matrix always included as baseline seed

    Returns deduplicated list of RichModelInfo with populated benchmark_score.
    """
    from app.install.benchmark_fetcher import fetch_all_benchmarks

    # Bundled matrix is always the seed (fast, offline-safe)
    bundled = models_from_capability_matrix(matrix)

    if offline:
        # Offline: use bundled matrix + bundled fallback scores
        from app.install.benchmark_fetcher import BenchmarkResult, load_bundled_benchmark_fallback
        fallback_result = BenchmarkResult()
        fallback_result.scores = load_bundled_benchmark_fallback()
        fallback_result.used_fallback = True
        return merge_benchmark_scores_into_models(fallback_result, bundled)

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)
    assert client is not None

    try:
        # ── Run all fetches concurrently ─────────────────────────────────────
        benchmark_result, ollama_r, hf_trending_r, hf_gguf_dl_r, \
            hf_gguf_recent_r, hf_liked_r, hf_7d_r = await asyncio.gather(
            fetch_all_benchmarks(client=client),
            fetch_ollama_search(client),
            fetch_huggingface_trending(client),
            fetch_huggingface_gguf_by_downloads(client),
            fetch_huggingface_gguf_recent(client),
            fetch_huggingface_top_liked(client),
            fetch_huggingface_trending_7d(client),
            return_exceptions=True,
        )

        # Safely unpack (exceptions → empty list / empty result)
        from app.install.benchmark_fetcher import BenchmarkResult
        bm = benchmark_result if not isinstance(benchmark_result, Exception) else BenchmarkResult()
        ollama_l = ollama_r if isinstance(ollama_r, list) else []
        hf_l = hf_trending_r if isinstance(hf_trending_r, list) else []
        gguf_dl_l = hf_gguf_dl_r if isinstance(hf_gguf_dl_r, list) else []
        gguf_rec_l = hf_gguf_recent_r if isinstance(hf_gguf_recent_r, list) else []
        liked_l = hf_liked_r if isinstance(hf_liked_r, list) else []
        td7_l = hf_7d_r if isinstance(hf_7d_r, list) else []

        logger.info(
            "Intelligence fetched: ollama=%d, hf_trending=%d, gguf_dl=%d, "
            "gguf_recent=%d, liked=%d, 7d=%d | benchmark_sources=%s",
            len(ollama_l), len(hf_l), len(gguf_dl_l),
            len(gguf_rec_l), len(liked_l), len(td7_l),
            bm.sources_fetched,
        )

        # ── Merge all model sources ──────────────────────────────────────────
        all_models = merge_model_sources(
            bundled, ollama_l, hf_l, gguf_dl_l, gguf_rec_l, liked_l, td7_l
        )

        # ── Wire in benchmark scores (PRIMARY signal) ────────────────────────
        return merge_benchmark_scores_into_models(bm, all_models)

    finally:
        if owns:
            await client.aclose()
