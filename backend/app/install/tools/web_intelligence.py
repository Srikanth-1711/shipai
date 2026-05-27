"""Fetch live model lists from HTTP APIs — no LLM."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

import httpx

logger = logging.getLogger("shipai.web_intelligence")

OLLAMA_SEARCH = "https://ollama.com/api/search"
HF_MODELS = "https://huggingface.co/api/models"


@dataclass
class RichModelInfo:
    ollama_id: str
    pull_count: int = 0
    size_gb: Optional[float] = None
    benchmark_score: float = 0.0
    hf_id: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    params_b: Optional[float] = None

    @property
    def name(self) -> str:
        return self.ollama_id


def _parse_params(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", name.lower())
    return float(m.group(1)) if m else 3.0


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
            models.append(
                RichModelInfo(
                    ollama_id=name,
                    pull_count=pulls,
                    sources=["ollama_web"],
                    params_b=_parse_params(name),
                )
            )
    except Exception as e:
        logger.debug("Ollama search failed: %s", e)
    return models


async def fetch_huggingface_trending(
    client: httpx.AsyncClient,
    limit: int = 50,
) -> List[RichModelInfo]:
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
            models.append(
                RichModelInfo(
                    ollama_id=short,
                    hf_id=mid,
                    pull_count=downloads,
                    sources=["huggingface"],
                    params_b=_parse_params(short),
                )
            )
    except Exception as e:
        logger.debug("HF trending failed: %s", e)
    return models


def models_from_capability_matrix(matrix: dict[str, Any]) -> List[RichModelInfo]:
    out: List[RichModelInfo] = []
    for key, entry in matrix.get("models", {}).items():
        oid = entry.get("ollama_id", key)
        caps = entry.get("capabilities", {})
        bench = float(caps.get("reasoning_score", 0) or caps.get("json_reliable", 0)) * 100
        out.append(
            RichModelInfo(
                ollama_id=oid,
                size_gb=entry.get("size_gb"),
                benchmark_score=bench,
                sources=["capability_matrix"],
                params_b=_parse_params(oid),
            )
        )
    return out


def merge_model_sources(*groups: List[RichModelInfo]) -> List[RichModelInfo]:
    merged: dict[str, RichModelInfo] = {}
    for group in groups:
        for m in group:
            key = m.ollama_id.lower().split(":")[0]
            if key not in merged:
                merged[key] = m
                continue
            ex = merged[key]
            ex.pull_count = max(ex.pull_count, m.pull_count)
            ex.benchmark_score = max(ex.benchmark_score, m.benchmark_score)
            if m.size_gb:
                ex.size_gb = m.size_gb
            for s in m.sources:
                if s not in ex.sources:
                    ex.sources.append(s)
    return list(merged.values())


async def fetch_all_intelligence(
    matrix: dict[str, Any],
    client: httpx.AsyncClient | None = None,
    offline: bool = False,
) -> List[RichModelInfo]:
    bundled = models_from_capability_matrix(matrix)
    if offline:
        return bundled

    owns = client is None
    if owns:
        client = httpx.AsyncClient(timeout=12.0, follow_redirects=True)
    assert client is not None
    try:
        ollama, hf = await asyncio.gather(
            fetch_ollama_search(client),
            fetch_huggingface_trending(client),
            return_exceptions=True,
        )
        ollama_l = ollama if isinstance(ollama, list) else []
        hf_l = hf if isinstance(hf, list) else []
        return merge_model_sources(bundled, ollama_l, hf_l)
    finally:
        if owns:
            await client.aclose()
