"""Probe all local and cloud LLM runtimes — no model name hardcoding."""
from __future__ import annotations

import os
from typing import Any, List

import httpx

from app.install.types import LLMRuntime, ModelInfo

# ShipAI API default — used to flag port collisions in conflicts.py
SHIPAI_DEFAULT_PORT = 8000


def _ollama_base_urls() -> List[str]:
    raw = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
    if not raw.startswith("http"):
        raw = f"http://{raw}"
    return [raw.rstrip("/")]


def _build_probe_targets() -> List[dict[str, Any]]:
    ollama_urls = _ollama_base_urls()
    default_ollama = "http://localhost:11434"

    targets: List[dict[str, Any]] = [
        {
            "name": "ollama",
            "urls": [default_ollama],
            "probe_path": "/api/tags",
            "api_style": "ollama",
            "response_style": "ollama",
        },
        {
            "name": "vllm",
            "urls": ["http://localhost:8000", "http://localhost:8001"],
            "probe_path": "/v1/models",
            "api_style": "openai_compatible",
            "response_style": "openai",
        },
        {
            "name": "lmstudio",
            "urls": ["http://localhost:1234"],
            "probe_path": "/v1/models",
            "api_style": "openai_compatible",
            "response_style": "openai",
        },
        {
            "name": "llamacpp",
            "urls": ["http://localhost:8080"],
            "probe_path": "/v1/models",
            "api_style": "openai_compatible",
            "response_style": "openai",
        },
        {
            "name": "openai",
            "env_var": "OPENAI_API_KEY",
            "api_style": "openai_compatible",
            "models_url": "https://api.openai.com/v1/models",
        },
        {
            "name": "anthropic",
            "env_var": "ANTHROPIC_API_KEY",
            "api_style": "anthropic",
        },
        {
            "name": "groq",
            "env_var": "GROQ_API_KEY",
            "api_style": "openai_compatible",
            "models_url": "https://api.groq.com/openai/v1/models",
        },
    ]

    custom = ollama_urls[0]
    if custom.rstrip("/") != default_ollama.rstrip("/"):
        targets.append(
            {
                "name": "ollama_custom",
                "urls": ollama_urls,
                "probe_path": "/api/tags",
                "api_style": "ollama",
                "response_style": "ollama",
            }
        )
    return targets


PROBE_TARGETS = _build_probe_targets()


def extract_models(data: dict, response_style: str) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    if response_style == "ollama":
        for item in data.get("models", []):
            name = item.get("name", "")
            if not name:
                continue
            size = round(item["size"] / (1024**3), 2) if item.get("size") else None
            details = item.get("details") or {}
            models.append(
                ModelInfo(
                    name=name,
                    size_gb=size,
                    quantization=details.get("quantization_level"),
                    family=details.get("family"),
                )
            )
    elif response_style == "openai":
        for item in data.get("data", []):
            name = item.get("id", "")
            if name:
                models.append(ModelInfo(name=name))
    return models


async def _fetch_cloud_models(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
) -> List[ModelInfo]:
    try:
        r = await client.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if r.status_code == 200:
            return extract_models(r.json(), "openai")
    except (httpx.HTTPError, ValueError):
        pass
    return []


def _dedupe_runtimes(runtimes: List[LLMRuntime]) -> List[LLMRuntime]:
    seen: set[tuple[str, str | None]] = set()
    out: List[LLMRuntime] = []
    for rt in runtimes:
        key = (rt.name, (rt.base_url or "").rstrip("/") if rt.base_url else None)
        if rt.name in ("ollama", "ollama_custom") and rt.base_url:
            canon = rt.base_url.rstrip("/")
            if ("ollama", canon) in seen or ("ollama_custom", canon) in seen:
                continue
            seen.add(("ollama", canon))
            seen.add(("ollama_custom", canon))
            if rt.name == "ollama_custom":
                rt = LLMRuntime(
                    name="ollama",
                    base_url=rt.base_url,
                    is_running=rt.is_running,
                    available_models=rt.available_models,
                    api_style=rt.api_style,
                    api_configured=rt.api_configured,
                )
        elif key in seen:
            continue
        else:
            seen.add(key)
        out.append(rt)
    return out


async def detect_all_runtimes(
    client: httpx.AsyncClient | None = None,
) -> List[LLMRuntime]:
    """
    Detect every configured LLM runtime.
    Pass an optional httpx client for tests (mock transport).
    """
    runtimes: List[LLMRuntime] = []
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=2.0)

    assert client is not None

    try:
        for target in PROBE_TARGETS:
            if "env_var" in target:
                env_var = target["env_var"]
                api_key = os.getenv(env_var)
                if not api_key:
                    continue
                models: List[ModelInfo] = []
                models_url = target.get("models_url")
                if models_url:
                    models = await _fetch_cloud_models(client, models_url, api_key)
                runtimes.append(
                    LLMRuntime(
                        name=target["name"],
                        is_running=True,
                        available_models=models,
                        api_style=target["api_style"],
                        api_configured=True,
                    )
                )
                continue

            for url in target.get("urls", []):
                try:
                    probe_path = target["probe_path"]
                    r = await client.get(url + probe_path)
                    if r.status_code != 200:
                        continue
                    models = extract_models(
                        r.json(),
                        target["response_style"],
                    )
                    runtimes.append(
                        LLMRuntime(
                            name=target["name"],
                            base_url=url,
                            is_running=True,
                            available_models=models,
                            api_style=target["api_style"],
                        )
                    )
                    break
                except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                    continue
    finally:
        if owns_client:
            await client.aclose()

    return _dedupe_runtimes(runtimes)
