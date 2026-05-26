"""
Phase C — Capability matrix fetcher tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.install.capability_fetcher import (
    CapabilityMatrixError,
    fetch_ollama_catalog,
    fetch_remote_matrix,
    get_capability_matrix,
    get_model_entry,
    load_bundled_matrix,
    load_cached_matrix,
    rank_catalog_for_hardware,
    save_cached_matrix,
    validate_matrix,
)


@pytest.fixture
def sample_matrix():
    return {
        "schema_version": "1.0",
        "models": {
            "test:1b": {
                "min_vram_gb": 0,
                "min_ram_gb": 4,
                "fits_cpu": True,
                "size_gb": 1.0,
                "capabilities": {"json_reliable": 0.9},
                "ollama_id": "test:1b",
            },
            "big:70b": {
                "min_vram_gb": 40,
                "min_ram_gb": 64,
                "fits_cpu": False,
                "size_gb": 40,
                "capabilities": {"json_reliable": 0.95},
                "ollama_id": "big:70b",
            },
        },
        "shipai_node_requirements": {
            "interview_node": {"minimum_json_reliable": 0.7}
        },
    }


def test_validate_matrix_rejects_invalid():
    with pytest.raises(CapabilityMatrixError):
        validate_matrix({"models": {}})


def test_load_bundled_matrix():
    matrix = load_bundled_matrix()
    assert matrix["schema_version"]
    assert "qwen2.5:3b" in matrix["models"]
    assert "interview_node" in matrix["shipai_node_requirements"]


def test_cache_roundtrip(tmp_path, monkeypatch, sample_matrix):
    cache_file = tmp_path / "model_capabilities.json"
    monkeypatch.setattr(
        "app.install.capability_fetcher._CACHE_FILE",
        cache_file,
    )
    monkeypatch.setattr(
        "app.install.capability_fetcher._CACHE_DIR",
        tmp_path,
    )
    save_cached_matrix(sample_matrix)
    loaded = load_cached_matrix(max_age=999999)
    assert loaded is not None
    assert loaded["models"]["test:1b"]["ollama_id"] == "test:1b"


@pytest.mark.asyncio
async def test_fetch_remote_matrix(monkeypatch, tmp_path, sample_matrix):
    monkeypatch.setattr(
        "app.install.capability_fetcher._CACHE_FILE",
        tmp_path / "cache.json",
    )
    monkeypatch.setattr(
        "app.install.capability_fetcher._CACHE_DIR",
        tmp_path,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_matrix)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    data = await fetch_remote_matrix("https://example.com/matrix.json", client=client)
    await client.aclose()
    assert data["models"]["test:1b"]


@pytest.mark.asyncio
async def test_get_capability_matrix_offline_uses_bundled(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.install.capability_fetcher._CACHE_FILE",
        tmp_path / "missing.json",
    )

    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail_handler))
    matrix = await get_capability_matrix(force_refresh=True, client=client)
    await client.aclose()
    assert "qwen2.5:3b" in matrix["models"]


def test_get_model_entry_by_ollama_id():
    matrix = load_bundled_matrix()
    entry = get_model_entry(matrix, "qwen2.5:3b")
    assert entry is not None
    assert entry.get("ollama_id") == "qwen2.5:3b"


def test_rank_catalog_for_hardware_filters_vram(sample_matrix):
    catalog = [
        {"name": "test:1b"},
        {"name": "big:70b"},
    ]
    ranked = rank_catalog_for_hardware(
        catalog,
        sample_matrix,
        effective_vram_gb=4.0,
        effective_ram_gb=8.0,
        has_cuda=True,
        top_n=10,
    )
    assert len(ranked) == 1
    assert ranked[0]["name"] == "test:1b"


@pytest.mark.asyncio
async def test_fetch_ollama_catalog_fallback_to_matrix():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    catalog = await fetch_ollama_catalog(client=client, matrix_fallback=load_bundled_matrix())
    await client.aclose()
    assert len(catalog) >= 3
    assert all("name" in c for c in catalog)
