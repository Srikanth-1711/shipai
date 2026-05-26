"""
Phase B — Deep inventory tests (disk, cross-link, readiness flags).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.install.canonical import normalize_canonical_id
from app.install.inventory_scanner import (
    build_inventory,
    consolidate_inventory,
    scan_gguf_files,
    scan_huggingface_cache,
    scan_ollama_disk,
)
from app.install.types import DiscoveredModel, LLMRuntime, ModelInfo


def test_normalize_canonical_id():
    assert normalize_canonical_id("llama3.1:8b") == "llama3.1"
    assert normalize_canonical_id("meta-llama/Llama-3-8B") == "meta-llama-llama-3-8b"


def test_hf_cache_only(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    model_dir = hub / "models--org--Test-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    items = scan_huggingface_cache()
    assert len(items) == 1
    assert items[0].name == "org/Test-7B"
    assert items[0].needs_runtime is True
    assert items[0].on_disk_only is True
    assert items[0].is_served is False


def test_gguf_only(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    gguf = models / "my-local-7b.Q4_K_M.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setenv("SHIPAI_GGUF_PATHS", str(models))

    items = scan_gguf_files(max_files=10)
    assert any("my-local-7b" in i.name for i in items)
    assert all(i.needs_runtime for i in items)


def test_ollama_disk_when_api_down(tmp_path, monkeypatch):
    ollama_root = tmp_path / "ollama" / "models"
    lib = (
        ollama_root
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "phi3"
        / "mini"
    )
    lib.mkdir(parents=True)
    (lib / "manifest.json").write_text("{}")
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama_root))

    items = scan_ollama_disk()
    assert any(i.name == "phi3:mini" for i in items)
    assert items[0].source == "ollama_disk"


def test_consolidate_links_served_and_hf(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hf_model = hub / "models--meta-llama--Llama-3-8B"
    hf_model.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    raw = [
        DiscoveredModel(
            name="llama3.1:8b",
            source="ollama",
            sources=["ollama"],
            is_served=True,
            serving_runtime="ollama",
            needs_runtime=False,
        ),
        DiscoveredModel(
            name="meta-llama/Llama-3-8B",
            source="huggingface_cache",
            sources=["huggingface_cache"],
            path=str(hf_model),
            needs_runtime=True,
            on_disk_only=True,
        ),
    ]
    merged = consolidate_inventory(raw)
    assert len(merged) == 2


@pytest.mark.asyncio
async def test_api_down_disk_inventory_still_works(tmp_path, monkeypatch):
    from app.install.environment import build_environment_report

    ollama_root = tmp_path / "ollama" / "models"
    lib = (
        ollama_root
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "gemma2"
        / "2b"
    )
    lib.mkdir(parents=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama_root))

    monkeypatch.setattr(
        "app.install.environment.detect_hardware_profile",
        lambda: __import__(
            "app.install.types", fromlist=["HardwareProfile"]
        ).HardwareProfile(
            os_name="Linux",
            os_version="1",
            architecture="x86_64",
            cpu_name="x",
            cpu_cores_physical=4,
            cpu_cores_logical=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            disk_free_gb=100.0,
            disk_total_gb=200.0,
            has_gpu=False,
        ),
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    report = await build_environment_report(client=client)
    await client.aclose()

    disk_models = [i for i in report.inventory if "gemma2" in i.name]
    assert disk_models
    assert disk_models[0].needs_runtime


def test_build_inventory_dedupes_ollama_api_and_disk():
    runtimes = [
        LLMRuntime(
            name="ollama",
            base_url="http://localhost:11434",
            is_running=True,
            available_models=[ModelInfo(name="phi3:mini", size_gb=2.0)],
            api_style="ollama",
        )
    ]
    raw_disk = [
        DiscoveredModel(
            name="phi3:mini",
            canonical_id="phi3",
            source="ollama_disk",
            sources=["ollama_disk"],
            needs_runtime=True,
            on_disk_only=True,
        )
    ]
    from app.install import inventory_scanner as scanner

    original = scanner.scan_all_artifacts
    scanner.scan_all_artifacts = lambda: raw_disk
    try:
        inv = build_inventory(runtimes)
    finally:
        scanner.scan_all_artifacts = original

    phi = [i for i in inv if "phi3" in i.name]
    assert len(phi) == 1
    assert phi[0].is_served
    assert "ollama" in phi[0].sources
