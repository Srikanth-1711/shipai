"""
Phase D — Model negotiator uses genuine inventory, not fake assignments.
"""
from __future__ import annotations

import pytest

from app.install.model_negotiator import ModelNegotiator, negotiate_from_report
from app.install.types import (
    DiscoveredModel,
    EnvironmentReport,
    HardwareProfile,
    LLMRuntime,
    ModelInfo,
)


def _hw():
    return HardwareProfile(
        os_name="Windows",
        os_version="10",
        architecture="AMD64",
        cpu_name="x",
        cpu_cores_physical=6,
        cpu_cores_logical=12,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        disk_free_gb=100.0,
        disk_total_gb=200.0,
        has_gpu=True,
        gpu_name="GTX 1650",
        gpu_vram_total_gb=4.0,
        gpu_vram_free_gb=3.5,
        has_cuda=True,
    )


def _matrix():
    from app.install.capability_fetcher import load_bundled_matrix

    return load_bundled_matrix()


@pytest.fixture
def ollama_report():
    inv = [
        DiscoveredModel(
            name="qwen3:4b",
            source="ollama",
            sources=["ollama"],
            is_served=True,
            serving_runtime="ollama",
            size_gb=2.33,
        ),
        DiscoveredModel(
            name="smollm2:1.7b",
            source="ollama",
            sources=["ollama"],
            is_served=True,
            serving_runtime="ollama",
            size_gb=1.7,
        ),
        DiscoveredModel(
            name="nomic-embed-text:latest",
            source="ollama",
            sources=["ollama"],
            is_served=True,
            serving_runtime="ollama",
        ),
    ]
    return EnvironmentReport(
        hardware=_hw(),
        runtimes=[
            LLMRuntime(
                name="ollama",
                base_url="http://localhost:11434",
                is_running=True,
                available_models=[
                    ModelInfo(name="qwen3:4b"),
                    ModelInfo(name="smollm2:1.7b"),
                ],
                api_style="ollama",
            )
        ],
        inventory=inv,
        conflicts=[],
    )


def test_assigns_only_installed_models(ollama_report):
    neg = negotiate_from_report(ollama_report, _matrix())
    for node, asn in neg.node_assignments.items():
        assert asn.status == "installed", node
        assert asn.model in ("qwen3:4b", "smollm2:1.7b")
        assert asn.confidence in ("installed_verified", "installed_unknown")


def test_does_not_invent_models_not_in_inventory():
    report = EnvironmentReport(
        hardware=_hw(),
        runtimes=[
            LLMRuntime(
                name="ollama",
                base_url="http://localhost:11434",
                is_running=True,
                available_models=[],
                api_style="ollama",
            )
        ],
        inventory=[],
        conflicts=[],
    )
    neg = negotiate_from_report(report, _matrix(), catalog=[])
    for asn in neg.node_assignments.values():
        assert asn.status != "installed" or asn.model == ""


def test_embedding_picked_from_inventory(ollama_report):
    neg = negotiate_from_report(ollama_report, _matrix())
    assert neg.embedding is not None
    assert "embed" in neg.embedding.model.lower()


def test_fresh_machine_suggests_download_not_fake_installed():
    report = EnvironmentReport(
        hardware=_hw(),
        runtimes=[],
        inventory=[],
        conflicts=[],
    )
    matrix = _matrix()
    catalog = [{"name": "qwen2.5:3b", "size_gb": 1.9}]
    neg = negotiate_from_report(report, matrix, catalog=catalog)
    assert any(a.status == "needs_download" for a in neg.node_assignments.values())
    assert "qwen2.5:3b" in neg.models_to_download or any(
        "qwen" in (a.model or "") for a in neg.node_assignments.values()
    )
