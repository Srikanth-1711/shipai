"""Fusion intelligence tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.install.ceo_policy import load_ceo_policy, ceo_boost_for_model
from app.install.intent_profile import parse_use_case
from app.install.tools.feasibility_filter import is_feasible
from app.install.tools.model_scorer import rank_models
from app.install.tools.web_intelligence import RichModelInfo, merge_model_sources
from app.install.types import DiscoveredModel, EnvironmentReport, HardwareProfile, LLMRuntime, ModelInfo


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
        effective_vram_gb=3.0,
        effective_ram_gb=12.0,
        max_model_params_b=5.0,
    )


def test_parse_use_case_rag():
    intent = parse_use_case("RAG on company PDFs offline")
    assert intent.project_type == "rag"
    assert intent.require_embed is True


def test_ceo_policy_loads():
    policy = load_ceo_policy()
    assert "approved_families" in policy
    assert ceo_boost_for_model("qwen3:4b", policy) > 0
    assert ceo_boost_for_model("tinyllama", policy) < 0


def test_feasibility_small_model_fits():
    feas = is_feasible("phi3:mini", _hw(), 2.2)
    assert feas.feasible


def test_rank_produces_final_three():
    from app.install.capability_fetcher import load_bundled_matrix

    matrix = load_bundled_matrix()
    candidates = merge_model_sources(
        [
            RichModelInfo("qwen3:4b", pull_count=1000, params_b=4),
            RichModelInfo("smollm2:1.7b", pull_count=500, params_b=1.7),
            RichModelInfo("qwen2.5:3b", pull_count=800, params_b=3),
        ]
    )
    report = EnvironmentReport(
        hardware=_hw(),
        runtimes=[
            LLMRuntime(name="ollama", is_running=True, available_models=[ModelInfo(name="qwen3:4b")])
        ],
        inventory=[
            DiscoveredModel(name="qwen3:4b", source="ollama", is_served=True),
        ],
        conflicts=[],
    )
    rank = rank_models(candidates, report, matrix, parse_use_case("multi-agent"))
    assert len(rank.top_10) >= 1
    assert "fast" in rank.final_three
    assert "heavy" in rank.final_three
    assert rank.node_assignments


@pytest.mark.asyncio
async def test_fusion_pipeline_mocked():
    from app.install.fusion_intelligence import run_fusion_intelligence

    with patch(
        "app.install.fusion_intelligence.build_environment_report",
        new_callable=AsyncMock,
    ) as mock_env, patch(
        "app.install.fusion_intelligence.get_capability_matrix",
        new_callable=AsyncMock,
    ) as mock_m, patch(
        "app.install.fusion_intelligence.fetch_all_intelligence",
        new_callable=AsyncMock,
    ) as mock_fetch:
        from app.install.capability_fetcher import load_bundled_matrix

        mock_m.return_value = load_bundled_matrix()
        mock_fetch.return_value = [
            RichModelInfo("qwen3:4b", pull_count=100),
            RichModelInfo("smollm2:1.7b", pull_count=50),
        ]
        mock_env.return_value = EnvironmentReport(
            hardware=_hw(),
            runtimes=[LLMRuntime(name="ollama", is_running=True)],
            inventory=[DiscoveredModel(name="qwen3:4b", source="ollama", is_served=True)],
            conflicts=[],
        )
        plan = await run_fusion_intelligence(offline=True, use_case="chat")
        assert plan.final_three
        assert plan.node_assignments
