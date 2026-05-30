"""Setup Fleet integration tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.install.model_plan import ModelPlan, NodeAssignment
from app.install.types import DiscoveredModel, EnvironmentReport, HardwareProfile, LLMRuntime, ModelInfo
from app.setup.fleet import run_setup_fleet


def _hw():
    return HardwareProfile(
        os_name="Linux",
        os_version="1",
        architecture="x86_64",
        cpu_name="cpu",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        disk_free_gb=50.0,
        disk_total_gb=100.0,
        has_gpu=True,
        gpu_vram_total_gb=4.0,
        gpu_vram_free_gb=3.5,
        has_cuda=True,
    )


def _report():
    return EnvironmentReport(
        hardware=_hw(),
        runtimes=[
            LLMRuntime(
                name="ollama",
                base_url="http://localhost:11434",
                is_running=True,
                available_models=[ModelInfo(name="qwen3:4b")],
                api_style="ollama",
            )
        ],
        inventory=[
            DiscoveredModel(
                name="qwen3:4b",
                source="ollama",
                is_served=True,
                serving_runtime="ollama",
            )
        ],
        conflicts=[],
    )


def _plan_installed():
    return ModelPlan(
        primary_runtime="ollama",
        runtime_base_url="http://localhost:11434",
        final_three={"fast": "smollm2:1.7b", "medium": "qwen3:4b", "heavy": "qwen3:4b"},
        node_assignments={
            "interview_node": NodeAssignment(
                model="qwen3:4b",
                runtime="ollama",
                status="installed",
                role="medium",
            ),
            "research_node": NodeAssignment(
                model="qwen3:4b",
                runtime="ollama",
                status="installed",
                role="heavy",
            ),
            "plan_node": NodeAssignment(
                model="qwen3:4b",
                runtime="ollama",
                status="installed",
                role="heavy",
            ),
            "explain_node": NodeAssignment(
                model="smollm2:1.7b",
                runtime="ollama",
                status="installed",
                role="fast",
            ),
        },
        models_to_download=[],
    )


def _plan_needs_pull():
    p = _plan_installed()
    p.models_to_download = ["qwen2.5:3b"]
    p.final_three["fast"] = "qwen2.5:3b"
    p.node_assignments["explain_node"] = NodeAssignment(
        model="qwen2.5:3b",
        runtime="ollama",
        status="needs_download",
        role="fast",
    )
    return p


@pytest.mark.asyncio
async def test_setup_fleet_end_to_end_no_pull():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    with patch(
        "app.setup.fleet.run_fusion_intelligence",
        new_callable=AsyncMock,
        return_value=_plan_installed(),
    ), patch(
        "app.setup.fleet.build_environment_report",
        new_callable=AsyncMock,
        return_value=_report(),
    ), patch(
        "app.setup.fleet.acquire_models",
        new_callable=AsyncMock,
    ) as mock_acquire:
        mock_acquire.return_value = type(
            "R", (), {"pulled": [], "failed": [], "skipped": []}
        )()

        result = await run_setup_fleet(auto_pull=False, client=client)

    assert "fusion:ok" in result.state.steps
    assert "validator:ok" in result.state.steps
    assert result.plan is not None
    assert result.plan.node_assignments
    mock_acquire.assert_not_called()


@pytest.mark.asyncio
async def test_setup_fleet_pulls_when_needed():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    with patch(
        "app.setup.fleet.run_fusion_intelligence",
        new_callable=AsyncMock,
        side_effect=[_plan_needs_pull(), _plan_installed()],
    ), patch(
        "app.setup.fleet.build_environment_report",
        new_callable=AsyncMock,
        return_value=_report(),
    ), patch(
        "app.setup.fleet.acquire_models",
        new_callable=AsyncMock,
    ) as mock_acquire:
        mock_acquire.return_value = type(
            "R", (), {"pulled": ["qwen2.5:3b"], "failed": [], "skipped": []}
        )()

        result = await run_setup_fleet(auto_pull=True, client=client)

    assert "acquirer:ok" in result.state.steps
    mock_acquire.assert_called_once()
    assert result.state.success
