"""Setup Fleet integration tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

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


@pytest.mark.asyncio
async def test_setup_fleet_end_to_end_no_pull():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    with patch(
        "app.setup.fleet.build_environment_report",
        new_callable=AsyncMock,
        return_value=_report(),
    ), patch(
        "app.setup.fleet.get_capability_matrix",
        new_callable=AsyncMock,
    ) as mock_matrix, patch(
        "app.setup.fleet.fetch_ollama_catalog",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "app.setup.fleet.acquire_models",
        new_callable=AsyncMock,
    ) as mock_acquire:
        from app.install.capability_fetcher import load_bundled_matrix

        mock_matrix.return_value = load_bundled_matrix()
        mock_acquire.return_value = type(
            "R", (), {"pulled": [], "failed": [], "skipped": []}
        )()

        result = await run_setup_fleet(auto_pull=False, client=client)

    assert "scout:ok" in result.state.steps
    assert "validator:ok" in result.state.steps
    assert result.plan is not None
    assert result.plan.node_assignments


@pytest.mark.asyncio
async def test_setup_fleet_pulls_when_needed():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    empty = EnvironmentReport(
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

    with patch(
        "app.setup.fleet.build_environment_report",
        new_callable=AsyncMock,
    ) as mock_env, patch(
        "app.setup.fleet.get_capability_matrix",
        new_callable=AsyncMock,
    ) as mock_matrix, patch(
        "app.setup.fleet.fetch_ollama_catalog",
        new_callable=AsyncMock,
        return_value=[{"name": "qwen2.5:3b"}],
    ), patch(
        "app.setup.fleet.acquire_models",
        new_callable=AsyncMock,
    ) as mock_acquire:
        from app.install.capability_fetcher import load_bundled_matrix

        mock_matrix.return_value = load_bundled_matrix()
        mock_env.side_effect = [empty, _report()]
        mock_acquire.return_value = type(
            "R", (), {"pulled": ["qwen2.5:3b"], "failed": [], "skipped": []}
        )()

        result = await run_setup_fleet(auto_pull=True, client=client)

    assert "acquirer:ok" in result.state.steps or "acquirer:skip" in result.state.steps
    mock_acquire.assert_called_once()
