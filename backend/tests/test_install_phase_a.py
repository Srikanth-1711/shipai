"""
Phase A — Environment detection tests (12 scenarios).
All HTTP/env mocked; no LLM calls; no hardcoded model names in assertions.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import httpx
import pytest

from app.install.environment import build_environment_report
from app.install.hardware_profile import detect_hardware_profile
from app.install.types import HardwareProfile, LLMRuntime


def _ollama_tags(models: list[dict]) -> dict:
    return {"models": models}


def _openai_models(ids: list[str]) -> dict:
    return {"data": [{"id": i} for i in ids]}


def make_mock_client(
    routes: Dict[str, Callable],
) -> httpx.AsyncClient:
    """routes: substring of URL -> lambda request -> Response"""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for key, factory in routes.items():
            if key in url:
                return factory(request)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _workstation_profile() -> HardwareProfile:
    return HardwareProfile(
        os_name="Windows",
        os_version="10",
        architecture="AMD64",
        cpu_name="Intel",
        cpu_cores_physical=8,
        cpu_cores_logical=16,
        ram_total_gb=16.0,
        ram_available_gb=10.0,
        disk_free_gb=200.0,
        disk_total_gb=500.0,
        has_gpu=True,
        gpu_name="NVIDIA GeForce GTX 1650",
        gpu_vram_total_gb=4.0,
        gpu_vram_free_gb=3.2,
        has_cuda=True,
    )


def _server_profile() -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_version="6.0",
        architecture="x86_64",
        cpu_name="Xeon",
        cpu_cores_physical=64,
        cpu_cores_logical=128,
        ram_total_gb=512.0,
        ram_available_gb=400.0,
        disk_free_gb=2000.0,
        disk_total_gb=4000.0,
        has_gpu=True,
        gpu_name="NVIDIA A100",
        gpu_vram_total_gb=80.0,
        gpu_vram_free_gb=78.0,
        has_cuda=True,
    )


def _cpu_only_profile() -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_version="6.0",
        architecture="x86_64",
        cpu_name="CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_total_gb=8.0,
        ram_available_gb=5.0,
        disk_free_gb=50.0,
        disk_total_gb=100.0,
        has_gpu=False,
    )


def _mac_m2_profile() -> HardwareProfile:
    return HardwareProfile(
        os_name="Darwin",
        os_version="14.0",
        architecture="arm64",
        cpu_name="Apple M2",
        cpu_cores_physical=8,
        cpu_cores_logical=8,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        disk_free_gb=100.0,
        disk_total_gb=256.0,
        has_gpu=False,
        is_apple_silicon=True,
        unified_memory_gb=16.0,
    )


@pytest.mark.asyncio
async def test_scenario_01_fresh_machine_nothing_installed(monkeypatch):
  """Fresh machine: no runtimes respond."""
  for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
    monkeypatch.delenv(var, raising=False)
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client({})
  report = await build_environment_report(client=client)
  assert not [r for r in report.runtimes if r.is_running and r.base_url]
  assert report.hardware.gpu_name and "1650" in report.hardware.gpu_name


@pytest.mark.asyncio
async def test_scenario_02_ollama_running_no_models(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200, json=_ollama_tags([])
          ),
      }
  )
  report = await build_environment_report(client=client)
  ollama = next(r for r in report.runtimes if r.name == "ollama")
  assert ollama.is_running
  assert ollama.available_models == []


@pytest.mark.asyncio
async def test_scenario_03_ollama_small_model_installed(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200,
              json=_ollama_tags(
                  [
                      {
                          "name": "tinyllama:latest",
                          "size": 637_534_208,
                          "details": {"quantization_level": "Q4_0", "family": "llama"},
                      }
                  ]
              ),
          ),
      }
  )
  report = await build_environment_report(client=client)
  served = [i for i in report.inventory if i.is_served]
  assert any("tinyllama" in i.name for i in served)


@pytest.mark.asyncio
async def test_scenario_04_ollama_good_models_already(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200,
              json=_ollama_tags(
                  [
                      {"name": "gemma4:9b", "size": 5_000_000_000},
                      {"name": "deepseek-r1:7b", "size": 4_500_000_000},
                  ]
              ),
          ),
      }
  )
  report = await build_environment_report(client=client)
  names = {i.name for i in report.inventory if i.is_served}
  assert "gemma4:9b" in names
  assert "deepseek-r1:7b" in names


@pytest.mark.asyncio
async def test_scenario_05_vllm_running(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "localhost:8000/v1/models": lambda r: httpx.Response(
              200,
              json=_openai_models(["meta-llama/Llama-3-8b-Instruct"]),
          ),
          "/api/tags": lambda r: httpx.Response(404),
      }
  )
  report = await build_environment_report(client=client)
  vllm = next((r for r in report.runtimes if r.name == "vllm"), None)
  assert vllm is not None
  assert vllm.is_running
  assert any("Llama-3" in m.name for m in vllm.available_models)


@pytest.mark.asyncio
async def test_scenario_06_lmstudio_running(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "localhost:1234/v1/models": lambda r: httpx.Response(
              200,
              json=_openai_models(["llama-3-8b-instruct"]),
          ),
      }
  )
  report = await build_environment_report(client=client)
  lm = next(r for r in report.runtimes if r.name == "lmstudio")
  assert lm.is_running
  assert lm.base_url == "http://localhost:1234"


@pytest.mark.asyncio
async def test_scenario_07_llamacpp_running(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "localhost:8080/v1/models": lambda r: httpx.Response(
              200,
              json=_openai_models(["local-gguf-model"]),
          ),
      }
  )
  report = await build_environment_report(client=client)
  ll = next(r for r in report.runtimes if r.name == "llamacpp")
  assert ll.is_running


@pytest.mark.asyncio
async def test_scenario_08_openai_key_no_local(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _cpu_only_profile,
  )
  monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("GROQ_API_KEY", raising=False)

  client = make_mock_client(
      {
          "api.openai.com/v1/models": lambda r: httpx.Response(
              200,
              json=_openai_models(["gpt-4o", "gpt-4o-mini"]),
          ),
      }
  )
  report = await build_environment_report(client=client)
  openai_rt = next(r for r in report.runtimes if r.name == "openai")
  assert openai_rt.is_running
  assert openai_rt.api_configured
  assert len(openai_rt.available_models) >= 1
  local = [r for r in report.runtimes if r.base_url]
  assert not local


@pytest.mark.asyncio
async def test_scenario_09_ollama_and_vllm_both(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200, json=_ollama_tags([{"name": "phi3:mini", "size": 2_000_000_000}])
          ),
          "localhost:8000/v1/models": lambda r: httpx.Response(
              200, json=_openai_models(["meta-llama/Llama-3-70b"])
          ),
      }
  )
  report = await build_environment_report(client=client)
  names = {r.name for r in report.runtimes if r.is_running}
  assert "ollama" in names
  assert "vllm" in names
  assert any("8000" in c or "ShipAI" in c for c in report.conflicts)
  assert len(report.inventory) >= 2


@pytest.mark.asyncio
async def test_scenario_10_unknown_custom_model_benchmark_ready(monkeypatch):
  """Unknown model id still appears in inventory from Ollama tags."""
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200,
              json=_ollama_tags(
                  [{"name": "some-custom-model:7b", "size": 4_000_000_000}]
              ),
          ),
      }
  )
  report = await build_environment_report(client=client)
  assert any("some-custom-model" in i.name for i in report.inventory)


@pytest.mark.asyncio
async def test_scenario_11_server_8xa100(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _server_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200,
              json=_ollama_tags(
                  [{"name": "llama3.1:70b", "size": 40_000_000_000}]
              ),
          ),
      }
  )
  report = await build_environment_report(client=client)
  assert report.hardware.gpu_vram_total_gb >= 40
  assert any("70b" in i.name for i in report.inventory if i.is_served)


@pytest.mark.asyncio
async def test_scenario_12_mac_m2_unified_memory(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _mac_m2_profile,
  )
  client = make_mock_client(
      {
          "/api/tags": lambda r: httpx.Response(
              200, json=_ollama_tags([{"name": "phi3:mini", "size": 2_000_000_000}])
          ),
      }
  )
  report = await build_environment_report(client=client)
  assert report.hardware.is_apple_silicon
  assert report.hardware.unified_memory_gb == 16.0
  assert not report.hardware.has_cuda


@pytest.mark.asyncio
async def test_environment_report_structure(monkeypatch):
  monkeypatch.setattr(
      "app.install.environment.detect_hardware_profile",
      _workstation_profile,
  )
  client = make_mock_client({})
  report = await build_environment_report(client=client)
  assert report.hardware.ram_total_gb > 0
  assert isinstance(report.runtimes, list)
  assert isinstance(report.inventory, list)
  assert isinstance(report.conflicts, list)


def test_detect_hardware_profile_real_machine():
  """Smoke test on host — no mocks."""
  hw = detect_hardware_profile()
  assert hw.ram_total_gb > 0
  assert hw.os_name
