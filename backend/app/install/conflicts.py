"""Detect port, runtime, and resource conflicts."""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from app.install.types import HardwareProfile, LLMRuntime

SHIPAI_DEFAULT_PORT = 8000
_VLLM_PORTS = {8000, 8001}


def detect_conflicts(
    runtimes: List[LLMRuntime],
    hardware: HardwareProfile,
) -> List[str]:
    conflicts: List[str] = []

    running = [r for r in runtimes if r.is_running]
    urls_seen: dict[str, str] = {}

    for rt in running:
        if not rt.base_url:
            continue
        normalized = rt.base_url.rstrip("/")
        if normalized in urls_seen and urls_seen[normalized] != rt.name:
            conflicts.append(
                f"Multiple runtimes on {normalized}: "
                f"{urls_seen[normalized]} and {rt.name}"
            )
        else:
            urls_seen[normalized] = rt.name

        parsed = urlparse(rt.base_url)
        port = parsed.port
        if port == SHIPAI_DEFAULT_PORT and rt.name in ("vllm", "llamacpp"):
            conflicts.append(
                f"{rt.name} uses port {port}, same as ShipAI API default — "
                "change vLLM/llama.cpp port or SHIPAI_PORT"
            )

    vllm_running = [r for r in running if r.name == "vllm"]
    if len(vllm_running) > 1:
        conflicts.append("Multiple vLLM endpoints detected")

    ollama_hosts = {
        r.base_url.rstrip("/")
        for r in running
        if r.name in ("ollama", "ollama_custom") and r.base_url
    }
    if len(ollama_hosts) > 1:
        conflicts.append(f"Multiple Ollama endpoints: {', '.join(sorted(ollama_hosts))}")

    if hardware.ram_available_gb < 4.0 and len(running) > 1:
        conflicts.append(
            f"Low available RAM ({hardware.ram_available_gb}GB) with "
            f"{len(running)} active runtimes — risk of OOM"
        )

    if hardware.has_gpu and hardware.gpu_vram_free_gb < 0.5:
        conflicts.append(
            f"Very low free VRAM ({hardware.gpu_vram_free_gb}GB) — "
            "large models may fail to load"
        )

    return conflicts
