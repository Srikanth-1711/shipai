"""
ShipAI — Hardware Checker (Legacy Compatibility Shim)

This module wraps the modern ``hardware_profile`` detector so that existing
routes and services continue to work without modification.  New code should
import from ``app.install.hardware_profile`` directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.install.hardware_profile import detect_hardware_profile

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int
    driver_version: str
    cuda_version: str


@dataclass
class HardwareInfo:
    # System
    os_name: str
    os_version: str
    architecture: str
    # CPU
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_freq_mhz: float
    # Memory
    ram_total_gb: float
    ram_available_gb: float
    ram_used_percent: float
    # GPU
    has_gpu: bool
    gpu: Optional[GPUInfo]
    # Disk
    disk_total_gb: float
    disk_free_gb: float
    # Computed
    hardware_tier: str
    recommended_models: list[str]
    available_features: list[str]
    tier_label: str


# ── Tier classification (derived from the modern HardwareProfile) ────────────

_TIER_MAP = {
    "hyperscale":  ("ultra",    "Ultra — Full capabilities"),
    "server":      ("ultra",    "Ultra — Full capabilities"),
    "multi_gpu":   ("high",     "High — Advanced features"),
    "workstation": ("high",     "High — Advanced features"),
    "laptop":      ("standard", "Standard — Core features"),
    "cpu_cluster": ("standard", "Standard — Core features"),
}

_FEATURES = {
    "ultra":    ["all_rag_types", "multi_agent", "fine_tuning", "cv", "load_balancing", "auto_scaling"],
    "high":     ["all_rag_types", "multi_agent", "fine_tuning_lora", "agents"],
    "standard": ["basic_rag", "agentic_rag", "single_agent", "data_analysis"],
    "basic":    ["basic_rag", "simple_chat", "data_analysis"],
    "minimal":  ["simple_chat", "basic_analysis"],
}


def _determine_tier(profile) -> tuple[str, str]:
    """Map modern scale_tier to legacy tier/label pair."""
    scale = getattr(profile, "scale_tier", "laptop")
    return _TIER_MAP.get(scale, ("standard", "Standard — Core features"))


def check_hardware() -> HardwareInfo:
    """
    Build a HardwareInfo using the modern multi-vendor hardware detector.

    This is the only function external code calls.    Internally it delegates
    everything to ``detect_hardware_profile()`` and maps the result to the
    legacy ``HardwareInfo`` dataclass.
    """
    profile = detect_hardware_profile()
    tier, label = _determine_tier(profile)

    gpu: Optional[GPUInfo] = None
    if profile.has_gpu and profile.gpu_name:
        gpu = GPUInfo(
            name=profile.gpu_name,
            vram_total_mb=int(profile.gpu_vram_total_gb * 1024),
            vram_free_mb=int(profile.gpu_vram_free_gb * 1024),
            driver_version="",
            cuda_version="",
        )

    try:
        import psutil
        freq = psutil.cpu_freq()
        cpu_freq = freq.current if freq else 0.0
        ram_used = psutil.virtual_memory().percent
    except Exception:
        cpu_freq = 0.0
        ram_used = 0.0

    return HardwareInfo(
        os_name=profile.os_name,
        os_version=profile.os_version,
        architecture=profile.architecture,
        cpu_name=profile.cpu_name,
        cpu_cores_physical=profile.cpu_cores_physical,
        cpu_cores_logical=profile.cpu_cores_logical,
        cpu_freq_mhz=cpu_freq,
        ram_total_gb=profile.ram_total_gb,
        ram_available_gb=profile.ram_available_gb,
        ram_used_percent=ram_used,
        has_gpu=profile.has_gpu,
        gpu=gpu,
        disk_total_gb=profile.disk_total_gb,
        disk_free_gb=profile.disk_free_gb,
        hardware_tier=tier,
        recommended_models=[],  # modern pipeline generates models dynamically
        available_features=_FEATURES.get(tier, _FEATURES["standard"]),
        tier_label=label,
    )
