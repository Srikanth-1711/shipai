"""Shared datatypes for the ShipAI bootstrap (install) layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelInfo:
    name: str
    size_gb: Optional[float] = None
    quantization: Optional[str] = None
    family: Optional[str] = None


@dataclass
class LLMRuntime:
    name: str
    base_url: Optional[str] = None
    is_running: bool = False
    available_models: List[ModelInfo] = field(default_factory=list)
    api_style: str = "openai_compatible"
    api_configured: bool = False


@dataclass
class HardwareProfile:
    """Measured machine facts only — no recommended model names."""

    os_name: str
    os_version: str
    architecture: str
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    disk_total_gb: float
    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_vram_total_gb: float = 0.0
    gpu_vram_free_gb: float = 0.0
    has_cuda: bool = False
    is_apple_silicon: bool = False
    unified_memory_gb: Optional[float] = None
    effective_vram_gb: float = 0.0
    effective_ram_gb: float = 0.0
    max_model_params_b: float = 0.0


@dataclass
class DiscoveredModel:
    """A model artifact found on disk or served by a runtime."""

    name: str
    source: str  # primary: ollama | ollama_disk | huggingface_cache | gguf_file | vllm | ...
    canonical_id: str = ""
    sources: List[str] = field(default_factory=list)
    path: Optional[str] = None
    size_gb: Optional[float] = None
    runtime_name: Optional[str] = None
    serving_runtime: Optional[str] = None
    is_served: bool = False
    needs_runtime: bool = False
    on_disk_only: bool = False
    quantization: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.canonical_id:
            from app.install.canonical import normalize_canonical_id

            object.__setattr__(self, "canonical_id", normalize_canonical_id(self.name))
        if not self.sources:
            object.__setattr__(self, "sources", [self.source])


@dataclass
class EnvironmentReport:
    hardware: HardwareProfile
    runtimes: List[LLMRuntime]
    inventory: List[DiscoveredModel]
    conflicts: List[str]
