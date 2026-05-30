"""ShipAI bootstrap layer — runs before any LLM is required."""
from app.install.capability_fetcher import (
    fetch_ollama_catalog,
    get_capability_matrix,
    get_model_entry,
    rank_catalog_for_hardware,
)
from app.install.environment import build_environment_report, format_environment_report
from app.install.inventory_scanner import build_inventory, scan_all_artifacts
from app.install.runtime_detector import detect_all_runtimes
from app.install.types import (
    DiscoveredModel,
    EnvironmentReport,
    GPUInfo,
    HardwareProfile,
    LLMRuntime,
    ModelInfo,
)

# New scale + acquisition exports
from app.install.gpu_topology import GPUTopology, detect_gpu_topology
from app.install.scale_classifier import ScaleProfile, classify
from app.install.runtime_recommender import RuntimeRecommendation, recommend_runtime
from app.install.deployment_generator import write_deployment_configs
from app.install.model_acquirer import AcquisitionResult, acquire_model
from app.install.model_registrar import register_gguf_with_ollama, register_hf_cache

__all__ = [
    # Types
    "DiscoveredModel",
    "EnvironmentReport",
    "GPUInfo",
    "GPUTopology",
    "HardwareProfile",
    "LLMRuntime",
    "ModelInfo",
    "ScaleProfile",
    "RuntimeRecommendation",
    "AcquisitionResult",
    # Detection
    "build_environment_report",
    "build_inventory",
    "detect_all_runtimes",
    "detect_gpu_topology",
    # Capability
    "fetch_ollama_catalog",
    "format_environment_report",
    "get_capability_matrix",
    "get_model_entry",
    "rank_catalog_for_hardware",
    "scan_all_artifacts",
    # Scale + Runtime
    "classify",
    "recommend_runtime",
    "write_deployment_configs",
    # Acquisition
    "acquire_model",
    "register_gguf_with_ollama",
    "register_hf_cache",
]
