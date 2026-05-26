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
    HardwareProfile,
    LLMRuntime,
    ModelInfo,
)

__all__ = [
    "DiscoveredModel",
    "EnvironmentReport",
    "HardwareProfile",
    "LLMRuntime",
    "ModelInfo",
    "build_environment_report",
    "build_inventory",
    "detect_all_runtimes",
    "fetch_ollama_catalog",
    "format_environment_report",
    "get_capability_matrix",
    "get_model_entry",
    "rank_catalog_for_hardware",
    "scan_all_artifacts",
]
