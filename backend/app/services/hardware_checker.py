"""
ShipAI — Hardware Checker
Detects system hardware and recommends optimal models/features.
This is a core differentiator — ShipAI adapts to what the user's machine can handle.
"""
import platform
import psutil
import subprocess
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Model recommendations by hardware tier
MODEL_TIERS = {
    "ultra": {
        "min_vram_gb": 40,
        "min_ram_gb": 64,
        "models": ["nemotron:70b", "llama3.1:70b", "mixtral:8x22b", "deepseek-coder:33b"],
        "label": "Ultra — Full capabilities",
        "features": ["all_rag_types", "multi_agent", "fine_tuning", "cv", "load_balancing", "auto_scaling"],
    },
    "high": {
        "min_vram_gb": 12,
        "min_ram_gb": 32,
        "models": ["llama3.1:8b", "mistral:7b", "codellama:7b", "gemma2:9b"],
        "label": "High — Advanced features",
        "features": ["all_rag_types", "multi_agent", "fine_tuning_lora", "agents"],
    },
    "standard": {
        "min_vram_gb": 4,
        "min_ram_gb": 16,
        "models": ["llama3.1:8b", "mistral:7b", "phi3:mini"],
        "label": "Standard — Core features",
        "features": ["basic_rag", "agentic_rag", "single_agent", "data_analysis"],
    },
    "basic": {
        "min_vram_gb": 2,
        "min_ram_gb": 8,
        "models": ["phi3:mini", "gemma:2b", "tinyllama"],
        "label": "Basic — Essential features",
        "features": ["basic_rag", "simple_chat", "data_analysis"],
    },
    "minimal": {
        "min_vram_gb": 0,
        "min_ram_gb": 4,
        "models": ["tinyllama", "phi3:mini"],
        "label": "Minimal — Lightweight only",
        "features": ["simple_chat", "basic_analysis"],
    },
}


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
    gpu: GPUInfo | None

    # Disk
    disk_total_gb: float
    disk_free_gb: float

    # Computed
    hardware_tier: str
    recommended_models: list[str]
    available_features: list[str]
    tier_label: str


def _detect_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                # Get CUDA version separately
                cuda_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                cuda_ver = "unknown"
                try:
                    full_output = subprocess.run(
                        ["nvidia-smi"], capture_output=True, text=True, timeout=5
                    )
                    for line in full_output.stdout.split("\n"):
                        if "CUDA Version" in line:
                            cuda_ver = line.split("CUDA Version:")[1].strip().split()[0]
                            break
                except Exception:
                    pass

                return GPUInfo(
                    name=parts[0],
                    vram_total_mb=int(float(parts[1])),
                    vram_free_mb=int(float(parts[2])),
                    driver_version=parts[3],
                    cuda_version=cuda_ver,
                )
    except FileNotFoundError:
        logger.info("nvidia-smi not found — no NVIDIA GPU detected")
    except Exception as e:
        logger.error(f"GPU detection error: {e}")
    return None


def _determine_tier(ram_gb: float, vram_gb: float) -> str:
    """Determine hardware tier based on RAM and VRAM."""
    for tier_name in ["ultra", "high", "standard", "basic", "minimal"]:
        tier = MODEL_TIERS[tier_name]
        if ram_gb >= tier["min_ram_gb"] and vram_gb >= tier["min_vram_gb"]:
            return tier_name
    return "minimal"


def check_hardware() -> HardwareInfo:
    """Run full hardware detection and return recommendations."""
    # OS
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()

    # CPU
    cpu_name = platform.processor() or "Unknown"
    cpu_cores_physical = psutil.cpu_count(logical=False) or 1
    cpu_cores_logical = psutil.cpu_count(logical=True) or 1
    cpu_freq = psutil.cpu_freq()
    cpu_freq_mhz = cpu_freq.current if cpu_freq else 0

    # Memory
    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 1)
    ram_available_gb = round(mem.available / (1024**3), 1)
    ram_used_percent = mem.percent

    # GPU
    gpu = _detect_gpu()
    has_gpu = gpu is not None
    vram_gb = (gpu.vram_total_mb / 1024) if gpu else 0

    # Disk
    disk = psutil.disk_usage("/") if os_name != "Windows" else psutil.disk_usage("C:\\")
    disk_total_gb = round(disk.total / (1024**3), 1)
    disk_free_gb = round(disk.free / (1024**3), 1)

    # Determine tier
    tier = _determine_tier(ram_total_gb, vram_gb)
    tier_info = MODEL_TIERS[tier]

    return HardwareInfo(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_cores_physical,
        cpu_cores_logical=cpu_cores_logical,
        cpu_freq_mhz=cpu_freq_mhz,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        ram_used_percent=ram_used_percent,
        has_gpu=has_gpu,
        gpu=gpu,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        hardware_tier=tier,
        recommended_models=tier_info["models"],
        available_features=tier_info["features"],
        tier_label=tier_info["label"],
    )
