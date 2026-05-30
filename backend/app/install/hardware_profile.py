"""Hardware detection for bootstrap layer — multi-vendor GPU + CPU features."""
from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from typing import List, Optional, Tuple

import psutil

from app.install.types import GPUInfo, HardwareProfile

logger = logging.getLogger("shipai.hardware")

# ── Memory bandwidth lookup (GB/s) — official vendor specs ───────────────────
GPU_BANDWIDTH_GB_S: dict[str, float] = {
    # NVIDIA
    "RTX 5090":   1792,
    "RTX 5080":   960,
    "RTX 5070 Ti": 896,
    "RTX 5070":   672,
    "RTX 4090":   1008,
    "RTX 4080 SUPER": 736,
    "RTX 4080":   717,
    "RTX 4070 Ti SUPER": 672,
    "RTX 4070 Ti": 504,
    "RTX 4070 SUPER": 504,
    "RTX 4070":   504,
    "RTX 4060 Ti": 288,
    "RTX 4060":   272,
    "RTX 3090 Ti": 1008,
    "RTX 3090":   936,
    "RTX 3080 Ti": 912,
    "RTX 3080":   760,
    "RTX 3070 Ti": 608,
    "RTX 3070":   448,
    "RTX 3060 Ti": 448,
    "RTX 3060":   360,
    "RTX 2080 Ti": 616,
    "RTX 2080 SUPER": 496,
    "RTX 2080":   448,
    "RTX 2070 SUPER": 448,
    "RTX 2070":   448,
    "RTX 2060 SUPER": 448,
    "RTX 2060":   336,
    "GTX 1080 Ti": 484,
    "GTX 1080":   320,
    "GTX 1070 Ti": 256,
    "GTX 1070":   256,
    "GTX 1660 Ti": 288,
    "GTX 1660 SUPER": 336,
    "GTX 1660":   192,
    "GTX 1650 SUPER": 192,
    "GTX 1650":   128,
    "GTX 1050 Ti": 112,
    # AMD
    "RX 9070 XT":  624,
    "RX 9070":    512,
    "RX 7900 XTX": 960,
    "RX 7900 XT":  800,
    "RX 7900 GRE": 576,
    "RX 7800 XT":  624,
    "RX 7700 XT":  432,
    "RX 7600":    288,
    "RX 6950 XT":  576,
    "RX 6900 XT":  512,
    "RX 6800 XT":  512,
    "RX 6800":    512,
    "RX 6700 XT":  384,
    "RX 6600 XT":  256,
    "RX 6600":    224,
    # Apple Silicon
    "M4 Ultra":   800,
    "M4 Max":     546,
    "M4 Pro":     273,
    "M4":         120,
    "M3 Ultra":   800,
    "M3 Max":     400,
    "M3 Pro":     150,
    "M3":         100,
    "M2 Ultra":   800,
    "M2 Max":     400,
    "M2 Pro":     200,
    "M2":         100,
    "M1 Ultra":   800,
    "M1 Max":     400,
    "M1 Pro":     200,
    "M1":          68,
    # Intel Arc
    "Arc A770":   512,
    "Arc A750":   512,
    "Arc A580":   512,
    "Arc A380":   179,
    # Data center (common)
    "A100":      2039,
    "A100 80GB": 2039,
    "H100":      3352,
    "L40":        864,
    "L4":         300,
    "T4":         300,
    "V100":       900,
}

# Conservative floor for unknown GPUs (~GTX 1060 level)
_DEFAULT_BANDWIDTH = 100.0


def get_bandwidth(gpu_name: str) -> float:
    """
    Fuzzy match GPU name to memory bandwidth (GB/s).

    Never returns 0 — uses conservative estimate if unknown.
    Matching is case-insensitive and checks if any known key
    appears as a substring of the GPU name.
    """
    if not gpu_name:
        return _DEFAULT_BANDWIDTH
    name_upper = gpu_name.upper()
    # Try longest keys first for specificity (e.g. "RTX 4080 SUPER" before "RTX 4080")
    sorted_keys = sorted(GPU_BANDWIDTH_GB_S.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key.upper() in name_upper:
            return float(GPU_BANDWIDTH_GB_S[key])
    return _DEFAULT_BANDWIDTH


# ── NVIDIA detection (Binding first, Text fallback) ──────────────────────────

def _detect_nvidia_pynvml() -> tuple[str | None, float, float, bool]:
    """Detect NVIDIA GPU via native pynvml binding. Returns (name, vram_total_gb, vram_free_gb, has_cuda)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return None, 0.0, 0.0, False

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
            
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = mem_info.total / 1e9
        free_gb = mem_info.free / 1e9

        pynvml.nvmlShutdown()
        return name, round(total_gb, 2), round(free_gb, 2), True
    except Exception as e:
        logger.debug("pynvml detection failed: %s", e)
        return None, 0.0, 0.0, False


def _parse_nvidia_smi() -> tuple[str | None, float, float, bool]:
    """Fallback: Detect NVIDIA GPU via nvidia-smi text parsing."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                name = parts[0]
                total_gb = float(parts[1]) / 1024
                free_gb = float(parts[2]) / 1024
                return name, round(total_gb, 2), round(free_gb, 2), True
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None, 0.0, 0.0, False


def _detect_nvidia() -> tuple[str | None, float, float, bool]:
    """Primary NVIDIA detection router: tries pynvml, falls back to nvidia-smi."""
    name, total, free, has_cuda = _detect_nvidia_pynvml()
    if has_cuda:
        return name, total, free, has_cuda
    return _parse_nvidia_smi()


# ── AMD detection ────────────────────────────────────────────────────────────

def _read_amd_vram_from_sysfs() -> float | None:
    """Try to read AMD VRAM from /sys/class/drm/card*/device/mem_info_vram_total."""
    try:
        import glob
        for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            with open(path) as f:
                vram_bytes = int(f.read().strip())
                if vram_bytes > 0:
                    return vram_bytes / 1e9
    except (OSError, ValueError):
        pass
    return None


def _detect_amd_gpu() -> tuple[str | None, float, bool]:
    """
    Detect AMD GPU. Returns (name, vram_total_gb, has_rocm).

    Methods tried in order:
    1. rocm-smi (Linux, ROCm installed)
    2. lspci + sysfs (Linux, no ROCm needed)
    3. PowerShell WMI (Windows)
    """
    # Method 1: rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                for card_id, info in data.items():
                    if not isinstance(info, dict):
                        continue
                    if "card" in str(card_id).lower():
                        vram_bytes = int(info.get("VRAM Total Memory (B)", 0))
                        name = str(info.get("Card series", "AMD GPU"))
                        if vram_bytes > 0:
                            return name, vram_bytes / 1e9, True
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass

    # Method 2: lspci + sysfs (Linux)
    try:
        result = subprocess.run(
            ["lspci", "-v"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "VGA" in line and ("AMD" in line or "ATI" in line or "Radeon" in line):
                    name = line.split(":")[-1].strip()
                    vram = _read_amd_vram_from_sysfs()
                    if vram and vram > 0:
                        return name, vram, False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: Windows WMI
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterRAM | "
                    "ConvertTo-Json",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                controllers = json.loads(result.stdout)
                if isinstance(controllers, dict):
                    controllers = [controllers]
                if isinstance(controllers, list):
                    for c in controllers:
                        if not isinstance(c, dict):
                            continue
                        name = str(c.get("Name", ""))
                        if "AMD" in name or "Radeon" in name:
                            vram_bytes = int(c.get("AdapterRAM") or 0)
                            if vram_bytes > 0:
                                return name, vram_bytes / 1e9, False
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            pass

    return None, 0.0, False


# ── Intel detection ──────────────────────────────────────────────────────────

def _estimate_intel_vram(gpu_name: str) -> float:
    """Conservative VRAM estimate for Intel Arc based on model name."""
    name_upper = gpu_name.upper()
    if "A770" in name_upper:
        return 16.0
    if "A750" in name_upper:
        return 8.0
    if "A580" in name_upper:
        return 8.0
    if "A380" in name_upper:
        return 6.0
    return 4.0  # conservative floor for unknown Intel GPUs


def _detect_intel_gpu() -> tuple[str | None, float]:
    """
    Detect Intel Arc / Xe GPU. Returns (name, vram_total_gb).

    Methods tried in order:
    1. xpu-smi (Linux, Intel oneAPI installed)
    2. lspci (Linux)
    3. PowerShell WMI (Windows)
    """
    # Method 1: xpu-smi
    try:
        result = subprocess.run(
            ["xpu-smi", "discovery", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                for device in data.get("device_list", []):
                    if not isinstance(device, dict):
                        continue
                    vram_bytes = int(device.get("memory_physical_size_byte", 0))
                    if vram_bytes > 0:
                        return str(device.get("device_name", "Intel GPU")), vram_bytes / 1e9
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass

    # Method 2: lspci
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Intel" in line and ("Arc" in line or "Xe" in line):
                    name = line.split(":")[-1].strip()
                    return name, _estimate_intel_vram(name)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: Windows WMI
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterRAM | "
                    "ConvertTo-Json",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                controllers = json.loads(result.stdout)
                if isinstance(controllers, dict):
                    controllers = [controllers]
                if isinstance(controllers, list):
                    for c in controllers:
                        if not isinstance(c, dict):
                            continue
                        name = str(c.get("Name", ""))
                        if "Intel" in name and ("Arc" in name or "Xe" in name):
                            vram_bytes = int(c.get("AdapterRAM") or 0)
                            vram_gb = vram_bytes / 1e9 if vram_bytes > 0 else _estimate_intel_vram(name)
                            return name, vram_gb
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            pass

    return None, 0.0


# ── CPU feature detection ───────────────────────────────────────────────────

def _detect_cpu_features() -> list[str]:
    """
    Detect CPU ISA features relevant to LLM inference speed.

    Looks for: avx, avx2, avx512f, f16c, fma, neon, sve
    """
    features: list[str] = []
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read().lower()
            for feat in ("avx512f", "avx2", "avx", "f16c", "fma", "neon", "sve"):
                if feat in content:
                    features.append(feat)
        except OSError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-a"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                content = result.stdout.lower()
                if "hw.optional.avx2_0: 1" in content or "avx2" in content:
                    features.append("avx2")
                if "hw.optional.avx512" in content:
                    features.append("avx512f")
                if "hw.optional.neon" in content or platform.machine() == "arm64":
                    features.append("neon")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Apple Silicon always has NEON
        if platform.machine() == "arm64" and "neon" not in features:
            features.append("neon")
    elif system == "Windows":
        try:
            # Use environment variable first (fast)
            proc_id = platform.processor().lower()
            # Most modern x86 CPUs support AVX2; check processor name heuristics
            if proc_id:
                # Intel 4th gen+ and AMD Zen+ support AVX2
                if any(kw in proc_id for kw in ("core", "ryzen", "xeon", "epyc", "threadripper")):
                    features.append("avx2")
        except Exception:
            pass

    return features


# ── Apple Silicon detection ──────────────────────────────────────────────────

def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def _detect_apple_chip_name() -> str | None:
    """Try to get the specific Apple chip name (M1, M2, M3, M4 etc)."""
    if not _is_apple_silicon():
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Apple Silicon"


# ── Main detection orchestrator ──────────────────────────────────────────────

def detect_hardware_profile() -> HardwareProfile:
    """Detect full hardware profile: OS, CPU, RAM, disk, GPU (multi-vendor), bandwidth."""
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()
    cpu_name = platform.processor() or "Unknown"
    cpu_physical = psutil.cpu_count(logical=False) or 1
    cpu_logical = psutil.cpu_count(logical=True) or 1

    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 2)
    ram_available_gb = round(mem.available / (1024**3), 2)

    disk_root = "C:\\" if os_name == "Windows" else "/"
    disk = psutil.disk_usage(disk_root)
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    cpu_features = _detect_cpu_features()
    apple = _is_apple_silicon()

    # GPU detection — priority: NVIDIA → AMD → Intel → Apple Silicon
    gpu_name: str | None = None
    vram_total: float = 0.0
    vram_free: float = 0.0
    has_cuda: bool = False
    has_rocm: bool = False
    has_metal: bool = False
    gpu_vendor: str = "none"

    # Try NVIDIA first
    nv_name, nv_total, nv_free, nv_cuda = _detect_nvidia()
    if nv_name:
        gpu_name = nv_name
        vram_total = nv_total
        vram_free = nv_free
        has_cuda = nv_cuda
        gpu_vendor = "nvidia"
    else:
        # Try AMD
        amd_name, amd_vram, amd_rocm = _detect_amd_gpu()
        if amd_name:
            gpu_name = amd_name
            vram_total = amd_vram
            vram_free = amd_vram * 0.9  # estimate free as 90% of total
            has_rocm = amd_rocm
            gpu_vendor = "amd"
        else:
            # Try Intel
            intel_name, intel_vram = _detect_intel_gpu()
            if intel_name:
                gpu_name = intel_name
                vram_total = intel_vram
                vram_free = intel_vram * 0.9
                gpu_vendor = "intel"

    # Apple Silicon uses unified memory
    if apple and gpu_vendor == "none":
        chip_name = _detect_apple_chip_name()
        gpu_name = chip_name or "Apple Silicon"
        gpu_vendor = "apple"
        has_metal = True

    has_gpu = gpu_name is not None
    unified: float | None = None
    if apple:
        unified = ram_total_gb
        has_metal = True

    bandwidth = get_bandwidth(gpu_name or "")

    profile = HardwareProfile(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_free_gb=disk_free_gb,
        disk_total_gb=disk_total_gb,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_vram_total_gb=round(vram_total, 2),
        gpu_vram_free_gb=round(vram_free, 2),
        has_cuda=has_cuda,
        has_rocm=has_rocm,
        has_metal=has_metal,
        gpu_vendor=gpu_vendor,
        gpu_bandwidth_gb_s=bandwidth,
        cpu_features=cpu_features,
        is_apple_silicon=apple,
        unified_memory_gb=unified,
    )
    profile = enrich_hardware_budget(profile)

    # ── Multi-GPU topology (deferred import avoids circular) ──────────────
    try:
        from app.install.gpu_topology import detect_gpu_topology
        topology = detect_gpu_topology(
            cpu_cores=cpu_logical,
            ram_gb=ram_total_gb,
        )
        profile.all_gpus = topology.gpus
        profile.gpu_count = len(topology.gpus)
        profile.total_vram_gb = topology.pooled_vram_gb
        profile.topology_type = topology.interconnect
        profile.scale_tier = topology.scale_tier
    except Exception as e:
        logger.debug("Multi-GPU topology detection failed: %s", e)

    return profile


def enrich_hardware_budget(hw: HardwareProfile) -> HardwareProfile:
    """Attach compute budget fields used by feasibility + fusion scorer."""
    if hw.is_apple_silicon and hw.unified_memory_gb:
        eff_vram = max(hw.unified_memory_gb * 0.55, 0.5)
    elif hw.has_gpu:
        eff_vram = max(hw.gpu_vram_free_gb - 0.5, 0.25)
    else:
        eff_vram = 0.0

    eff_ram = max(hw.ram_available_gb, hw.ram_total_gb * 0.85)
    max_params = eff_vram / 0.6 if eff_vram > 0 else eff_ram / 1.2

    hw.effective_vram_gb = round(eff_vram, 2)
    hw.effective_ram_gb = round(eff_ram, 2)
    hw.max_model_params_b = round(max_params, 2)
    return hw
